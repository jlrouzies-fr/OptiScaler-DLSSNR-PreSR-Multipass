#pragma once

// Dynamic model resolution: steps the NR working scale between [MinScale, MaxScale] to hold a frame-rate target.
// Each step changes the model size, which resets NR history, so steps are coarse (5%) and sticky; a level that
// just went over budget is not retried for 20 s. Driven by the GPU time of the whole frame (FrameGpuTime), which
// still shows the margin under a frame cap: down above 92% of the budget, before a frame is late, up below 80%.
// Falls back to the present-to-present time (down above 105%, up at or below 102%) without GPU samples.

#include "Config.h"
#include "State.h"
#include "Util.h"
#include "DlssNr_FrameGpuTime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>

namespace DlssNr::DynamicScale
{
constexpr float Step = 0.05f;
constexpr float Floor = 0.05f;

struct Controller
{
    std::mutex mutex;
    bool active = false;
    int level = -1;              // steps above min
    double windowStart = 0.0;
    double sum = 0.0;
    unsigned count = 0;
    unsigned goodWindows = 0;
    double holdUntil = 0.0;      // no change before this time
    int blockedLevel = -1;       // level that missed the target
    double blockedUntil = 0.0;
    double lastReport = 0.0;
    std::atomic<float> current { 1.0f };
    std::atomic<float> averageMs { 0.0f };
    std::atomic<bool> usingGpu { false };
};

inline Controller& Instance()
{
    static Controller c;
    return c;
}

inline void Range(const Config& cfg, float& lo, float& hi)
{
    lo = cfg.DlssNrDynamicMinScale.value_or_default();
    hi = cfg.DlssNrDynamicMaxScale.value_or_default();
    lo = std::isfinite(lo) ? std::clamp(lo, Floor, 2.0f) : Floor;
    hi = std::isfinite(hi) ? std::clamp(hi, lo, 2.0f) : lo;
}

// Working scale for this frame: the static setting, or the controller's choice when dynamic mode is on.
inline float WorkScale(const Config& cfg)
{
    auto& c = Instance();
    if (!cfg.DlssNrDynamicScale.value_or_default())
    {
        c.active = false;
        float s = cfg.DlssNrWorkingScale.value_or_default();
        return std::isfinite(s) ? std::clamp(s, 0.25f, 2.0f) : 1.0f;
    }

    std::lock_guard lock(c.mutex);
    float lo, hi;
    Range(cfg, lo, hi);
    const int top = (int) std::floor((hi - lo) / Step + 0.001f);
    const double now = Util::MillisecondsNow();
    if (!c.active || c.level < 0)
    {
        c.active = true;
        c.level = top;
        c.windowStart = now;
        c.sum = 0.0;
        c.count = 0;
        c.goodWindows = 0;
        c.holdUntil = now + 2000.0;
        c.blockedLevel = -1;
    }
    c.level = std::clamp(c.level, 0, top);

    const double gpu = FrameGpuTime::Instance().Recent(now);
    const bool useGpu = gpu > 0.0 && gpu < 1000.0;
    const double ft = useGpu ? gpu : ::State::Instance().presentFrameTime;
    if (ft > 0.0 && ft < 1000.0)
    {
        c.sum += ft;
        ++c.count;
    }

    // GPU time is steady under a cap, so it can react within a few frames
    if (now - c.windowStart >= (useGpu ? 150.0 : 500.0) && c.count > 0)
    {
        float fps = cfg.DlssNrDynamicTargetFps.value_or_default();
        fps = std::isfinite(fps) ? std::clamp(fps, 10.0f, 500.0f) : 40.0f;
        const double target = 1000.0 / fps;
        const double avg = c.sum / c.count;
        c.averageMs = (float) avg;
        c.windowStart = now;
        c.sum = 0.0;
        c.count = 0;

        const double downAt = useGpu ? 0.92 : 1.05, upAt = useGpu ? 0.80 : 1.02;
        c.usingGpu = useGpu;
        if (now - c.lastReport >= 5000.0)
        {
            c.lastReport = now;
            LOG_INFO("DLSS-NR dynamic: {} {:.1f} ms (target {:.1f} ms), model resolution {:.0f}%",
                     useGpu ? "GPU" : "frame (no GPU timer)", avg, target, (lo + c.level * Step) * 100.0f);
        }
        if (avg > target * downAt)
        {
            c.goodWindows = 0;
            if (now >= c.holdUntil && c.level > 0)
            {
                c.blockedLevel = c.level;
                c.blockedUntil = now + 20000.0;
                --c.level;
                c.holdUntil = now + (useGpu ? 400.0 : 1000.0);
                LOG_INFO("DLSS-NR dynamic: {} {:.1f} ms over {:.0f}% of {:.1f} ms, model resolution down to {:.0f}%",
                         useGpu ? "GPU" : "frame", avg, downAt * 100.0, target, (lo + c.level * Step) * 100.0f);
            }
        }
        else if (avg <= target * upAt)
        {
            ++c.goodWindows;
            const bool blocked = c.level + 1 == c.blockedLevel && now < c.blockedUntil;
            if (c.goodWindows >= (useGpu ? 13u : 4u) && now >= c.holdUntil && c.level < top && !blocked)
            {
                ++c.level;
                c.goodWindows = 0;
                c.holdUntil = now + 2000.0;
                LOG_INFO("DLSS-NR dynamic: {} {:.1f} ms, model resolution up to {:.0f}%", useGpu ? "GPU" : "frame", avg,
                         (lo + c.level * Step) * 100.0f);
            }
        }
        else
            c.goodWindows = 0;
    }

    const float scale = std::min(hi, lo + c.level * Step);
    c.current = scale;
    return scale;
}
} // namespace DlssNr::DynamicScale
