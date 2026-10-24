#pragma once

// Dynamic model resolution: steps the NR working scale between [MinScale, MaxScale] to hold a frame-rate target.
// Each step changes the model size, which resets NR history, so steps are coarse (5%), slow and sticky:
// one step down after 0.5 s over target, one step up after 2 s on target, and a level that just missed the
// target is not retried for 20 s. Driven by the present-to-present time OptiScaler already measures.

#include "Config.h"
#include "State.h"
#include "Util.h"

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
    std::atomic<float> current { 1.0f };
    std::atomic<float> averageMs { 0.0f };
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

    const double ft = ::State::Instance().presentFrameTime;
    if (ft > 0.0 && ft < 1000.0)
    {
        c.sum += ft;
        ++c.count;
    }

    if (now - c.windowStart >= 500.0 && c.count > 0)
    {
        float fps = cfg.DlssNrDynamicTargetFps.value_or_default();
        fps = std::isfinite(fps) ? std::clamp(fps, 10.0f, 500.0f) : 40.0f;
        const double target = 1000.0 / fps;
        const double avg = c.sum / c.count;
        c.averageMs = (float) avg;
        c.windowStart = now;
        c.sum = 0.0;
        c.count = 0;

        if (avg > target * 1.05)
        {
            c.goodWindows = 0;
            if (now >= c.holdUntil && c.level > 0)
            {
                c.blockedLevel = c.level;
                c.blockedUntil = now + 20000.0;
                --c.level;
                c.holdUntil = now + 1000.0;
                LOG_INFO("DLSS-NR dynamic: {:.1f} ms over {:.1f} ms target, model resolution down to {:.0f}%", avg,
                         target, (lo + c.level * Step) * 100.0f);
            }
        }
        else if (avg <= target * 1.02)
        {
            ++c.goodWindows;
            const bool blocked = c.level + 1 == c.blockedLevel && now < c.blockedUntil;
            if (c.goodWindows >= 4 && now >= c.holdUntil && c.level < top && !blocked)
            {
                ++c.level;
                c.goodWindows = 0;
                c.holdUntil = now + 2000.0;
                LOG_INFO("DLSS-NR dynamic: on target, model resolution up to {:.0f}%", (lo + c.level * Step) * 100.0f);
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
