#pragma once

// GPU busy time of a frame: a timestamp before and after every submission on a direct queue, summed as the union
// of those intervals between two Presents. Idle gaps (waiting on the CPU or a frame cap) are left out, so under a
// cap this still shows how much of the budget the GPU really used, and the dynamic model resolution can step down
// before a frame is late. Async compute overlapping the direct queue is not counted separately.

#include <Util.h>

#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

namespace DlssNr::FrameGpuTime
{
using ExecuteFn = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

class Timer
{
    static constexpr unsigned Slots = 8;   // frames in flight (other queues may finish after the present fence)
    static constexpr unsigned Stamps = 512; // per frame: 256 submissions
    struct Slot
    {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> alloc;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> resolve;
        std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>> stamps; // one prebuilt list per query
        UINT64 fenceValue = 0;
        unsigned used = 0;
        bool pending = false;
    };

    std::mutex mutex;
    ID3D12Device* device = nullptr; // identity
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> queries;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    std::array<Slot, Slots> slots;
    UINT64 frequency = 0, nextFence = 0;
    int current = -1; // slot collecting this frame
    bool failed = false;

    bool Init(ID3D12CommandQueue* q)
    {
        Microsoft::WRL::ComPtr<ID3D12Device> dev;
        if (FAILED(q->GetDevice(IID_PPV_ARGS(&dev))) || FAILED(q->GetTimestampFrequency(&frequency)))
            return false;
        D3D12_QUERY_HEAP_DESC hd { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Slots * Stamps, 0 };
        D3D12_HEAP_PROPERTIES hp { D3D12_HEAP_TYPE_READBACK };
        D3D12_RESOURCE_DESC rd {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = Slots * Stamps * sizeof(UINT64);
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(dev->CreateQueryHeap(&hd, IID_PPV_ARGS(&queries))) ||
            FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&readback))) ||
            FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return false;
        for (unsigned s = 0; s < Slots; ++s)
        {
            auto& slot = slots[s];
            if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot.alloc))))
                return false;
            // Prebuilt and never reset: each list only writes its own query, so it can be re-executed every frame.
            slot.stamps.resize(Stamps);
            for (unsigned k = 0; k < Stamps; ++k)
            {
                auto& list = slot.stamps[k];
                if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.alloc.Get(), nullptr,
                                                  IID_PPV_ARGS(&list))))
                    return false;
                list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, s * Stamps + k);
                list->Close();
            }
            if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.alloc.Get(), nullptr,
                                              IID_PPV_ARGS(&slot.resolve))))
                return false;
            slot.resolve->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, s * Stamps, Stamps,
                                           readback.Get(), s * Stamps * sizeof(UINT64));
            slot.resolve->Close();
        }
        return true;
    }

    void Collect()
    {
        const UINT64 done = fence->GetCompletedValue();
        if (done == UINT64_MAX)
            return;
        for (unsigned s = 0; s < Slots; ++s)
        {
            auto& slot = slots[s];
            if (!slot.pending || done < slot.fenceValue)
                continue;
            slot.pending = false;
            UINT64* data = nullptr;
            D3D12_RANGE range { s * Stamps * sizeof(UINT64), (s * Stamps + slot.used) * sizeof(UINT64) };
            if (slot.used < 2 || FAILED(readback->Map(0, &range, (void**) &data)))
                continue;
            std::vector<std::pair<UINT64, UINT64>> spans;
            spans.reserve(slot.used / 2);
            for (unsigned k = 0; k + 1 < slot.used; k += 2)
            {
                const UINT64 b = data[s * Stamps + k], e = data[s * Stamps + k + 1];
                if (b && e > b)
                    spans.emplace_back(b, e);
            }
            D3D12_RANGE none { 0, 0 };
            readback->Unmap(0, &none);
            if (spans.empty())
                continue;
            std::sort(spans.begin(), spans.end());
            UINT64 busy = 0, curB = spans[0].first, curE = spans[0].second;
            for (size_t i = 1; i < spans.size(); ++i)
            {
                if (spans[i].first <= curE)
                    curE = std::max(curE, spans[i].second);
                else
                {
                    busy += curE - curB;
                    curB = spans[i].first;
                    curE = spans[i].second;
                }
            }
            busy += curE - curB;
            lastMs = double(busy) * 1000.0 / double(frequency);
            lastAt = Util::MillisecondsNow();
        }
    }

    // Our lists may only run on queues of the device that created them (games can create several devices).
    bool SameDevice(ID3D12CommandQueue* q)
    {
        thread_local ID3D12CommandQueue* lastQueue = nullptr;
        thread_local ID3D12Device* lastDevice = nullptr;
        thread_local bool lastSame = false;
        if (q == lastQueue && device == lastDevice)
            return lastSame;
        ID3D12Device* dev = nullptr;
        lastSame = SUCCEEDED(q->GetDevice(IID_PPV_ARGS(&dev))) && dev == device;
        if (dev)
            dev->Release();
        lastQueue = q;
        lastDevice = device;
        return lastSame;
    }

  public:
    std::atomic<double> lastMs { 0.0 };
    std::atomic<double> lastAt { 0.0 };

    // Replaces the real ExecuteCommandLists for the game's lists: stamps before and after on direct queues.
    void Execute(ID3D12CommandQueue* q, UINT count, ID3D12CommandList* const* lists, ExecuteFn execute)
    {
        ID3D12CommandList* before = nullptr;
        ID3D12CommandList* after = nullptr;
        if (q != nullptr && device != nullptr && q->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT &&
            SameDevice(q))
        {
            std::lock_guard lock(mutex);
            if (!failed && current >= 0)
            {
                auto& slot = slots[current];
                if (slot.used + 2 <= Stamps)
                {
                    before = slot.stamps[slot.used].Get();
                    after = slot.stamps[slot.used + 1].Get();
                    slot.used += 2;
                }
            }
        }
        if (before)
            execute(q, 1, &before);
        execute(q, count, lists);
        if (after)
            execute(q, 1, &after);
    }

    // Before Present on the swapchain's queue: close this frame's slot and open the next one.
    void BeforePresent(ID3D12CommandQueue* q, ExecuteFn execute)
    {
        std::lock_guard lock(mutex);
        if (failed || q == nullptr || execute == nullptr)
            return;
        Microsoft::WRL::ComPtr<ID3D12Device> dev;
        if (FAILED(q->GetDevice(IID_PPV_ARGS(&dev))))
            return;
        if (dev.Get() != device)
        {
            for (auto& s : slots)
            {
                s.pending = false;
                s.used = 0;
                s.stamps.clear();
                s.resolve.Reset();
                s.alloc.Reset();
            }
            queries.Reset();
            readback.Reset();
            fence.Reset();
            current = -1;
            device = dev.Get();
            if (!Init(q))
            {
                failed = true;
                LOG_WARN("DLSS-NR dynamic: GPU busy timer unavailable, using frame time");
                return;
            }
        }
        if (current >= 0)
        {
            auto& slot = slots[current];
            ID3D12CommandList* list = slot.resolve.Get();
            execute(q, 1, &list);
            slot.fenceValue = ++nextFence;
            slot.pending = SUCCEEDED(q->Signal(fence.Get(), slot.fenceValue));
            current = -1;
        }
        Collect();
        for (unsigned s = 0; s < Slots; ++s)
            if (!slots[s].pending)
            {
                slots[s].used = 0;
                current = (int) s;
                break;
            }
    }

    // Latest GPU busy time of a frame in ms, or 0 when there is no recent sample.
    double Recent(double now) const
    {
        const double at = lastAt.load();
        return at > 0.0 && now - at < 250.0 ? lastMs.load() : 0.0;
    }
};

inline Timer& Instance()
{
    static Timer t;
    return t;
}

// The real ExecuteCommandLists, set by the queue hook; null until NR hooks the queue.
inline std::atomic<ExecuteFn>& RealExecute()
{
    static std::atomic<ExecuteFn> fn { nullptr };
    return fn;
}
} // namespace DlssNr::FrameGpuTime
