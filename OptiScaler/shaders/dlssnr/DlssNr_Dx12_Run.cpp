#include "pch.h"
#include "DlssNr_Dx12_State.h"

auto DlssNr_Dx12::State::Run(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* colour, ID3D12Resource* depth, ID3D12Resource* motion,
             ID3D12Resource* output, const DlssNrFrameInfo& frame, ID3D12CommandQueue* timingQueue) -> void
{
    std::lock_guard<std::recursive_mutex> nrLock(mutex);
    const Config& cfg = *Config::Instance();

    if (nr.failed || cmdList == nullptr || colour == nullptr || depth == nullptr || motion == nullptr ||
        output == nullptr)
    {
        ReportSkipOnce(nr.failed ? "it already failed this session" : "a resource was missing");
        return;
    }

    ID3D12Resource* target = output;

    // Guard creation and dispatch together: either can record GPU work and alter compute bindings.
    const bool restoreRequired =
        cfg.RestoreComputeSignature.value_or_default() || cfg.RestoreGraphicSignature.value_or_default();
    if (restoreRequired && !frame.IndependentCommands && !D3D12Hooks::CanRestoreRootSignature(cmdList))
    {
        ReportSkipOnce("the upscaler could not restore state this frame");
        return;
    }
    lifetime.Record(cmdList);
    ScopedNrStateEnvelope stateEnvelope(cmdList);

    // A completed upscaler output normally arrives as a UAV. The pre-SR colour input instead arrives
    // readable. Track every transition so both paths return the resource exactly as their caller gave
    // it to us; a pre-SR resource without UAV support is written through a scratch-and-copy fallback.
    const D3D12_RESOURCE_STATES outputArrival =
        frame.PipelineManagedStates ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        : frame.FinishedPicture     ? (D3D12_RESOURCE_STATES) frame.OutputArrivalState
        : frame.BeforeUpscale ? (!frame.PrivateColorCopy && Config::Instance()->ColorResourceBarrier.has_value()
                                     ? (D3D12_RESOURCE_STATES) Config::Instance()->ColorResourceBarrier.value()
                                     : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        : Config::Instance()->OutputResourceBarrier.has_value()
            ? (D3D12_RESOURCE_STATES) Config::Instance()->OutputResourceBarrier.value()
            : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES targetState = outputArrival;
    const auto TransitionTarget = [&](D3D12_RESOURCE_STATES to)
    {
        Barrier(cmdList, target, targetState, to);
        targetState = to;
    };

    Microsoft::WRL::ComPtr<ID3D12Device> deviceRef;

    if (FAILED(target->GetDevice(IID_PPV_ARGS(&deviceRef))))
    {
        ReportSkipOnce("the output texture belongs to no D3D12 device");
        return;
    }

    auto* device = deviceRef.Get();
    const D3D12_RESOURCE_DESC desc = target->GetDesc();
    const auto active =
        frame.BeforeUpscale
            ? DlssNr::PreSrColorExtent(desc, frame.RenderSubrectWidth, frame.RenderSubrectHeight)
            : std::optional<DlssNr::ColorExtent> { DlssNr::ColorExtent { (unsigned int) desc.Width, desc.Height } };
    if (!active)
    {
        ReportSkipOnce("the pre-SR active colour size is invalid");
        return;
    }
    const auto width = active->width;
    const auto height = active->height;
    const bool cropColor = frame.BeforeUpscale && (width != desc.Width || height != desc.Height);
    const bool targetSupportsUav = cropColor || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;

    const auto guideDesc = depth->GetDesc();
    const auto motionDesc = motion->GetDesc();
    const auto guides = DlssNr::ResolveGuideRegions(
        { (unsigned int) guideDesc.Width, guideDesc.Height },
        { (unsigned int) motionDesc.Width, motionDesc.Height },
        { frame.RenderSubrectWidth, frame.RenderSubrectHeight }, { frame.OutputWidth, frame.OutputHeight },
        frame.MotionVectorsLowResolution, frame.DepthSubrectBaseX, frame.DepthSubrectBaseY,
        frame.MotionSubrectBaseX, frame.MotionSubrectBaseY);
    if (!guides.depth.valid() || !guides.motion.valid())
    {
        ReportSkipOnce("depth or motion-vector subrect is empty");
        return;
    }
    const auto guideWidth = guides.depth.width, guideHeight = guides.depth.height;

    if (frame.Reset)
    {
        nr.reset = true;

        ++resets;

        if (resets <= 3 || resets % 100 == 0)
            LOG_INFO("DLSS-NR: the game asked for a history reset ({} so far)", resets);
    }

    // Guide dimensions can change without rebuilding the model; report changes as they occur.

    const GuideReport guidesNow {
        true,  frame.DepthInverted, frame.MvScaleX, frame.MvScaleY, guideWidth, guideHeight,
        width, (unsigned int) height
    };

    if (loggedGuides != guidesNow)
    {
        loggedGuides = guidesNow;
        LOG_INFO("DLSS-NR guides: depth {}, motion vector scale {} x {}, guides {}x{} for a {}x{} frame",
                 frame.DepthInverted ? "inverted" : "not inverted", frame.MvScaleX, frame.MvScaleY,
                 guideWidth, guideHeight, width, height);
    }

    const unsigned int requestedPasses =
        std::clamp(cfg.DlssNrPasses.value_or_default(), 1u,
                   cfg.DlssNrUnlockPasses.value_or_default() ? DlssNr::MaxPassCount : DlssNr::DefaultMaxPassCount);
    for (auto& model : nr.models)
        model.Collect();
    if ((!NVNGXProxy::IsDx12Inited() && !NVNGXProxy::InitDx12(device)) || !DlssNr::Proxy::Context::Available())
    {
        nr.failed = true;
        nr.reason = "the NVIDIA NGX driver does not provide Neural Rendering";
        LOG_ERROR("DLSS-NR unavailable: {}", nr.reason);
        return;
    }

    // Only the model runs at working resolution; source and composition remain at native size.
    float workScale = cfg.DlssNrWorkingScale.value_or_default();
    if (!std::isfinite(workScale))
        workScale = 1.0f;
    workScale = std::clamp(workScale, 0.25f, 2.0f);
    const auto workWidth = (unsigned int) (width * workScale + 0.5f);
    const auto workHeight = (unsigned int) (height * workScale + 0.5f);
    const bool reduced = workWidth != width || workHeight != height;
    if (!PrepareRunModels(cmdList, device, frame, desc, { width, height }, { workWidth, workHeight },
                          workScale, requestedPasses))
        return;
    // The parameter adapter already combined the HDR flag with the active color format.
    const bool isHdrBuffer = frame.ColourIsLinearHdr;

    if (!reportedHdr || reportedHdrValue != isHdrBuffer || reportedBefore != frame.BeforeUpscale)
    {
        reportedHdr = true;
        reportedHdrValue = isHdrBuffer;
        reportedBefore = frame.BeforeUpscale;
        LOG_INFO("DLSS-NR {} SR: the game's DLSS colour space is {} so the colour transform is {}",
                 frame.BeforeUpscale ? "before" : "after", isHdrBuffer ? "linear HDR" : "already tone-mapped",
                 isHdrBuffer ? "on" : "off");
    }

    if (!shader.IsInit())
    {
        nr.failed = true;
        nr.reason = "the colour codec would not compile";
        LOG_ERROR("DLSS-NR unavailable: {}", nr.reason);
        return;
    }

    // Advance capture scheduling only once the codec and models are ready.
    ++frames;
    lifetime.Collect();
    CheckCaptureTrigger();

    if (captureFrames.isActive())
    {
        const auto captureDir = Util::DllPath().remove_filename() / "dlssnr-capture";
        const auto written = captureFrames.write(captureDir);

        if (!written.empty())
            LOG_INFO("DLSS-NR wrote matched before/after frames to {}", written);
    }

    ResTrack_Dx12::HookLateNrQueue(device);
    if (gpuTime == nullptr)
        gpuTime = std::make_unique<DlssNrGpuTime>(device);

    if (ngxTime == nullptr)
        ngxTime = std::make_unique<DlssNrGpuTime>(device);

    gpuTime->Start(cmdList);

    // Copy just the live image, not the stale right/bottom margins. Do this only after model
    // creation/pending-submission early returns, and inside the measured GPU interval. The compact
    // texture lets every existing codec/compare/hold/capture path use unmodified pixel coordinates.
    ID3D12Resource* const gameColor = target;
    if (cropColor)
    {
        TransitionTarget(D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cmdList, nr.activeColor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        DlssNr::CopyActiveColor(cmdList, nr.activeColor, gameColor, *active);
        TransitionTarget(outputArrival);
        Barrier(cmdList, nr.activeColor, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        target = nr.activeColor;
        targetState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    const auto FinishColor = [&](bool copyBack)
    {
        if (cropColor)
        {
            if (copyBack)
            {
                TransitionTarget(D3D12_RESOURCE_STATE_COPY_SOURCE);
                Barrier(cmdList, gameColor, outputArrival, D3D12_RESOURCE_STATE_COPY_DEST);
                DlssNr::CopyActiveColor(cmdList, gameColor, target, *active);
                Barrier(cmdList, gameColor, D3D12_RESOURCE_STATE_COPY_DEST, outputArrival);
            }
            TransitionTarget(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else
        {
            TransitionTarget(outputArrival);
        }
    };

    EncodeContext encoded { cmdList, device, target, targetState, frame, workScale, targetSupportsUav };
    EncodeInput(encoded);
    targetState = encoded.targetState;
    auto* modelInput = encoded.modelInput;

    ID3D12Resource* depthIn = ReadableGuide(device, cmdList, depth, &nr.depthClone);
    ID3D12Resource* motionIn = ReadableGuide(device, cmdList, motion, &nr.motionClone);

    if (depthIn == nullptr || motionIn == nullptr)
    {
        nr.reset = true;
        ReportSkipOnce("the game's depth or motion vectors could not be made readable this frame");
        FinishColor(false);
        return;
    }

    // The vectors were scaled to full-frame pixels; the image the model reprojects is the working size.
    const float mvToWorkX = width != 0 ? (float) workWidth / (float) width : 1.0f;
    const float mvToWorkY = height != 0 ? (float) workHeight / (float) height : 1.0f;

    // Below native, the model was handed a colour at the working size and depth and motion at the
    // frame's size, with only the vector magnitudes rescaled. Nothing says the model resamples a
    // guide that is larger than its colour, and the picture said it does not: every scale below
    // 100% flickered and settled for frames after the camera stopped, while the same model at
    // 100% of a frame the game had already shrunk was steady. So give it guides at its own size --
    // the same point resample the DLSS-enlargement path already builds for its private upscaler --
    // and describe them as a full, zero-origin region. The vectors keep the game's units; the
    // working-size scale below still applies.
    ID3D12Resource* modelDepth = depthIn;
    ID3D12Resource* modelMotion = motionIn;
    auto modelGuides = guides;
    bool matchedGuides = false;
    if (reduced && workWidth < width && cfg.DlssNrMatchGuides.value_or_default())
    {
        if (nr.depthSmall == nullptr)
            nr.depthSmall = CreateScratch(device, DXGI_FORMAT_R32_FLOAT, workWidth, workHeight);
        if (nr.motionSmall == nullptr)
            nr.motionSmall = CreateScratch(device, DXGI_FORMAT_R32G32_FLOAT, workWidth, workHeight);
        if (nr.depthSmall != nullptr && nr.motionSmall != nullptr)
        {
            DlssNrConstants resize {};
            resize.Mode = DlssNrMode_ResizePrivateGuides;
            resize.Width = workWidth;
            resize.Height = workHeight;
            resize.GuideWidth = guides.depth.width;
            resize.GuideHeight = guides.depth.height;
            resize.DebugView = guides.depth.x;
            resize.CompareMode = guides.depth.y;
            resize.TransferStrength = float(guides.motion.width);
            resize.ColourStrength = float(guides.motion.height);
            resize.CompareSwap = guides.motion.x;
            resize.Transfer = guides.motion.y;
            resize.MvScaleX = resize.MvScaleY = 1.0f;
            if (shader.DispatchPass(cmdList, resize, depthIn, motionIn, nullptr, nullptr, nullptr, nr.depthSmall,
                                    nr.motionSmall))
            {
                Barrier(cmdList, nr.depthSmall, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(cmdList, nr.motionSmall, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                modelDepth = nr.depthSmall;
                modelMotion = nr.motionSmall;
                modelGuides.depth = { 0, 0, workWidth, workHeight };
                modelGuides.motion = { 0, 0, workWidth, workHeight };
                matchedGuides = true;
            }
        }
        static bool loggedMatch = false;
        if (!loggedMatch)
        {
            loggedMatch = true;
            if (matchedGuides)
                LOG_INFO("DLSS-NR guides matched to the working size: depth and motion {}x{} for a {}x{} model "
                         "(the frame's guides are {}x{})",
                         workWidth, workHeight, workWidth, workHeight, guides.depth.width, guides.depth.height);
            else
                LOG_WARN("DLSS-NR guides could not be matched to the working size; the model keeps the frame's "
                         "{}x{} guides for its {}x{} colour",
                         guides.depth.width, guides.depth.height, workWidth, workHeight);
        }
    }

    ngxTime->Start(cmdList);

    // Count only a contiguous set of ready, separate feature histories. A failed extra creation never
    // falls back to reusing the main feature: that tells one temporal model several frames elapsed in
    // one game frame and makes its history fight the later layers.
    unsigned int effectivePasses = 1;
    if (nr.passScratch != nullptr)
    {
        for (unsigned int pass = 1; pass < requestedPasses; ++pass)
        {
            if (!nr.models[pass].Ready(frame.SubmissionEpoch))
                break;
            ++effectivePasses;
        }
    }

    if (loggedConfigured != requestedPasses || loggedEffective != effectivePasses)
    {
        loggedConfigured = requestedPasses;
        loggedEffective = effectivePasses;
        LOG_INFO("DLSS-NR model passes: configured {}, effective {}", requestedPasses, effectivePasses);
    }

    // Keep the encoded base immutable; ping-pong model outputs and compose the final delta once.
    ID3D12Resource* passInput = modelInput;
    ID3D12Resource* passOutput = nr.output;
    ID3D12Resource* finalAnswer = nullptr;
    bool outputReadable = false;
    bool scratchReadable = false;
    bool clampReadable = false;
    bool clampFailed = false;
    uint32_t clampSlots[2] = { UINT32_MAX, UINT32_MAX };

    const auto MakeModelReadable = [&](ID3D12Resource* resource)
    {
        bool& readable = resource == nr.output      ? outputReadable
                         : resource == nr.passClamp ? clampReadable
                                                    : scratchReadable;
        if (!readable)
        {
            Barrier(cmdList, resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            readable = true;
        }
    };

    const auto MakeModelWritable = [&](ID3D12Resource* resource)
    {
        bool& readable = resource == nr.output      ? outputReadable
                         : resource == nr.passClamp ? clampReadable
                                                    : scratchReadable;
        if (readable)
        {
            Barrier(cmdList, resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            readable = false;
        }
    };

    int result = NVSDK_NGX_Result_Success;
    const bool enlargementReset = nr.reset;
    bool compositionSucceeded = false;

    DlssNr::Proxy::Frame modelFrame {};
    modelFrame.depth = modelDepth;
    modelFrame.motion = modelMotion;
    modelFrame.size = { workWidth, workHeight };
    modelFrame.guides = modelGuides;
    modelFrame.depthInverted = frame.DepthInverted;
    modelFrame.reset = nr.reset;
    modelFrame.mvScaleX = frame.MvScaleX * mvToWorkX;
    modelFrame.mvScaleY = frame.MvScaleY * mvToWorkY;

    for (unsigned int pass = 0; pass < effectivePasses && result == NVSDK_NGX_Result_Success; ++pass)
    {
        MakeModelWritable(passOutput);
        bool evaluated = false;
        modelFrame.color = passInput;
        modelFrame.output = passOutput;
        result = static_cast<int>(nr.models[pass].Run(cmdList, device, modelFrame, PassSettings(cfg, pass),
                                                     frame.SubmissionEpoch, &evaluated));
        modelRunning = evaluated && result == NVSDK_NGX_Result_Success;
        if (!evaluated || result != NVSDK_NGX_Result_Success)
            break;

        finalAnswer = passOutput;
        MakeModelReadable(finalAnswer);

        if (pass + 1 < effectivePasses)
        {
            MakeModelWritable(nr.passClamp);
            DlssNrConstants clamp {};
            clamp.Mode = DlssNrMode_ClampProxy;
            clamp.Width = workWidth;
            clamp.Height = workHeight;
            if (!shader.DispatchPass(cmdList, clamp, finalAnswer, nullptr, nullptr, nullptr, nullptr, nr.passClamp,
                                     nullptr, &clampSlots[pass % 2]))
            {
                // Keep this frame's last valid answer; later histories skipped a frame.
                clampFailed = true;
                effectivePasses = pass + 1;
                break;
            }
            MakeModelReadable(nr.passClamp);
            passInput = nr.passClamp;
            passOutput = passOutput == nr.output ? nr.passScratch : nr.output;
        }
    }

    ngxTime->End(cmdList);

    nr.reset = clampFailed || finalAnswer == nullptr;

    // Supersampling probe: report the model working ABOVE native so a test log tells us whether NGX even
    // accepts a super-native evaluate and what it returns. Once per working-size change, or on any error.
    if (workWidth > width || workHeight > height)
    {

        if (lastSuper != workWidth || result != 1)
        {
            lastSuper = workWidth;
            LOG_INFO("DLSS-NR SUPERSAMPLE: model at {}x{} = {:.2f}x native {}x{}, evaluate result {} ({})",
                     workWidth, workHeight, (float) workWidth / (float) width, width, height, result,
                     NgxResultName((unsigned int) result));
        }
    }

    if (result == NVSDK_NGX_Result_Success && finalAnswer != nullptr)
    {
        // Resolve takes the difference between what the model returned and what it was shown, and adds
        // that back to the frame. At strength zero the result is what the upscaler produced, exactly, and
        // anything the model left alone is untouched rather than round-tripped through the curve.
        auto resolveParams = MakeResolveConstants(encoded, effectivePasses);

        // Downsample the model answer to native for composition; fall back to the working-size pair.
        // The final answer is NPSR and the native output rests in UAV.
        bool superDownOk = false;
        if (workScale > 1.0f && nr.superDown != nullptr && nr.outputNative != nullptr &&
            nr.superDown->DispatchResources(cmdList, finalAnswer, nr.outputNative))
        {
            Barrier(cmdList, nr.outputNative, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            superDownOk = true;
        }

        ID3D12Resource* resolveProxy = superDownOk ? nr.colorCopy : modelInput;
        ID3D12Resource* resolveAnswer = superDownOk ? nr.outputNative : finalAnswer;
        bool enlargementReady = true;
        const auto transfer = cfg.DlssNrTransfer.value_or_default();
        bool resizeFieldReadable = false;
        if (DlssNrUsesDlssEnlargement(transfer) && reduced && (transfer == 2 || workScale < 1.0f))
        {
            auto* enlarged = EnlargeMatchedResidual(cmdList, device, modelInput, finalAnswer, depthIn, motionIn,
                                                    frame, resolveParams, enlargementReset, timingQueue);
            enlargementReady = enlarged != nullptr;
            if (enlarged) { resolveAnswer = enlarged; resolveParams.Transfer = transfer; }
            if (enlarged && transfer == 4)
            {
                // Retain the spatial field for the inverse-HDR range guard.
                resolveProxy = enlarger->input.Get();
                Barrier(cmdList, resolveProxy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                resizeFieldReadable = true;
            }
            if (enlarged && (resolveParams.DebugView == 2 || (transfer == 4 && resolveParams.DebugView == 1)))
            {
                resolveProxy = modelInput; resolveAnswer = finalAnswer;
                resolveParams.Transfer = DlssNrSpatialTransfer(transfer); // Inspect the actual model pair.
            }
        }
        else
        {
            ReleaseEnlarger();
            enlargementStatus.clear();
        }

        // Resolve pre-SR inputs without UAV support through an owned scratch and copy-back.
        ID3D12Resource* resolveOriginal = targetSupportsUav ? nr.hdrCopy : target;
        ID3D12Resource* resolveTarget =
            targetSupportsUav ? target : nr.hdrCopy;

        if (targetSupportsUav)
        {
            TransitionTarget(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else
        {
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        const bool resolved =
            enlargementReady && shader.DispatchPass(cmdList, resolveParams, resolveProxy, resolveAnswer,
                                                    resolveOriginal, encoded.exposure, nullptr, resolveTarget, nullptr);
        compositionSucceeded = resolved;
        if (resizeFieldReadable)
            Barrier(cmdList, enlarger->input.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (resolved && !targetSupportsUav)
        {
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            const D3D12_RESOURCE_STATES priorTargetState = targetState;
            TransitionTarget(D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->CopyResource(target, nr.hdrCopy);
            TransitionTarget(priorTargetState);
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (!targetSupportsUav)
        {
            // The resolve target was made writable even while private DLSS was
            // warming up. Restore it before the common end-of-frame transition.
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        if (superDownOk)
            Barrier(cmdList, nr.outputNative, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // Schedule matched proxy/output capture for delayed readback.
        if (captureFrames.isActive())
        {
            captureFrames.record(cmdList, device, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 target, targetState);

        }
    }
    else if (result != NVSDK_NGX_Result_Success)
    {
        nr.failed = true;
        nr.reason = "the model refused to run";

        LOG_ERROR("DLSS-NR evaluate returned 0x{:X} ({}); use Retry to recreate the model", (uint32_t) result,
                  NgxResultName((unsigned int) result));
    }

    // Restore all intermediate surfaces to the UAV state expected by the next frame.
    MakeModelWritable(nr.output);
    if (nr.passScratch != nullptr)
        MakeModelWritable(nr.passScratch);

    Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    if (nr.passClamp != nullptr)
        MakeModelWritable(nr.passClamp);

    // Failed evaluations leave the game's original image intact. A successful copy-back writes
    // only the active rectangle and restores both resources before DLSS consumes the image.
    FinishColor(compositionSucceeded);
    if (compositionSucceeded)
        ++nr.successfulDispatches;

    EndGpuTiming(cmdList);

    // Restore guide clones to COPY_DEST for the next frame's refresh.
    if (depthIn == nr.depthClone)
        Barrier(cmdList, nr.depthClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST);

    if (motionIn == nr.motionClone)
        Barrier(cmdList, nr.motionClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST);

    if (reduced && nr.colorSmall != nullptr)
        Barrier(cmdList, nr.colorSmall, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // The matched guides were written this frame and read by the model; back to UAV for the next.
    if (matchedGuides)
    {
        Barrier(cmdList, nr.depthSmall, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmdList, nr.motionSmall, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // Leave the staging copy as the next frame expects to find it.
    Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
