# v0.8.93 (fork) - neural-rendering settings re-read while running, and host fixes

Fork release: [v0.8.92 (fork)](https://github.com/jlrouzies-fr/OptiScaler-DLSSNR-PreSR-Multipass/releases/tag/v0.8.92) plus six commits. Everything else is wilsjo2's work; see their releases for what v0.8.x does.

## New: `[DlssNr]` is re-read from OptiScaler.ini while the game runs

OptiScaler looks at the time stamp of `OptiScaler.ini` every 200 ms. When it has changed and then stood still for one look, the neural-rendering keys are read again and applied, with no restart: `Enabled`, `Preset`, `Style`, `Intensity`, `LocalStructure`, `LocalTone`, `SkinStructure`, `AutoMask`, `UnlockPasses`, `Passes`, `WorkingScale`, `TransferStrength`, `ColourStrength`, `MaxRatio`, `WhitePointScale`, `WhitePointSource`, `WhitePointTrim`, `AutoExposureTrim`, `AutoExposureHighlightProtection`, `SkinProtection`, `SkinDetail`, `SkinColour`, `EnvironmentDetail`, `EnvironmentColour`.

This is what lets another tool drive the neural pass. [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder)'s in-game menu writes the ini and recognises a build that does this by the text `DLSS-NR live settings` in the DLL. The log says `DLSS-NR live settings: read from OptiScaler.ini while running: <key>=<value>` for each key that changed. OptiScaler's own menu is unchanged and still applies its changes at once.

Only those keys are re-read. Every other section of the ini is read at start, as before.

## Fixes since v0.8.92

- **Neural pass follows the command list's device, not the output texture's.** A wrapper layer (ReShade's proxies) can give the output texture one identity for the game's device and the command list another; using the wrong one faulted inside the model (access violation in D3D12Core, or a hang with `RunBeforeSR`).
- **Finished picture with ReShade loaded through `[Plugins] LoadReshade`.** The Present queue was another wrapper of the game's device, so every Present returned early and nothing was composed. On a device mismatch with the same adapter, the work is submitted on the DLSS-side queue.
- **WinUI 3 hosts:** `[FrameGen] WrapCompositionSwapChain=false` leaves composition swap chains unwrapped (a wrapper handed to a SwapChainPanel crashed Elgato Studio), and `[Menu] InputHooks=false` runs the menu on polled input with no Win32 detours. Both default to the old behaviour. Neither key is in the shipped `OptiScaler.ini`: add the line under its section.
- **`SkipD3D11FeatureLevelElevation` quirk** (Metro / Luma), and no conflict with a dummy D3D12 device probe.
- **In-picture menu no longer rebuilt every other frame while the host is in the background.** That leaked a descriptor heap 30 times a second until the device refused new ones and the neural pass went off for the session. The menu now only ever attaches to a window of its own process, and a neural pass whose construction failed is tried again every 600 frames.

## What was and was not run

- Live settings: run on Direct3D 12 through DLSS5-Feeder's lab (a 32-bit game with the 64-bit helper, and a 64-bit game with OptiScaler in its own process): on/off, style, intensity, passes, working scale and the composition keys all took effect while running. **The native Vulkan path makes the same call but was not run.**
- The command-list device fix was verified in Kameo (through the feeder's helper). The WinUI 3 options are what Elgato Studio runs with.
- The in-picture menu fix builds but its behaviour in Elgato Studio was not re-checked after the change.
- Not run: RTX40 MFG. No RTX40-MFG package is provided for this release.

NVIDIA model / FG runtime DLLs are not bundled. Preserve game-specific INI settings when updating.
