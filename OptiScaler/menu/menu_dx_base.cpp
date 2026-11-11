#include "pch.h"

#include <Config.h>
#include <Logger.h>
#include <resource.h>
#include <Util.h>

#include "menu_common.h"
#include "menu_dx_base.h"

#include <imgui/imgui_impl_win32.h>

bool MenuDxBase::RenderMenu()
{
    if (Config::Instance()->OverlayMenu.value_or_default())
        return false;

    if (MenuCommon::RenderMenu())
    {
        ImGui::Render();
        return true;
    }

    return false;
}

static bool IsOwnLiveWindow(HWND hwnd)
{
    if (hwnd == nullptr || !IsWindow(hwnd))
        return false;

    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    return pid == GetCurrentProcessId();
}

HWND MenuDxBase::ResolveHostWindow()
{
    HWND hwnd = Util::GetProcessWindow();
    return IsOwnLiveWindow(hwnd) ? hwnd : nullptr;
}

// This used to be IsHandleDifferent(): "is the process window still the one the menu was made
// for?", and on a no the caller destroyed the menu and made a new one for GetForegroundWindow().
// With the host in the background that is somebody else's window, so the next frame said no
// again: the whole menu -- ImGui context, platform backend, two descriptor heaps -- was built
// and torn down every other frame for as long as another application had the focus (seen in
// Elgato Studio: "Initialize requested target:<foreign hwnd>" in the log every 33 ms). A menu
// that never rendered did not release its heaps (see ~Menu_Dx12), so a shader-visible heap
// leaked 30 times a second until the device refused new ones ("Failed to init heap"), and the
// first Render() of the instance that happened to be made when the host got the focus back is
// where an access violation in the driver was caught.
// Now the handle is kept for as long as it is a live window of this process -- which window
// EnumWindows names first changes with the z-order, and that is no reason to move -- and when it
// does have to move, only the platform backend and the input layer are re-targeted.
void MenuDxBase::SyncHandle()
{
    if (Config::Instance()->OverlayMenu.value_or_default())
        return;

    if (IsOwnLiveWindow(_handle))
        return;

    HWND host = ResolveHostWindow();

    if (host == nullptr || host == _handle)
        return;

    LOG_INFO("Menu host window {:X} is gone or not ours, re-targeting to {:X}", (size_t) _handle, (size_t) host);
    MenuCommon::Retarget(host);
    _handle = host;
}

void MenuDxBase::Dx11Ready() { MenuCommon::Dx11Inited(); }

void MenuDxBase::Dx12Ready() { MenuCommon::Dx12Inited(); }

MenuDxBase::MenuDxBase(HWND handle) : _handle(handle)
{
    if (Config::Instance()->OverlayMenu.value_or_default())
        return;

    MenuCommon::Init(handle, false);

    _baseInit = MenuCommon::IsInited();
}

MenuDxBase::~MenuDxBase()
{
    if (!_baseInit)
        return;

    MenuCommon::Shutdown();
}

bool MenuDxBase::IsVisible() { return MenuCommon::IsVisible(); }
