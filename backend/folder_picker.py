"""Open the operating system's own folder-chooser and return what was picked.

A browser cannot hand a folder *path* back to a web app — ``st.file_uploader``
gives you file contents, not a location.  This app runs locally though, so the
browser and this Python process are the same machine, and we can open the
platform's native dialog here and read back the chosen path.

Each platform is driven through a subprocess rather than an in-process GUI
toolkit.  ``tkinter`` would do the job on Windows and Linux, but on macOS a Tk
window has to be created on the main thread, and Streamlit runs page code on a
worker thread — doing it in-process there can hang or crash the whole app.  A
subprocess has no such constraint, so the same approach is used everywhere.

Returns the chosen path, or ``None`` if the person cancelled the dialog.
Raises :class:`PickerUnavailable` when no dialog can be shown at all (no
desktop session, or the platform's helper is missing), so callers can fall back
to the text box instead of pretending the feature worked.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

# Generous: this is however long someone takes to find their folder.
DIALOG_TIMEOUT_SECONDS = 600


class PickerUnavailable(RuntimeError):
    """No native folder dialog can be shown on this machine."""


def _clean(prompt: str) -> str:
    """Strip quotes and newlines so a prompt cannot break out of the script."""
    return "".join(c for c in prompt if c not in '"\\\n\r')[:120]


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            # Folder names routinely contain non-ASCII characters (å, ä, ö).
            # Windows consoles do not default to UTF-8, so say so explicitly
            # rather than letting the locale mangle the path.
            encoding="utf-8",
            errors="replace",
            timeout=DIALOG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise PickerUnavailable(f"{command[0]} is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise PickerUnavailable("the folder window was left open too long") from error

    # Every backend below exits non-zero (or prints nothing) when cancelled.
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def _macos(prompt: str) -> str | None:
    return _run(
        ["osascript", "-e", f'POSIX path of (choose folder with prompt "{_clean(prompt)}")']
    )


def _windows(prompt: str) -> str | None:
    script = (
        # Force UTF-8 out, so accented folder names survive the trip back.
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog;"
        f'$dialog.Description = "{_clean(prompt)}";'
        "$dialog.ShowNewFolderButton = $false;"
        # Without an owner window the dialog can open *behind* the browser and
        # look like the app has frozen. A throwaway top-most form fixes that.
        "$owner = New-Object System.Windows.Forms.Form;"
        "$owner.TopMost = $true;"
        "if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK)"
        " { Write-Output $dialog.SelectedPath };"
        "$owner.Dispose()"
    )
    # Windows PowerShell 5.1 ships on every Windows machine and is already STA;
    # PowerShell 7 (pwsh) dropped -STA, so it is tried without the flag.
    attempts = (
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        ["pwsh", "-NoProfile", "-Command", script],
    )
    last_error: PickerUnavailable | None = None
    for command in attempts:
        try:
            return _run(command)
        except PickerUnavailable as error:
            last_error = error
    raise last_error or PickerUnavailable("PowerShell is not available")


def _linux(prompt: str) -> str | None:
    for chooser, args in (
        ("zenity", ["--file-selection", "--directory", f"--title={_clean(prompt)}"]),
        ("kdialog", ["--getexistingdirectory", str(Path.home())]),
    ):
        if shutil.which(chooser):
            return _run([chooser, *args])
    raise PickerUnavailable("install zenity or kdialog to browse for a folder")


def _macos_file(prompt: str, extensions: tuple[str, ...]) -> str | None:
    of_type = ""
    if extensions:
        listed = ", ".join(f'"{e.lstrip(".")}"' for e in extensions)
        of_type = f" of type {{{listed}}}"
    return _run(
        ["osascript", "-e", f'POSIX path of (choose file with prompt "{_clean(prompt)}"{of_type})']
    )


def _windows_file(prompt: str, extensions: tuple[str, ...]) -> str | None:
    if extensions:
        patterns = ";".join(f"*{e if e.startswith('.') else '.' + e}" for e in extensions)
        file_filter = f"Supported files ({patterns})|{patterns}|All files (*.*)|*.*"
    else:
        file_filter = "All files (*.*)|*.*"
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$dialog = New-Object System.Windows.Forms.OpenFileDialog;"
        f'$dialog.Title = "{_clean(prompt)}";'
        f'$dialog.Filter = "{file_filter}";'
        "$owner = New-Object System.Windows.Forms.Form;"
        "$owner.TopMost = $true;"
        "if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK)"
        " { Write-Output $dialog.FileName };"
        "$owner.Dispose()"
    )
    for command in (
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        ["pwsh", "-NoProfile", "-Command", script],
    ):
        try:
            return _run(command)
        except PickerUnavailable:
            continue
    raise PickerUnavailable("PowerShell is not available")


def _linux_file(prompt: str, extensions: tuple[str, ...]) -> str | None:
    if shutil.which("zenity"):
        args = ["zenity", "--file-selection", f"--title={_clean(prompt)}"]
        if extensions:
            patterns = " ".join(f"*{e if e.startswith('.') else '.' + e}" for e in extensions)
            args.append(f"--file-filter={patterns}")
        return _run(args)
    if shutil.which("kdialog"):
        return _run(["kdialog", "--getopenfilename", str(Path.home())])
    raise PickerUnavailable("install zenity or kdialog to browse for a file")


def _choose(kind: str, prompt: str, extensions: tuple[str, ...]) -> str | None:
    folder_choosers = {"Darwin": _macos, "Windows": _windows, "Linux": _linux}
    file_choosers = {"Darwin": _macos_file, "Windows": _windows_file, "Linux": _linux_file}

    system = platform.system()
    chooser = (folder_choosers if kind == "folder" else file_choosers).get(system)
    if chooser is None:
        raise PickerUnavailable(f"no {kind} window is available on {system}")

    chosen = chooser(prompt) if kind == "folder" else chooser(prompt, extensions)
    if chosen is None:
        return None
    # macOS returns folders with a trailing slash; everything else does not.
    return str(Path(chosen.rstrip("/\\")).expanduser().resolve())


def pick_folder(prompt: str = "Choose a folder") -> str | None:
    """Show a folder chooser and return the selected path, or None if cancelled."""
    return _choose("folder", prompt, ())


def pick_file(prompt: str = "Choose a file", extensions: tuple[str, ...] = ()) -> str | None:
    """Show a file chooser and return the selected path, or None if cancelled."""
    return _choose("file", prompt, extensions)


# ---------------------------------------------------------------------------
# Streamlit widget
# ---------------------------------------------------------------------------
#
# Kept here rather than in its own module so every page gets the same layout
# and the same behaviour when no dialog can be shown.

def browse_input(
    label: str,
    *,
    state_key: str,
    default: str = "",
    prompt: str = "Choose a folder",
    help: str | None = None,
    extensions: tuple[str, ...] = (),
    is_file: bool = False,
) -> str:
    """A path box with a Browse button beside it. Returns the path now in the box.

    The button writes its result to ``state_key`` and reruns; the box reads that
    back, so typing a path by hand keeps working exactly as before.
    """
    import streamlit as st

    column_path, column_browse = st.columns([5, 1])

    with column_path:
        path = st.text_input(label, value=st.session_state.get(state_key) or default, help=help)

    with column_browse:
        st.markdown('<div style="height: 7mm;"></div>', unsafe_allow_html=True)
        if st.button("Browse…", key=f"browse_{state_key}", width="stretch"):
            try:
                chosen = pick_file(prompt, extensions) if is_file else pick_folder(prompt)
            except PickerUnavailable as error:
                st.warning(f"Could not open a window: {error}. Type the path instead.")
            else:
                if chosen:
                    st.session_state[state_key] = chosen
                    st.rerun()

    return path
