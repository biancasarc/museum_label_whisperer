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
            command, capture_output=True, text=True, timeout=DIALOG_TIMEOUT_SECONDS
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
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog;"
        f'$dialog.Description = "{_clean(prompt)}";'
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
        " { Write-Output $dialog.SelectedPath }"
    )
    # -STA is required: FolderBrowserDialog will not open on an MTA thread.
    return _run(["powershell", "-NoProfile", "-STA", "-Command", script])


def _linux(prompt: str) -> str | None:
    for chooser, args in (
        ("zenity", ["--file-selection", "--directory", f"--title={_clean(prompt)}"]),
        ("kdialog", ["--getexistingdirectory", str(Path.home())]),
    ):
        if shutil.which(chooser):
            return _run([chooser, *args])
    raise PickerUnavailable("install zenity or kdialog to browse for a folder")


def pick_folder(prompt: str = "Choose a folder") -> str | None:
    """Show a folder chooser and return the selected path, or None if cancelled."""
    system = platform.system()
    chooser = {"Darwin": _macos, "Windows": _windows, "Linux": _linux}.get(system)
    if chooser is None:
        raise PickerUnavailable(f"no folder window is available on {system}")

    chosen = chooser(prompt)
    if chosen is None:
        return None
    # macOS returns a trailing slash; everything else does not.
    return str(Path(chosen.rstrip("/\\")).expanduser().resolve())
