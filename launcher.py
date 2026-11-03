"""One-click launcher: sets up Python, then runs the hand tracker and the Processing sketch.

What it does, in order:
1. Creates a virtual environment in .venv and installs requirements.txt
   (only on the first run, or when requirements.txt changes).
2. Finds Processing. It supports both the old ``processing-java`` command
   (Processing <= 4.3) and the newer ``Processing cli`` (Processing >= 4.4).
   If it can't find it, it asks you to pick the program once and remembers it.
3. Starts hand_detector.py, waits for it to listen, then starts the sketch.
4. When you close either one, it closes the other.

Usage:
    python launcher.py                 # run everything
    python launcher.py --check         # verify the setup and compile the sketch, then exit
    python launcher.py --pick-processing   # choose the Processing program again
    python launcher.py --camera 1      # any other options go to hand_detector.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
CONFIG = ROOT / "launcher_config.json"
BUILD = ROOT / ".build" / "robotic_arm"  # Processing wants the folder named after the main .pde
SKETCH_FILES = ["robotic_arm.pde", "*.obj", "*.mtl"]
HOST, PORT = "127.0.0.1", 65432
IS_WINDOWS = os.name == "nt"


def say(msg: str) -> None:
    print(f"[launcher] {msg}", flush=True)


def fail(msg: str) -> NoReturn:
    say(f"ERROR: {msg}")
    sys.exit(1)


# --- Python environment ----------------------------------------------------


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def ensure_venv() -> Path:
    """Create .venv and install requirements when they change."""
    if sys.version_info < (3, 9):
        fail(f"Python 3.9 or newer is needed (this is {sys.version.split()[0]}).")
    py = venv_python()
    if not py.exists():
        say("First run: creating a Python environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV)

    req = ROOT / "requirements.txt"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    stamp = VENV / ".requirements.sha256"
    if not stamp.exists() or stamp.read_text().strip() != digest:
        say("Installing Python packages (MediaPipe, OpenCV). This takes a minute the first time ...")
        cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)]
        if subprocess.call(cmd) != 0:
            fail("Package installation failed. Check your internet connection, and that your "
                 "Python version is supported by MediaPipe (3.9-3.13 at the time of writing).")
        stamp.write_text(digest)
    return py


# --- Processing ------------------------------------------------------------


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG.write_text(json.dumps(cfg, indent=2))


def processing_candidates() -> list[str]:
    """Places Processing is usually installed, newest layouts first."""
    home = Path.home()
    pats: list[str] = []
    if IS_WINDOWS:
        roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        roots = [r for r in roots if r]
        for r in roots:
            pats += [
                rf"{r}\Processing\Processing.exe",
                rf"{r}\Programs\Processing\Processing.exe",
                rf"{r}\processing-*\processing-java.exe",
                rf"{r}\processing-*\Processing.exe",
            ]
        for base in (home, home / "Downloads", home / "Documents", Path("C:/"), Path("D:/")):
            pats += [str(base / "processing-*" / "processing-java.exe"),
                     str(base / "processing-*" / "Processing.exe"),
                     str(base / "Processing" / "Processing.exe")]
    elif sys.platform == "darwin":
        pats += ["/Applications/Processing.app/Contents/MacOS/Processing",
                 str(home / "Applications/Processing.app/Contents/MacOS/Processing")]
    else:
        pats += [str(home / "processing-*" / "processing-java"),
                 str(home / "processing-*" / "bin" / "Processing"),
                 "/opt/processing*/processing-java", "/opt/processing*/bin/Processing",
                 "/usr/local/bin/processing-java", "/snap/bin/processing"]
    found: list[str] = []
    for p in pats:
        # Newest version first when several processing-* folders exist
        found += sorted(glob.glob(p), reverse=True)
    return found


def is_processing_java(exe: str) -> bool:
    name = exe.replace("\\", "/").rsplit("/", 1)[-1]  # accept Windows paths on any OS
    return name.lower().startswith("processing-java")


def find_processing(force_pick: bool = False) -> str:
    cfg = load_config()
    if not force_pick:
        candidates = [cfg.get("processing"), os.environ.get("PROCESSING")]
        candidates += [shutil.which("processing-java"), shutil.which("Processing"), shutil.which("processing")]
        candidates += processing_candidates()
        for c in candidates:
            if c and Path(c).is_file():
                if c != cfg.get("processing"):
                    cfg["processing"] = c
                    save_config(cfg)
                return c

    exe = pick_processing_dialog()
    if not exe:
        fail("Processing was not found. Install Processing 4 from https://processing.org/download, "
             "or set the PROCESSING environment variable to Processing.exe / processing-java.")
    cfg["processing"] = exe
    save_config(cfg)
    return exe


def pick_processing_dialog() -> str | None:
    """Ask the user to point at the Processing program, once."""
    say("Could not find Processing automatically. Pick the Processing program in the window that opens.")
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Visuocontrol launcher",
        "Processing was not found.\n\nSelect Processing.exe (Processing 4.4 or newer) or "
        "processing-java.exe (older versions). This is only asked once.",
    )
    types = [("Processing", "Processing.exe processing-java.exe"), ("All files", "*.*")] if IS_WINDOWS else []
    path = filedialog.askopenfilename(title="Select Processing", filetypes=types or [("All files", "*")])
    root.destroy()
    return path or None


def sketch_command(processing: str, action: str) -> list[str]:
    """``--run`` or ``--build`` must be the last argument for both CLIs."""
    base = [processing] if is_processing_java(processing) else [processing, "cli"]
    return base + [f"--sketch={BUILD}", f"--output={BUILD.parent / 'out'}", "--force", action]


def stage_sketch() -> None:
    """Copy the sketch into .build/robotic_arm.

    Processing requires the sketch folder to be named like its main .pde file,
    and this repository's folder is not. Files are only copied when they changed.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    for pattern in SKETCH_FILES:
        for src in ROOT.glob(pattern):
            dst = BUILD / src.name
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)


# --- processes -------------------------------------------------------------


def start(cmd: list[str], name: str) -> subprocess.Popen:
    say(f"Starting {name}")
    kwargs: dict = {"cwd": str(ROOT)}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def stop(proc: subprocess.Popen | None) -> None:
    """Stop a process and everything it started (Processing launches a Java child)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()


def wait_until_ready(proc: subprocess.Popen, ready_file: Path, timeout: float) -> bool:
    """Wait until the tracker reports it is listening. False if it exits or times out first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if ready_file.exists():
            return True
        time.sleep(0.3)
    return False


def port_in_use() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


# --- main ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hand tracker and the Processing sketch together.")
    parser.add_argument("--check", action="store_true", help="verify the setup and compile the sketch, then exit")
    parser.add_argument("--pick-processing", action="store_true", help="choose the Processing program again")
    args, tracker_args = parser.parse_known_args()

    os.chdir(ROOT)
    py = ensure_venv()
    processing = find_processing(force_pick=args.pick_processing)
    say(f"Processing: {processing}")
    stage_sketch()

    if args.check:
        say("Checking Python packages ...")
        probe = "import cv2, mediapipe; from mediapipe.tasks.python import vision; print('ok')"
        if subprocess.call([str(py), "-c", probe]) != 0:
            fail("The Python packages did not import correctly.")
        say("Compiling the sketch ...")
        if subprocess.call(sketch_command(processing, "--build")) != 0:
            fail("The sketch did not compile.")
        say("Everything is ready. Run the launcher without --check to start.")
        return

    if port_in_use():
        fail(f"Port {PORT} is already in use. Is the tracker already running in another window?")

    ready_file = BUILD.parent / "tracker.ready"
    ready_file.unlink(missing_ok=True)
    os.environ["VISUOCONTROL_READY_FILE"] = str(ready_file)

    tracker = sketch = None
    try:
        tracker = start([str(py), "hand_detector.py", *tracker_args], "hand tracker")
        # The first run downloads the hand model, so allow some time.
        if not wait_until_ready(tracker, ready_file, timeout=120):
            fail("The hand tracker did not start. See its messages above (camera in use or not found?).")

        sketch = start(sketch_command(processing, "--run"), "Processing sketch")
        say("Running. Close the 3D window, press q in the camera window, or Ctrl+C here to stop both.")

        sketch_started = time.monotonic()
        while tracker.poll() is None and sketch.poll() is None:
            time.sleep(0.5)

        if tracker.poll() is not None:
            say("Hand tracker closed; stopping the sketch.")
        elif sketch.returncode != 0:
            fail("The Processing sketch stopped with an error. See its messages above.")
        elif time.monotonic() - sketch_started < 5:
            # Some Processing versions hand the sketch to a separate Java
            # process and return at once. Don't take that as "closed".
            say("Processing returned right away; the sketch may still be open.")
            say("Leaving the tracker running: press q in the camera window to stop.")
            tracker.wait()
        else:
            say("Processing sketch closed; stopping the tracker.")
    except KeyboardInterrupt:
        say("Stopping ...")
    finally:
        stop(sketch)
        stop(tracker)


if __name__ == "__main__":
    main()
