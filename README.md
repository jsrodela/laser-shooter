# Laser Shooter

## 1. Print the target

Print `target.png` on A4 paper. The camera must be able to see all four Aruco
markers around the target.

## 2. Create a virtual environment

Python 3.10 or newer is required.

### Windows one-click setup

On a clean Windows 10 or 11 installation, double-click `run.bat`. It will:

1. Find a compatible Python installation.
2. Install Python 3.13 with `winget` when Python is missing.
3. Create `.venv` without requiring PowerShell activation.
4. Install the required packages.
5. Open the graphical camera, calibration, and scoring interface.

An internet connection is required on the first run. Command-line options are
also available:

```bat
run.bat setup
run.bat gui
run.bat calibrate
run.bat main
```

Double-clicking `run.bat` opens the GUI by default. The `calibrate` and `main`
options run the legacy terminal interfaces.

## Graphical interface

The GUI combines the full workflow in one window:

- Live camera preview with Aruco marker count
- Rectified target preview
- Live red-laser threshold calibration and slider
- Webcam and video-file selection
- Player name and target settings
- Shot history, shot count, and total score
- Start, stop, and round reset controls

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow it for the current terminal
session only and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Activation is optional. You can explicitly use the virtual environment's
Python instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Linux and macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Adjust the laser threshold

```sh
python red_difference.py
```

Select `0` for a webcam and enter its camera id. The default camera id is `0`.
Adjust the white threshold until only the red laser point is detected.

- `Q`: Quit
- `C`: Toggle the preview mode

## 4. Run the shooter

```sh
python main.py
```

- Enter `settings` instead of a player name to change the target dimensions,
  score ring dimensions, and white threshold.
- Select `0` for a webcam.
- Select `1` to use a video placed in the optional `videos` directory.
- `R`: Reset the current shots
- `H`: Print hit and shot data
- `Esc`: Quit

On Windows, if a webcam cannot be opened, check **Settings > Privacy &
security > Camera**, close other apps using the camera, and try another camera
id.
