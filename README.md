# Laser Shooter

## 1. Print the target

Print `target.png` on A4 paper. The camera must be able to see all four Aruco
markers around the target.

## 2. Create a virtual environment

Python 3.10 or newer is required.

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
