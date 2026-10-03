# SeSuCo – Solar Observation Tools 🌞

SeSuCo is a small Windows GUI for three solar-observation tools:

- **Sunspot Counter** – white-light Seestar sunspot and facular-field analysis
- **H-alpha Analyzer** – filament and active-region analysis
- **Monthly Sunspot Summary** – monthly VOTable and relative-sunspot-number plot

© Zsolt Szamosvari — szamos.photo@gmail.com

## Repository files

Keep these files in the repository root:

- `sesuco_gui.py` – English graphical user interface
- `sunspot.py` – white-light analysis
- `halpha.py` – H-alpha analysis
- `combine_sunspot.py` – monthly summary
- `sesuco.ico` – application icon
- `requirements.txt` – Python dependencies
- `sesuco.spec` – PyInstaller build definition
- `.github/workflows/build-windows.yml` – automatic Windows build

## Build with GitHub Actions

1. Create a GitHub repository and upload the complete contents of this folder, including the hidden `.github` folder.
2. Commit/push the files to the `main` branch.
3. Open the repository's **Actions** tab.
4. Select **Build SeSuCo for Windows**.
5. Choose **Run workflow**.
6. When the workflow finishes, download the **SeSuCo-Windows** artifact. It contains `SeSuCo.exe`.

The build uses **Python 3.11** and creates a single Windows executable with no console window. The three processing scripts and the application icon are embedded in the executable.

## Releases

If a Git tag beginning with `v` is pushed, for example `v1.0.0`, the same workflow also creates a GitHub Release and attaches `SeSuCo.exe` to it.

## Local Python use

The program can still be run without building an EXE:

```text
python sesuco_gui.py
```

Install the required packages first:

```text
pip install -r requirements.txt
```

## Notes

The scientific processing remains in the three separate Python modules. The GUI only provides the Windows-style interface and invokes those modules. This keeps future algorithm updates simple: update the relevant Python module and rebuild the executable.
