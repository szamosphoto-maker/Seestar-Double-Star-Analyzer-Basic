# DuoStar Basic — Windows EXE build with GitHub Actions

Ez a workflow a GitHub Actions segítségével készíti el a Windowsos
**DuoStar Basic.exe** fájlt. A saját gépen nem kell PyInstallert telepíteni.

## A repository-ba szükséges fájlok

- `DuoStar_Basic_GUI_EN.py`
- `duostar_basic_engine.py`
- `Seestar_wds.csv`
- `.github/workflows/build-windows.yml`

A `.github/workflows` mappaszerkezet fontos.

## Build indítása

1. Nyisd meg a DuoStar Basic GitHub repository-t.
2. Kattints az **Actions** fülre.
3. Bal oldalon válaszd a **Build DuoStar Basic Windows EXE** workflow-t.
4. Kattints a **Run workflow** gombra.
5. A megjelenő panelen ismét kattints a **Run workflow** gombra.
6. Ha a folyamat elkészült, zöld pipa jelenik meg.
7. Nyisd meg az elkészült workflow futást.
8. Az oldal alján, az **Artifacts** résznél töltsd le:
   `DuoStar-Basic-Windows`

## A letöltött csomag tartalma

- `DuoStar Basic.exe`
- `Seestar_wds.csv`

A két fájlt ugyanabban a mappában kell tartani.

A végfelhasználónak nem kell:
- Python
- PyInstaller
- BAT fájl
- parancssor

Csak a `DuoStar Basic.exe` fájlt kell elindítani.

## Megjegyzés

Az EXE digitálisan nincs aláírva, ezért a Windows SmartScreen az első
indításkor figyelmeztetést jeleníthet meg.
