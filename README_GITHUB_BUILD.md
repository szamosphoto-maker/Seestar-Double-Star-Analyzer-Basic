# Seestar Double Star Analyzer Basic — Windows EXE build

Ez a csomag GitHub Actions segítségével készíti el a Windows EXE-t.
A saját gépeden nem kell PyInstallert telepíteni és nem kell parancssorban fordítani.

## Feltöltendő fájlok

A ZIP teljes tartalmát töltsd fel egy GitHub repository-ba, beleértve:

- `Seestar_Double_Star_Analyzer_Basic_GUI.py`
- `photoanalyzer_basic.py`
- `Seestar_wds.csv`
- `.github/workflows/build-windows.yml`

A `.github` mappa fontos.

## Build indítása

1. Nyisd meg a repository-t GitHubon.
2. Kattints felül az **Actions** fülre.
3. Bal oldalon válaszd: **Build Windows EXE**.
4. Kattints: **Run workflow**.
5. Ismét: **Run workflow**.
6. Várd meg, amíg a build zöld pipát kap.
7. Nyisd meg az elkészült workflow futást.
8. Legalul az **Artifacts** résznél töltsd le:
   `Seestar-Double-Star-Analyzer-Basic-Windows`

A letöltött ZIP-ben ez a két fájl lesz:

- `Seestar Double Star Analyzer Basic.exe`
- `Seestar_wds.csv`

Ezt a két fájlt kell együtt tartani.

A végfelhasználónak:
- nem kell Python,
- nem kell PyInstaller,
- nem kell BAT fájl,
- nem kell parancssor,
- csak az EXE-re kell duplán kattintani.

## Megjegyzés

Az EXE digitálisan nincs aláírva, ezért a Windows SmartScreen első indításkor figyelmeztethet.
