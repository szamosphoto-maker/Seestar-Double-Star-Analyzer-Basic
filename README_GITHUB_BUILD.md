# DuoStar Basic — GitHub Actions Windows EXE build

Ez a csomag a DuoStar Basic adaptív PSF-illesztéssel és az automatikus **5′ × 5′ annotált PNG-mentéssel** frissített forrásait tartalmazza.

## Feltöltendő fájlok (azonos elérési útvonalon)

- `Seestar_Double_Star_Analyzer_Basic_GUI.py` (a meglévő GUI)
- `photoanalyzer_basic.py` (az STF 2528 AC-n kipróbált adaptív mérőmotor)
- `duostar_annotation.py` (automatikus, 5′-es PNG-készítés)
- `Seestar_wds.csv` (külső WDS-katalógus)
- `.github/workflows/build-windows.yml` (frissített GitHub Actions workflow)

**Fontos:** a `.github/workflows/` könyvtárstruktúrát pontosan meg kell tartani. A GitHub repository gyökerébe töltendők fel a három `.py` fájl és a CSV.

## Build a GitHubon

1. A repository-ban töltsd fel / cseréld le az előző fájlokat a fenti elérési utakon (commit).
2. **Actions** → **Build Windows EXE** → **Run workflow** → **Run workflow**.
3. Várd meg a zöld pipát. Hiba esetén a sikertelen lépés naplója megmutatja az okot.
4. A futás **Artifacts** részében töltsd le a `Seestar-Double-Star-Analyzer-Basic-Windows` ZIP-et.
5. A ZIP-ből az EXE-t és a `Seestar_wds.csv`-t tartsd együtt egy mappában.
6. Kipróbálás: ugyanazon a Seestar FITS-képen mérd meg az **STF 2528 AB** és **STF 2528 AC** párt. Ellenőrizd, hogy a TXT és az 5′-es annotált PNG is elkészül-e.

## Mi változott a buildben?

- `matplotlib==3.9.2` települ és bekerül az EXE-be.
- A dinamikusan importált `duostar_annotation` modult explicit hozzáadjuk (`--hidden-import`).
- A PyInstaller parancs PowerShell-kompatibilis (egyetlen parancssor).
- A Windowsos build automatikusan ellenőrzi a Python-források szintaxisát és az importokat.

**Az EXE itt nem készült el.** Ezt a GitHub Actions Windows runner fogja előállítani. A végfelhasználónak továbbra sem kell Python vagy PyInstaller; a Seestar WDS CSV továbbra is az EXE mellett marad. Az EXE nincs digitálisan aláírva, ezért a SmartScreen figyelmeztethet.
