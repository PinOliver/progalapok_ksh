# Ingatlan Adatfeldolgozó és Vizualizációs Eszköz

Ez a Python projekt egy **ingatlanárakat elemző és vizualizáló eszköz**, amely támogatja az alapvető statisztikai elemzéseket, diagramkészítést és lineáris regresszió alkalmazását. Az adatokat egy CSV fájlból olvassa be és dolgozza fel.

---

## Fő Funkciók
1. **Adatbetöltés és tisztítás:**
   - A program betölti az `ingatlan.csv` fájlt.
   - Az "vármegye" szöveg eltávolításra kerül a helynevekből.
   - Az adatokat számokká alakítja a feldolgozáshoz.

2. **Statisztikai elemzés:**
   - Átlag, medián, minimum és maximum értékek számítása.
   - Az eredményeket konzolon jeleníti meg.

3. **Diagramok:**
   - **Vonaldiagramok**:
     - Családi házak átlagos m² ára megyék szerint.
     - Lakások átlagos m² ára megyék szerint.
   - **Csoportosított oszlopdiagram**:
     - Több ingatlan típus (családi ház, többlakásos, panel, összesített átlag) m² ára.

4. **Lineáris regresszió:**
   - A helyek sorszámai és az árak között regressziós egyenest illeszt.
   - R² és MSE metrikák kiszámítása.
   - Az eredményeket egy scatter plot ábrán jeleníti meg.

---

## Követelmények
A program futtatásához az alábbi Python könyvtárak szükségesek:
- `csv` (beépített)
- `tabulate`
- `numpy`
- `matplotlib`
- `scikit-learn`

Telepítés:
```bash
pip install tabulate numpy matplotlib scikit-learn
```

## Használat

A KSH oldaláról adatok .csv formátumban letöltése ingatlan.csv névre
- https://www.ksh.hu/s/ingatlanadattar/adattar?year=2023
- https://www.ksh.hu/

A futttás:
```bash
python3 ingatlan.py
```
