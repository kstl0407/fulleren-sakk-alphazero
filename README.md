# AlphaZero-ágens egy gráfalapú háromdimenziós sakkvariánshoz

Szakdolgozati projekt (PPKE ITK, Mérnökinformatikus BSc, 2026): *Mesterséges intelligencia fejlesztése
egy gráfalapú háromdimenziós sakkvariánshoz.*

A játékot két fél játssza egy fullerénszerű test 72 lapján (12 ötszög, 60 hatszög). Mindkét félnek egy
zászlója, öt bástyája és öt futója van, a parti a zászló leütésével ér véget. Az ágens AlphaZero-elvű:
önjátékból tanul, a lépést egy policy-value neurális hálózattal irányított Monte-Carlo fakeresés (MCTS)
választja. A tároló a tanítókódot (a dolgozat utolsó, v6-os változatát), a betanított modellt, a kari
játékfelülethez készült gépi játékost, valamint a mérő- és elemzőprogramokat tartalmazza.

## Mappák

| Mappa | Tartalom |
|---|---|
| `tanitas/` | szabálymotor, hálózat, tanítási ciklus, körmérkőzés és kezdőelőny-mérés |
| `tanitas/alpha-zero-general-master/` | az [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) keretrendszer módosított része (MIT-licenc) |
| `modellek/` | a v6 futás legjobb modellje (`v6_best.pth.tar`, 6,6 millió paraméter) |
| `gepi_jatekos/` | gépi játékos a kari webes felülethez, a felületet utánzó tesztkörnyezet |
| `elemzes/` | önjáték-naplók elemzése, vakfolt- és fenyegetés-teszt, futások összesítése |

## Telepítés

Python 3.10 vagy újabb szükséges (a tesztek Python 3.14-gyel és PyTorch 2.14-gyel futottak).

```
python -m venv .venv
.venv\Scripts\activate          # Linuxon: source .venv/bin/activate
pip install -r tanitas/requirements.txt
```

A gépi játékoshoz és a mérésekhez a CPU-s PyTorch is elég, a tanításhoz GPU ajánlott.

## Tanítás (`tanitas/`)

```
cd tanitas
python main.py --profil v6 --gyors-teszt --checkpoint-dir proba     # pár perces próbafutás
python main.py --profil v6 --iters 100 --load-model ../modellek/v6_best.pth.tar --ref-model ../modellek/v6_best.pth.tar --log-every 10
python main.py --profil v6h --iters 500 --ref-model ../modellek/v6_best.pth.tar --log-every 10    # nulláról
```

A `--profil` kapcsoló választja ki a beállításcsomagot, így a dolgozat összes változata ugyanabból a
kódból futtatható:

| Profil | Lényeg |
|---|---|
| `regi` | az eredeti beállítás (one-hot célok, nyerés-ellenőrzés nélkül) |
| `v2` | nyerés-ellenőrzés a csúcs kibontásakor, diszkontálás, látogatási eloszlás mint cél, ismétlés = döntetlen |
| `v3` | + két véletlen nyitó féllépés az önjátékban |
| `v4` | + fenyegetés-ellenőrzés a gyökérben, párosított aréna, referencia-kapu |
| `v5` | + előjelpróbás aréna, újramérő referencia-kapu, rövidebb sorsolt szakasz |
| `v6` | + 150 önjáték-parti iterációnként, szigorúbb referencia-kapu (40 pár, tűrés nélkül) |
| `v6h` | v6 takarékos mentéssel, a nulláról indított 500 iterációs futáshoz |

A futás kimenete a checkpoint-mappába kerül: `best.pth.tar`, `iteraciok.csv` (iterációnkénti
összesítő), a `--log-every N` kapcsolóval `jatszma_naplo.txt` (minden N-edik önjáték-parti lépésenként).

További programok:

```
python korverseny.py ../modellek/v6_best.pth.tar@50 moho veletlen --parok 20    # párosított körmérkőzés
python kezdoelony.py ../modellek/v6_best.pth.tar --partik 300 --nyito 0 2 4      # a világos győzelmi aránya
python play.py --ai-color black --checkpoint ../modellek/v6_best.pth.tar          # ember a gép ellen, konzolon
```

A `korverseny.py` modelljei mellett a `moho` (ha tud, üt, és nem hagy egylépéses zászlóütést) és a
`veletlen` játékos is szerepelhet, a `fajl.pth.tar@N` alak pedig modellenként megadja a szimulációszámot.

## Gépi játékos (`gepi_jatekos/`)

A program soronként egy JSON-üzenetet olvas a szabványos bemenetről (az állás FEN72-leírása, a szabályos
lépések listája és az időkorlát), és egyetlen sorban válaszol a választott lépéssel. A döntés sorrendje:
zászlóütés, ha van; különben időkorlátos MCTS nyerés-ellenőrzéssel; végül biztonsági szűrő, amely nem
választ olyan lépést, amely után az ellenfél azonnal leütheti a zászlót.

```
cd gepi_jatekos
python onteszt.py                                                   # kezdőállás, "ÖNTESZT OK"
python teszt_adapter.py --jatszmak 20 --ellenfel moho --idokorlat 1500
python csomagol.py                                                  # önálló csomag: ../kimenet/gepi_jatekos/
```

## Elemzés és mérés (`elemzes/`)

| Program | Mit mér |
|---|---|
| `naplo_elemzes.py <jatszma_naplo.txt>` | a napló visszajátszása: egylépéses nyerés kihasználása, fenyegetés kivédése, elkerülhető baklövések, színek aránya |
| `futas_elemzes.py <futas_mappa> --nev v6 [--osszevet <mappa>]` | egy futás összesítése: elfogadások, aréna, referencia-kapu, önjáték, napló |
| `vakfolt_teszt.py --log <napló> --n 60` | nyerő állásokban a hálózat priorja és a keresés döntése különböző szimulációszámokkal |
| `fenyegetes_teszt.py --log <napló> --modellek ...` | fenyegetett állásokban a kivédő lépésekre jutó prior és a keresés döntése |
| `kivedes_naplo.py <napló> ...` | a naplóbeli kivédési arány kivédhető és kivédhetetlen fenyegetésekre bontva |
| `jatszma_export.py <napló> --iter 50 --ep 100` | egy naplózott parti kiírása FEN72-állássorozatként |

## Megjegyzések

- A játék szabályainak Python-megvalósítása a projektcsapat C++ játékmotorján alapul.
- A tanítási ciklus az alpha-zero-general keretrendszerre épül. A fakeresés, az önjáték, az aréna
  és a tanítóminta-kezelés jelentősen módosított, a változásokat a `tanitas/main.py` profiljai és a kód kommentjei írják le.
- A mérési adatok (tanítási naplók, játszmanaplók) mérete miatt nem részei a tárolónak.
