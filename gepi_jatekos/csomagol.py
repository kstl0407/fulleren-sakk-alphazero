"""
csomagol.py — önálló, a GUI-adapter gépére másolható mappa készítése a gépi játékoshoz.

    python csomagol.py                    # → gepi_jatekos/ (best.pth.tar-ral)
    python csomagol.py --model temp.pth.tar --ki gepi_jatekos_temp

A mappa tartalma: ai_jatekos.py, fen72.py, a szabálymotor (game.py, grid.py, grid_cache.json),
a háló (nnet.py), az alpha-zero-general szükséges részei (alpha-zero-general-master/: Game.py,
NeuralNet.py, MCTS.py, utils.py),
a modell (best.pth.tar), requirements.txt és INDITAS.md.
A célgépen:  pip install -r requirements.txt  →  python3 ai_jatekos.py --log ai.log
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "tanitas")
AZG = os.path.join(CODE, "alpha-zero-general-master")

FILES = [
    (HERE, "ai_jatekos.py"), (HERE, "fen72.py"), (HERE, "mcts_javitott.py"), (HERE, "onteszt.py"),
    (CODE, "game.py"), (CODE, "grid.py"), (CODE, "grid_cache.json"), (CODE, "nnet.py"),
]
# külön almappában, mert a Game.py és a game.py kis-/nagybetűre érzéketlen fájlrendszeren ütközne
AZG_FILES = ["Game.py", "NeuralNet.py", "MCTS.py", "utils.py"]

INDITAS = """# Fullerén-sakk gépi játékos – átadási leírás

AlphaZero-típusú játékos (önjátékkal tanított policy-value háló + Monte-Carlo fakeresés) a
kari játékfelület protokolljához: soronként egy JSON a stdin-en, egy válaszsor a stdout-on.

## Telepítés és indítás

    pip install -r requirements.txt          # numpy + torch (CPU-s build elég)
    python3 ai_jatekos.py --log ai.log       # ez az adapter által indítandó parancs

- **Nyelv:** Python 3.10+. GPU nem kell.
- **Egy folyamat = egy parti.** A modellt induláskor tölti be (~1–3 s a torch importjával együtt). Ha az
  első `lepj` már a betöltés alatt megérkezik, ezt az időt levonja a gondolkodási keretéből.
- **A stdout-ra csak a válasz megy,** minden napló a stderr-re, illetve a `--log` fájlba.
- A `vege` üzenet után kilép.

## Gyors ellenőrzés a célgépen

    python3 onteszt.py        # elküldi a kezdőállást a teljes 115 lépéses listával; „ÖNTESZT OK” = rendben

Kézi próba:

    echo '{"tipus":"lepj","fen":"1Z0/2F/3F/4F/5F/6F/8B/14B/20B/26B/32B/38b/44b/50b/56b/62b/67z0/68f/69f/70f/71f/72f v 600.0/600.0 1","szin":"vilagos","lepesek":[[1,2],[8,9]],"idokorlat_ms":5000}' | python3 ai_jatekos.py

## A döntés menete

1. Ha a listában van zászlóütés, azt lépi.
2. Különben időkorlátos MCTS a hálóval (az `idokorlat_ms` 60%-a, kemény határ 80%).
3. Biztonsági szűrő: nem választ olyan lépést, amely után az ellenfél azonnal leütheti a zászlót (ha van más).
4. Bármilyen hiba esetén szabályos lépést küld (nem akad el), a hibát a naplóba írja.

**Mindig a kapott `lepesek` listából választ.** A saját szabálymotorja (game.py) minden lépésnél
összeveti a listát a saját lépésgenerálásával; eltérés esetén `SZABÁLYELTÉRÉS` sor kerül a naplóba (kérjük,
jelezzék, ha ilyet látnak). A sáncot mindkét irányban (`[1,2]` és `[2,1]`) kezeli.

## Kapcsolók

`--ido-arany 0.6` (az időkorlát ekkora része gondolkodásra), `--max-sims 1000`, `--min-sims 32`,
`--szalak 2` (CPU-szálak), `--log FÁJL`, `--halo-nelkul` (torch nélküli tartalék mód: azonnali
ütés + biztonsági szűrő), `--model FÁJL`.

## Válaszok a játékfelület fejlesztőinek kérdéseire

- **Nyelv, parancs:** Python 3, `python3 ai_jatekos.py --log ai.log`.
- **Előre számol?** Igen, saját szabálymotorral.
- **Idő lépésenként:** az alapértelmezett 5 s megfelel; rövidebb korláttal is működik (≥ 0,5 s).
- **A parti végén:** a `vege` üzenet elég; ha a teljes lépéssor is benne lenne, azt elemzésre használnánk.
"""

REQ = "numpy\ntorch  # CPU-s build elég: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "modellek", "v6_best.pth.tar"))
    ap.add_argument("--ki", default=os.path.join(os.path.dirname(HERE), "kimenet", "gepi_jatekos"))
    a = ap.parse_args()
    os.makedirs(a.ki, exist_ok=True)
    for d, f in FILES:
        shutil.copy2(os.path.join(d, f), os.path.join(a.ki, f))
    os.makedirs(os.path.join(a.ki, "alpha-zero-general-master"), exist_ok=True)
    for f in AZG_FILES:
        shutil.copy2(os.path.join(AZG, f), os.path.join(a.ki, "alpha-zero-general-master", f))
    shutil.copy2(a.model, os.path.join(a.ki, "best.pth.tar"))
    open(os.path.join(a.ki, "INDITAS.md"), "w", encoding="utf-8").write(INDITAS)
    open(os.path.join(a.ki, "requirements.txt"), "w", encoding="utf-8").write(REQ)
    print(f"Csomag kész: {a.ki} ({len(FILES) + len(AZG_FILES) + 3} fájl), modell: {a.model}")


if __name__ == "__main__":
    main()
