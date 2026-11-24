"""
onteszt.py — a gépi játékos gyors önellenőrzése.

Elindítja az ai_jatekos.py-t (ugyanabból a mappából), elküldi a kezdőállást a teljes, 115 elemű
lépéslistával (világos), majd egy sötét állást, végül a `vege` üzenetet, és ellenőrzi:
  * a válasz egyetlen sor, érvényes JSON, a `lepesek` listából való;
  * időben érkezik (az első lépésnél a modell betöltésére is jut idő);
  * a stdout-ra a válaszokon kívül semmi nem kerül; a program a `vege` után kilép.

    python3 onteszt.py            # kilépési kód 0 = minden rendben
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fen72  # noqa: E402

G = fen72._game()


def main():
    prog = os.path.join(HERE, "ai_jatekos.py")
    p = subprocess.Popen([sys.executable, "-u", prog], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8", cwd=HERE)
    hibak = []
    board, player = G.getInitBoard(), 1
    for lepes_szam, idokorlat in ((1, 5000), (2, 2000)):
        gui = [list(m) for m in fen72.engine_gui_moves(board, player)]
        msg = {"tipus": "lepj", "fen": fen72.to_fen(board, player, lepes_szam), "szin": "vilagos" if player == 1 else "sotet",
               "lepesek": gui, "idokorlat_ms": idokorlat}
        t0 = time.time()
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        sor = p.stdout.readline()
        dt = time.time() - t0
        try:
            m = json.loads(sor)
            assert m in gui, f"nem a listából: {m}"
        except Exception as e:  # noqa: BLE001
            hibak.append(f"{lepes_szam}. lépés: rossz válasz {sor!r} ({e})")
            break
        limit = idokorlat / 1000 + (30 if lepes_szam == 1 else 0)  # az első lépésnél modellbetöltés
        if dt > limit:
            hibak.append(f"{lepes_szam}. lépés: időtúllépés ({dt:.2f} s)")
        print(f"{lepes_szam}. lépés ({'világos' if player == 1 else 'sötét'}, {len(gui)} szabályos lépés): "
              f"{m}  {dt:.2f} s")
        board, player = G.getNextState(board, player, fen72.gui_to_action(board, player, m))
    p.stdin.write(json.dumps({"tipus": "vege", "gyoztes": "vilagos", "mod": "teszt", "sajat_szin": "vilagos"}) + "\n")
    p.stdin.close()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        hibak.append("a program a vege üzenet után nem lépett ki")
        p.kill()
    maradek = p.stdout.read().strip()
    if maradek:
        hibak.append(f"felesleges stdout: {maradek[:200]!r}")
    if hibak:
        print("HIBA:\n  " + "\n  ".join(hibak))
        print("stderr vége:\n" + p.stderr.read()[-2000:])
        sys.exit(1)
    print("ÖNTESZT OK")


if __name__ == "__main__":
    main()
