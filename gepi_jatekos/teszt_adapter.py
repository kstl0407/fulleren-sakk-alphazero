"""
teszt_adapter.py — a szerveroldali adapter helyi utánzata az ai_jatekos.py teszteléséhez.

egy játszma = egy folyamat; `lepj` üzenet FEN72-vel és a GUI-formátumú lépéslistával
(a sánc mindkét irányával), a válasz egy JSON-sor; a végén `vege`.
Ellenőrzi: a válasz egy sor, érvényes JSON, a listából való, határidőn belül érkezett.

Ellenfelek: `veletlen` (véletlen szabályos lépés), `mohó` (zászlót üt, ha tud, különben
véletlen, de nem hagy egylépéses zászlóütést), vagy `ai` (még egy ai_jatekos.py folyamat).

Futtatás (ebből a mappából):
    python teszt_adapter.py --jatszmak 4 --ellenfel veletlen
    python teszt_adapter.py --jatszmak 2 --ellenfel moho --idokorlat 2000
    python teszt_adapter.py --jatszmak 2 --ellenfel ai --ai-args="--halo-nelkul"
"""
import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
import queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import fen72  # noqa: E402
from game import NUM_TILES  # noqa: E402

G = fen72._game()


class Folyamat:
    """Egy ai_jatekos.py példány stdin/stdout-on keresztül."""

    def __init__(self, extra_args, nev, stderr_path, program=None):
        prog = os.path.abspath(program) if program else os.path.join(HERE, "ai_jatekos.py")
        # a program a saját mappájából fut (mint a valódi adapternél), így egy átadott csomag önmagában tesztelhető
        cmd = [sys.executable, "-u", prog] + shlex.split(extra_args)
        self.cwd = os.path.dirname(prog)
        self.nev = nev
        self.err = open(stderr_path, "a", encoding="utf-8")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  text=True, encoding="utf-8", bufsize=1, cwd=self.cwd)
        self.q = queue.Queue()
        threading.Thread(target=self._olvas, daemon=True).start()

    def _olvas(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)

    def kerdez(self, msg, timeout):
        self.p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return "__TIMEOUT__"

    def vege(self, msg):
        try:
            self.p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
        self.err.close()
        # a válaszon kívül nem szabad semmit a stdout-ra írnia
        maradek = []
        while not self.q.empty():
            x = self.q.get()
            if x:
                maradek.append(x)
        return maradek


def flag_win_moves(gui_moves, board, player):
    f = fen72._flag_tile(board, -player)
    return [m for m in gui_moves if m[1] - 1 == f]


def safe_moves(gui_moves, board, player):
    out = []
    for m in gui_moves:
        a = fen72.gui_to_action(board, player, m)
        nb, npl = G.getNextState(board, player, a)
        mf = fen72._flag_tile(nb, player)
        v = G.getValidMoves(nb, npl)
        if not v[np.arange(NUM_TILES) * NUM_TILES + mf].any():
            out.append(m)
    return out


def jatszma(ai_szin, ellenfel, args, idx, rng):
    """Egy játszma. ai_szin: +1 (világos) vagy -1. Visszaad: (győztes, lépésszám, hibák, statok)."""
    errlog = os.path.join(HERE, "teszt_adapter_stderr.log")
    procs = {ai_szin: Folyamat(args.ai_args, "AI", errlog, args.program)}
    if ellenfel == "ai":
        procs[-ai_szin] = Folyamat(args.ellenfel_args, "AI2", errlog)

    board, player = G.getInitBoard(), 1
    clocks = {1: 600.0, -1: 600.0}
    hibak, idok = [], []
    winner, ply, mod = 0, 0, "lépéslimit"
    while ply < args.max_lepes:
        ply += 1
        gui = [list(m) for m in fen72.engine_gui_moves(board, player)]
        if not gui:
            winner, mod = -player, "nincs lépés"
            break
        if player in procs:
            msg = {"tipus": "lepj", "fen": fen72.to_fen(board, player, ply, (clocks[1], clocks[-1])),
                   "szin": "vilagos" if player == 1 else "sotet", "lepesek": gui,
                   "idokorlat_ms": args.idokorlat}
            t0 = time.time()
            # az első lépésnél a modell betöltése is belefér (a valódi adapter a parti elején indít)
            resp = procs[player].kerdez(msg, timeout=args.idokorlat / 1000 + (30 if ply <= 2 else 1))
            dt = time.time() - t0
            idok.append(dt)
            clocks[player] -= dt
            try:
                m = json.loads(resp)
                assert isinstance(m, list) and m in gui, f"nem a listából: {m}"
                if dt > args.idokorlat / 1000 and ply > 2:
                    hibak.append(f"lépés {ply}: időtúllépés {dt:.2f} s")
            except Exception as e:
                hibak.append(f"lépés {ply}: rossz válasz {resp!r} ({e}) → véletlen lépés")
                m = rng.choice(gui)
        else:
            if ellenfel == "moho":
                m = (flag_win_moves(gui, board, player) or safe_moves(gui, board, player) or gui)
                m = rng.choice(m)
            else:
                m = rng.choice(gui)
        a = fen72.gui_to_action(board, player, m)
        board, player = G.getNextState(board, player, a)
        if fen72._flag_tile(board, player) < 0:
            winner, mod = -player, "Zászló leütve"
            break

    for szin, p in procs.items():
        maradek = p.vege({"tipus": "vege", "gyoztes": {1: "vilagos", -1: "sotet", 0: None}[winner],
                          "mod": mod, "sajat_szin": "vilagos" if szin == 1 else "sotet"})
        if maradek:
            hibak.append(f"{p.nev}: felesleges stdout-sorok: {maradek[:3]}")
    return winner, ply, mod, hibak, idok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jatszmak", type=int, default=2)
    ap.add_argument("--ellenfel", choices=["veletlen", "moho", "ai"], default="veletlen")
    ap.add_argument("--idokorlat", type=int, default=5000, help="idokorlat_ms az AI-nak")
    ap.add_argument("--max-lepes", type=int, default=300)
    ap.add_argument("--ai-args", default="", help="extra argumentumok az ai_jatekos.py-nak")
    ap.add_argument("--ellenfel-args", default="--halo-nelkul", help="a második AI argumentumai (--ellenfel ai)")
    ap.add_argument("--program", default=None,
                    help="a tesztelt játékosprogram (alap: ai_jatekos.py ebből a mappából), pl. egy csomagolt példány ai_jatekos.py-ja")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"# Adapter-teszt: AI vs {args.ellenfel}, {args.jatszmak} játszma, idokorlat {args.idokorlat} ms\n")
    print("| # | AI színe | Győztes | Mód | Féllépések | Átl. válaszidő [s] | Max [s] | Hibák |")
    print("|---|---|---|---|---|---|---|---|")
    ai_nyert = 0
    osszes_hiba = 0
    for i in range(args.jatszmak):
        szin = 1 if i % 2 == 0 else -1
        w, ply, mod, hibak, idok = jatszma(szin, args.ellenfel, args, i, rng)
        ai_nyert += w == szin
        osszes_hiba += len(hibak)
        later = idok[1:] or idok
        print(f"| {i + 1} | {'világos' if szin == 1 else 'sötét'} | "
              f"{'AI' if w == szin else ('ellenfél' if w else 'nincs')} | {mod} | {ply} | "
              f"{np.mean(later):.2f} | {max(later):.2f} | {'; '.join(hibak) or '–'} |", flush=True)
    print(f"\n**AI győzelem: {ai_nyert}/{args.jatszmak}**, protokollhibák: {osszes_hiba}")


if __name__ == "__main__":
    main()
