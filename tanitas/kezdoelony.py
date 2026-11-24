"""
kezdoelony.py — a kezdőelőny (világos előnye) mérése egy modell önmaga elleni partijaival (v4, 2026-10-19).

Mindkét oldalon ugyanaz a modell és ugyanaz a keresés játszik, így a győzelmi arány eltérése
50%-tól a színnek (a kezdés jogának) tulajdonítható. A partik változatosságát `k` egyenletesen
véletlen nyitó féllépés adja. Több `k` értéket is mér: ha a világos előnye `k` növelésével (azaz
véletlenebb kiinduló állásokkal) is megmarad, az a játék sajátossága, nem egy nyitás terméke.

    python kezdoelony.py checkpoints_v4/best.pth.tar --partik 500 --nyito 0 2 4 --sims 100

Kimenet: Markdown-táblázat (stdout) 95%-os Wilson-intervallummal; folyamat a stderr-en.
"""
import argparse
import math
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "alpha-zero-general-master"))

from ertekeles import MCTSJatekos, jatek, sorsolt_nyitas  # noqa: E402
from game import FullerenSakkGame  # noqa: E402
from nnet import DEFAULT_ARGS, FullerenNNet  # noqa: E402
from utils import dotdict  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description="Kezdőelőny mérése önjátékkal")
    ap.add_argument("modell")
    ap.add_argument("--partik", type=int, default=300, help="partik száma nyitó-beállításonként")
    ap.add_argument("--nyito", type=int, nargs="+", default=[0, 2, 4], help="véletlen nyitó féllépések (több érték)")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--temp-lepes", type=int, default=2,
                    help="ennyi saját lépés a látogatási eloszlásból sorsolva (k=0-nál is legyen változatosság)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    game = FullerenSakkGame()
    p = os.path.abspath(a.modell)
    net = FullerenNNet(game, dict(DEFAULT_ARGS, checkpoint_dir=os.path.dirname(p)))
    _out, sys.stdout = sys.stdout, sys.stderr
    try:
        net.load_checkpoint(os.path.dirname(p), os.path.basename(p))
    finally:
        sys.stdout = _out
    margs = dotdict(dict(numMCTSSims=a.sims, cpuct=1.5, mcts_win_check=True, mcts_gamma=0.99, mcts_threat_check="root"))
    rng = np.random.RandomState(a.seed)
    np.random.seed(a.seed)

    print(f"# Kezdőelőny – `{os.path.basename(p)}` önmaga ellen, {a.sims} szimuláció, "
          f"az első {a.temp_lepes} saját lépés sorsolva a látogatási eloszlásból\n")
    print("| Véletlen nyitó féllépés | Partik | Világos nyer | Sötét nyer | Döntetlen | Világos aránya (95% CI) | Átl. hossz |")
    print("|---|---|---|---|---|---|---|")
    t0 = time.time()
    for k in a.nyito:
        c = Counter()
        hossz = []
        for n in range(a.partik):
            board, player = sorsolt_nyitas(game, k, rng)
            feher = MCTSJatekos(game, net, margs, temp_moves=a.temp_lepes)
            fekete = MCTSJatekos(game, net, margs, temp_moves=a.temp_lepes)
            lepes = [0]
            def szamlal(f):
                def g(x):
                    lepes[0] += 1
                    return f(x)
                g.startGame = f.startGame
                return g
            r = jatek(game, szamlal(feher), szamlal(fekete), board, player)
            c[r] += 1
            hossz.append(lepes[0] + k)
            if (n + 1) % 50 == 0:
                print(f"  k={k}: {n + 1}/{a.partik}  világos {c[1]}, sötét {c[-1]}  [{time.time() - t0:.0f} s]",
                      file=sys.stderr, flush=True)
        dontott = c[1] + c[-1]
        lo, hi = wilson(c[1], dontott) if dontott else (float("nan"), float("nan"))
        print(f"| {k} | {a.partik} | {c[1]} | {c[-1]} | {c[0]} | {100*c[1]/max(1, dontott):.1f}% "
              f"({100*lo:.1f}–{100*hi:.1f}%) | {np.mean(hossz):.1f} |", flush=True)
    print(f"\nFutásidő: {(time.time() - t0)/60:.1f} perc.")


if __name__ == "__main__":
    main()
