"""
fenyegetes_teszt.py — megtanulta-e a HÁLÓ a fenyegetés kivédését? (v4 elemzés, 2026-10-21)

A v4 a gyökérben a nem kivédő lépések priorját 0,05-szörösére csökkentette, hogy a tanítási cél
(látogatási eloszlás) a kivédést tanítsa. Itt a keresés segítsége NÉLKÜL mérjük a hálót:
olyan állásokat veszünk egy játszmanaplóból, ahol a soron lévő fél zászlaját az ellenfél a
következő lépésben leüthetné, nincs saját nyerő lépés, de van kivédés (game.getParryingMoves).

Modellenként:
  * a háló priorjának hány %-a esik a kivédő lépésekre (átlag, medián),
  * a prior top-1 lépése kivédés-e,
  * MCTS (fenyegetés-ellenőrzés KI, nyerés-ellenőrzés be, zaj nélkül) top-1 lépése kivédés-e.

Futtatás:
    python fenyegetes_teszt.py --log <jatszma_naplo.txt> --modellek ../modellek/v6_best.pth.tar --n 200 --sims 150
A dolgozat mérései a v3 futás naplójából vett 200 állást használták, minden modellnél ugyanazokat.
"""
import argparse
import os
import random
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = os.path.join(REPO, "tanitas")
for _p in (CODE, os.path.join(CODE, "alpha-zero-general-master"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game import NUM_TILES, FullerenSakkGame  # noqa: E402
from MCTS import MCTS  # noqa: E402
from nnet import DEFAULT_ARGS, FullerenNNet  # noqa: E402
from utils import dotdict  # noqa: E402
import naplo_elemzes as ne  # noqa: E402  


def allasok(game, log_path, n, seed=0):
    """Játszmánként az első fenyegetett, kivédhető, nyerés nélküli állás (kanonikus tábla)."""
    out = []
    for g in ne.parse(log_path):
        if "len" not in g:
            continue
        b, pl = game.getInitBoard(), 1
        for ply, mv in enumerate(g["moves"]):
            c = game.getCanonicalForm(b, pl)
            v = game.getValidMoves(c, 1)
            if not len(game.getWinningMoves(c, v)):
                par = game.getParryingMoves(c, v)
                if par is not None and len(par):
                    out.append(dict(board=c, valids=v, parry=set(int(a) for a in par), it=g["it"], ply=ply + 1))
                    break
            b, pl = game.getNextState(b, pl, mv["src"] * NUM_TILES + mv["dst"])
    random.Random(seed).shuffle(out)
    return out[:n]


def betolt(game, path):
    p = os.path.abspath(path)
    net = FullerenNNet(game, dict(DEFAULT_ARGS, checkpoint_dir=os.path.dirname(p)))
    _o, sys.stdout = sys.stdout, sys.stderr
    try:
        net.load_checkpoint(os.path.dirname(p), os.path.basename(p))
    finally:
        sys.stdout = _o
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--modellek", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--sims", type=int, default=150)
    a = ap.parse_args()

    game = FullerenSakkGame()
    pos = allasok(game, a.log, a.n)
    nv = st.mean(int(p["valids"].sum()) for p in pos)
    npar = st.mean(len(p["parry"]) for p in pos)
    print(f"# Fenyegetés-kivédés teszt – a háló önmagában\n")
    print(f"Napló: `{os.path.basename(a.log)}`, {len(pos)} állás (játszmánként az első fenyegetett, kivédhető "
          f"állás, nyerő lépés nélkül). Átlagosan {nv:.0f} szabályos lépésből {npar:.1f} kivédés "
          f"(véletlen lépés esélye ≈ {100*st.mean(len(p['parry'])/p['valids'].sum() for p in pos):.0f}%).\n")
    print(f"MCTS: {a.sims} szimuláció, cpuct 1,5, nyerés-ellenőrzés be, **fenyegetés-ellenőrzés ki**, zaj nélkül.\n")
    print("| Modell | Prior a kivédésekre (átlag) | Medián | Prior top-1 = kivédés | MCTS top-1 = kivédés | Idő |")
    print("|---|---|---|---|---|---|")
    margs = dotdict(dict(numMCTSSims=a.sims, cpuct=1.5, mcts_win_check=True, mcts_gamma=0.99,
                         mcts_threat_check=False))
    for mp in a.modellek:
        t0 = time.time()
        net = betolt(game, mp)
        mass, top1, mtop1 = [], 0, 0
        for p in pos:
            pi, _v = net.predict(p["board"])
            pi = pi * p["valids"]
            pi = pi / pi.sum()
            mass.append(sum(pi[x] for x in p["parry"]))
            top1 += int(np.argmax(pi)) in p["parry"]
            m = MCTS(game, net, margs)
            pr = m.getActionProb(p["board"], temp=0)
            mtop1 += int(np.argmax(pr)) in p["parry"]
        n = len(pos)
        print(f"| `{os.path.basename(mp)}` | {100*st.mean(mass):.1f}% | {100*st.median(mass):.1f}% | "
              f"{top1}/{n} ({100*top1/n:.0f}%) | {mtop1}/{n} ({100*mtop1/n:.0f}%) | {time.time()-t0:.0f} s |", flush=True)


if __name__ == "__main__":
    main()
