"""
vakfolt_teszt.py — az "MCTS nem üti le a zászlót" hipotézis közvetlen ellenőrzése.

Olyan állásokat vesz a játszmanaplóból, ahol a soron lévő félnek volt
egylépéses nyerése (zászlóütés), és megnézi:
  1. a háló priorját (P) a nyerő lépésre: értéke, rangja a szabályos lépések között;
  2. a háló értékbecslését (v) az állásra;
  3. friss MCTS-t (zaj nélkül, mint az arénában) különböző szimulációszámokkal:
     a legtöbbet látogatott lépés nyerő-e, és hányszor látogatta a nyerő lépést.

Futtatás (torch kell):
    python vakfolt_teszt.py --log <jatszma_naplo.txt> --n 60 --sims 50 150 400 800
    python vakfolt_teszt.py --log <jatszma_naplo.txt> --model ../modellek/v6_best.pth.tar --mcts javitott
A dolgozat mérései a 3. futás naplójának 41. iteráció utáni partijaiból vett 60 állást használták,
minden modellnél ugyanazokat (azonos --log, --min-iter, --n és seed).
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
for _p in (CODE, os.path.join(CODE, "alpha-zero-general-master"), os.path.join(REPO, "gepi_jatekos"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import naplo_elemzes as ne  # noqa: E402
from game import NUM_TILES  # noqa: E402
from MCTS import MCTS  # noqa: E402
from nnet import DEFAULT_ARGS, FullerenNNet  # noqa: E402
from utils import dotdict  # noqa: E402


def collect_positions(log_path, min_iter, n, seed=0):
    """Return up to n (board, player, wins, meta) positions with a win-in-1 available."""
    games = [g for g in ne.parse(log_path) if "len" in g and g["it"] >= min_iter]
    pos = []
    for g in games:
        b, pl = ne.G.getInitBoard(), 1
        for ply, mv in enumerate(g["moves"]):
            wins = ne.flag_captures(b, pl)
            if wins:
                a = mv["src"] * NUM_TILES + mv["dst"]
                pos.append(dict(board=b, pl=pl, wins=wins, it=g["it"], ep=g["ep"], ply=ply + 1,
                                played=a, taken=a in wins))
                break  # csak a játszma első ilyen állása (független minták)
            b, pl = ne.G.getNextState(b, pl, mv["src"] * NUM_TILES + mv["dst"])
    random.Random(seed).shuffle(pos)
    return pos[:n]


def mv_str(a):
    return f"{a // NUM_TILES + 1}→{a % NUM_TILES + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO, "modellek", "v6_best.pth.tar"))
    ap.add_argument("--log", required=True, help="önjáték-napló (jatszma_naplo.txt), ebből vesz állásokat")
    ap.add_argument("--min-iter", type=int, default=41)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sims", type=int, nargs="+", default=[50, 150, 400])
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--mcts", choices=["eredeti", "diszkont", "javitott"], default="eredeti",
                    help="eredeti = alpha-zero-general MCTS; diszkont = csak gamma; javitott = gamma + nyerés-ellenőrzés")
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args()
    MCTSCls = MCTS
    extra = {}
    if a.mcts != "eredeti":
        from mcts_javitott import MCTSJavitott as MCTSCls
        extra = dict(mcts_gamma=a.gamma, mcts_win_check=(a.mcts == "javitott"))

    nnet = FullerenNNet(ne.G, dict(DEFAULT_ARGS, checkpoint_dir=HERE))
    nnet.load_checkpoint(os.path.dirname(a.model), os.path.basename(a.model))
    pos = collect_positions(a.log, a.min_iter, a.n)
    print(f"# Vakfolt-teszt\n\nModell: `{os.path.basename(a.model)}`, állások: {len(pos)} "
          f"(iter ≥ {a.min_iter}, játszmánként az első nyerési lehetőség), cpuct={a.cpuct}, MCTS: {a.mcts}"
          + (f" (gamma={a.gamma})" if a.mcts != "eredeti" else "") + "\n")

    rows, res = [], {s: [] for s in a.sims}
    priors, ranks, values = [], [], []
    for p in pos:
        canon = ne.G.getCanonicalForm(p["board"], p["pl"])
        P, v = nnet.predict(canon)
        valids = ne.G.getValidMoves(canon, 1)
        Pm = P * valids
        Pm /= Pm.sum()
        legal = np.nonzero(valids)[0]
        order = legal[np.argsort(-Pm[legal])]
        best_w = max(p["wins"], key=lambda w: Pm[w])
        rank = int(np.where(order == best_w)[0][0]) + 1
        priors.append(float(Pm[best_w]))
        ranks.append(rank)
        values.append(float(v))
        row = [f"{p['it']}/{p['ep']}/{p['ply']}", "F" if p["pl"] == 1 else "S", mv_str(best_w),
               f"{Pm[best_w]:.4f}", f"{rank}/{len(legal)}", mv_str(order[0]) + f" ({Pm[order[0]]:.2f})",
               f"{v:+.2f}", "igen" if p["taken"] else "nem"]
        for s in a.sims:
            m = MCTSCls(ne.G, nnet, dotdict(dict(numMCTSSims=s, cpuct=a.cpuct, **extra)))
            t0 = time.time()
            m.getActionProb(canon, temp=1)
            key = ne.G.stringRepresentation(canon)
            counts = np.array([m.Nsa.get((key, x), 0) for x in range(ne.G.getActionSize())])
            top = int(np.argmax(counts))
            nw = int(sum(counts[w] for w in p["wins"]))
            ok = top in p["wins"]
            res[s].append((ok, nw, time.time() - t0))
            row.append(f"{'✔' if ok else '✘'} {nw}/{s}")
        rows.append(row)
        print("  ", " | ".join(row), file=sys.stderr, flush=True)

    print("| Állás (it/ep/lépés) | Ki | Nyerő lépés | P(nyerő) | Rang | Háló top-1 | v | Naplóban megtette | "
          + " | ".join(f"MCTS {s} sim: top-1 nyerő? (látog.)" for s in a.sims) + " |")
    print("|" + "---|" * (8 + len(a.sims)))
    for r in rows:
        print("| " + " | ".join(r) + " |")

    print("\n## Összegzés\n")
    print(f"- P(nyerő lépés) medián: {st.median(priors):.4f}, átlag: {st.mean(priors):.4f}; "
          f"egyenletes prior lenne ≈ 1/{round(st.mean(len(np.nonzero(ne.G.getValidMoves(ne.G.getCanonicalForm(p['board'], p['pl']), 1))[0]) for p in pos))}")
    print(f"- A nyerő lépés rangja a háló szerint: medián {st.median(ranks)}, top-1: {sum(r == 1 for r in ranks)}/{len(ranks)}, "
          f"top-5: {sum(r <= 5 for r in ranks)}/{len(ranks)}")
    print(f"- v (a soron lévő fél szemszögéből, nyerő állásban ideálisan ≈ +1): medián {st.median(values):+.2f}")
    print(f"- Naplóban (önjáték, zajjal) megtette: {sum(p['taken'] for p in pos)}/{len(pos)}")
    for s in a.sims:
        ok = sum(x[0] for x in res[s])
        nv = sum(x[1] > 0 for x in res[s])
        print(f"- MCTS {s} szim.: top-1 nyerő {ok}/{len(pos)} ({100*ok/len(pos):.0f}%), "
              f"a nyerő lépést egyszer sem látogatta: {len(pos)-nv}/{len(pos)}, "
              f"idő/lépés: {st.mean(x[2] for x in res[s]):.2f} s (CPU)")


if __name__ == "__main__":
    main()
