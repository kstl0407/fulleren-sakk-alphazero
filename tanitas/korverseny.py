"""
korverseny.py — modellek körmérkőzése párosított partikkal (v4, 2026-10-19).

Minden modellpár `--parok` nyitáspárt játszik: ugyanabból a sorsolt nyitásból (`--nyito`
véletlen féllépés) egy parti az egyik, egy a másik modell világosával. Így a kezdőelőny
(~80% világos) kiesik az összevetésből.

    python korverseny.py ../checkpoint_32.pth.tar ../checkpoint_59.pth.tar ... --parok 20 --sims 50
    python korverseny.py --mappa .. --parok 20            # a mappa összes *.pth.tar fájlja
    python korverseny.py moho veletlen ../best.pth.tar    # alapvonalak is: mohó és véletlen játékos
    python korverseny.py v3.pth.tar@5 v4.pth.tar@50 v5.pth.tar@400   # játékosonkénti szimulációszám (szintek)

Kimenet: Markdown-táblázat (stdout) – pontszám (megnyert partik aránya), megnyert/elvesztett párok,
párosított eredménymátrix, valamint a világos győzelmi aránya (kezdőelőny) összesítve.
GPU-n gyors; CPU-n (50 szim.) kb. 1–2 perc / modellpár / 10 pár.
"""
import argparse
import glob
import itertools
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "alpha-zero-general-master"))

from ertekeles import MCTSJatekos, MohoJatekos, paros_parbaj  # noqa: E402
from game import FullerenSakkGame  # noqa: E402
from nnet import DEFAULT_ARGS, FullerenNNet  # noqa: E402
from utils import dotdict  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


ALAPVONALAK = ("moho", "veletlen")


class VeletlenJatekos:
    def __init__(self, game, rng):
        self.game, self.rng = game, rng

    def __call__(self, canonical_board):
        return int(self.rng.choice(np.flatnonzero(self.game.getValidMoves(canonical_board, 1))))


def sajat(margs, n):
    """A közös MCTS-beállítás, saját szimulációszámmal (ha meg van adva)."""
    return margs if n is None else dotdict(dict(margs, numMCTSSims=n))


def jatekos(game, net, margs, rng):
    """MCTS-játékos egy hálóval, vagy a mohó / véletlen alapvonal (`net` ilyenkor a neve)."""
    if net == "moho":
        return MohoJatekos(game, seed=int(rng.randint(1 << 30)))
    if net == "veletlen":
        return VeletlenJatekos(game, np.random.RandomState(int(rng.randint(1 << 30))))
    return MCTSJatekos(game, net, margs, temp_moves=0)


def main():
    ap = argparse.ArgumentParser(description="Modellek körmérkőzése párosított partikkal")
    ap.add_argument("modellek", nargs="*", help=".pth.tar fájlok (opcionálisan @szimulációszám), vagy a 'moho' / 'veletlen' alapvonal")
    ap.add_argument("--mappa", help="ebből a mappából az összes *.pth.tar")
    ap.add_argument("--parok", type=int, default=20, help="nyitáspárok modellpáronként (2 parti/pár)")
    ap.add_argument("--nyito", type=int, default=2, help="véletlen nyitó féllépések száma")
    ap.add_argument("--sims", type=int, default=50, help="MCTS-szimulációk lépésenként")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = list(a.modellek)
    if a.mappa:
        paths += sorted(glob.glob(os.path.join(a.mappa, "*.pth.tar")))
    paths = [p for p in dict.fromkeys(paths) if "temp.pth" not in p]
    if len(paths) < 2:
        sys.exit("Legalább két modell kell.")

    game = FullerenSakkGame()
    margs = dotdict(dict(numMCTSSims=a.sims, cpuct=1.5, mcts_win_check=True, mcts_gamma=0.99))
    nets, names, sims = [], [], []
    for p in paths:
        if p in ALAPVONALAK:  # háló nélküli játékos: a nets-ben a neve áll
            nets.append(p)
            names.append(p)
            sims.append(None)
            continue
        p, _, s = p.rpartition("@") if "@" in os.path.basename(p) else (p, "", "")
        sims.append(int(s) if s else None)
        n = FullerenNNet(game, dict(DEFAULT_ARGS, checkpoint_dir=os.path.dirname(os.path.abspath(p))))
        _out, sys.stdout = sys.stdout, sys.stderr  # a load_checkpoint print-je ne kerüljön a táblázatba
        try:
            n.load_checkpoint(os.path.dirname(os.path.abspath(p)), os.path.basename(p))
        finally:
            sys.stdout = _out
        nets.append(n)
        names.append(os.path.basename(p).replace(".pth.tar", "") + (f"@{s}" if s else ""))

    rng = np.random.RandomState(a.seed)
    k = len(nets)
    pont = np.zeros(k)          # megnyert partik
    partik = np.zeros(k)
    par_ny = np.zeros((k, k), dtype=int)   # i megnyert párjai j ellen
    feher_ny = osszes = 0
    t0 = time.time()
    for i, j in itertools.combinations(range(k), 2):
        pi = jatekos(game, nets[i], sajat(margs, sims[i]), rng)
        pj = jatekos(game, nets[j], sajat(margs, sims[j]), rng)
        r = paros_parbaj(game, pi, pj, a.parok, a.nyito, rng)
        wi, wj = r["a_vilagos"] + r["a_sotet"], r["b_vilagos"] + r["b_sotet"]
        pont[i] += wi
        pont[j] += wj
        partik[i] += 2 * a.parok
        partik[j] += 2 * a.parok
        par_ny[i, j], par_ny[j, i] = r["a_par"], r["b_par"]
        feher_ny += r["a_vilagos"] + r["b_vilagos"]
        osszes += 2 * a.parok - r["dontetlen"]
        print(f"  {names[i]} – {names[j]}: partik {wi}:{wj}, párok {r['a_par']}:{r['b_par']} "
              f"(döntetlen pár {r['dontetlen_par']})  [{time.time() - t0:.0f} s]", file=sys.stderr, flush=True)

    print(f"# Körmérkőzés – {k} modell, {a.parok} nyitáspár / modellpár, {a.nyito} véletlen nyitó féllépés, "
          f"{a.sims} szimuláció\n")
    order = np.argsort(-pont / partik)
    print("| Hely | Modell | Megnyert partik | Arány (95% CI) | Megnyert párok | Elvesztett párok |")
    print("|---|---|---|---|---|---|")
    for h, i in enumerate(order, 1):
        lo, hi = wilson(pont[i], partik[i])
        print(f"| {h} | {names[i]} | {int(pont[i])}/{int(partik[i])} | {100*pont[i]/partik[i]:.1f}% "
              f"({100*lo:.0f}–{100*hi:.0f}%) | {par_ny[i].sum()} | {par_ny[:, i].sum()} |")
    print("\n**Megnyert párok (sor modell a oszlop modell ellen):**\n")
    print("| | " + " | ".join(names[i] for i in order) + " |")
    print("|---" * (k + 1) + "|")
    for i in order:
        print(f"| {names[i]} | " + " | ".join("–" if i == j else str(par_ny[i, j]) for j in order) + " |")
    lo, hi = wilson(feher_ny, osszes)
    print(f"\n**Kezdőelőny:** a világos {feher_ny}/{osszes} döntött partit nyert = {100*feher_ny/osszes:.1f}% "
          f"(95% CI {100*lo:.1f}–{100*hi:.1f}%). Futásidő: {(time.time() - t0)/60:.1f} perc.")


if __name__ == "__main__":
    main()
