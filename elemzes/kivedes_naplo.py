"""
kivedes_naplo.py — a naplóbeli „Fenyegetés → kivédve” arány felbontása (v4 elemzés, 2026-10-21).

Az ne.analyse minden fenyegetett állást számol, akkor is, ha a fenyegetés nem védhető ki.
Itt a v4 egzakt getParryingMoves-ával szétválasztjuk: kivédhető-e, és ha igen, kivédte-e.
Futtatás: python kivedes_naplo.py <jatszma_naplo.txt> [<masik_naplo.txt> ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = os.path.join(REPO, "tanitas")
for _p in (CODE, os.path.join(CODE, "alpha-zero-general-master"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from game import NUM_TILES, FullerenSakkGame  # noqa: E402
import naplo_elemzes as ne  # noqa: E402  


def main():
    game = FullerenSakkGame()
    print("| Napló | Fenyegetett állás (nyerés nélkül) | Ebből kivédhető | Kivédte (kivédhetőből) | Kivédhetetlen |")
    print("|---|---|---|---|---|")
    for path in sys.argv[1:]:
        fen = ved = kiv = 0
        for g in ne.parse(path):
            if "len" not in g:
                continue
            b, pl = game.getInitBoard(), 1
            for mv in g["moves"]:
                a = mv["src"] * NUM_TILES + mv["dst"]
                c = game.getCanonicalForm(b, pl)
                v = game.getValidMoves(c, 1)
                if not len(game.getWinningMoves(c, v)):
                    par = game.getParryingMoves(c, v)
                    if par is not None:
                        fen += 1
                        if len(par):
                            ved += 1
                            kiv += a in set(int(x) for x in par)
                b, pl = game.getNextState(b, pl, a)
        print(f"| `{os.path.basename(path)}` | {fen} | {ved} ({100*ved/fen:.0f}%) | {kiv}/{ved} ({100*kiv/ved:.1f}%) | {fen-ved} |")


if __name__ == "__main__":
    main()
