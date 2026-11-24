"""
jatszma_export.py — egy naplózott játszma kiírása a GUI formátumában (FEN72 + lépéslista).

A FEN72 egyetlen állást ír le, ezért egy játszmából egy állás-sorozat lesz:
  * <kimenet>.json : {"kezdo_fen", "lepesek": [[honnan, hova], …], "fenek": [minden lépés utáni FEN],
                      "gyoztes", "mod", "forras", …}
  * <kimenet>.fen  : soronként egy FEN (a kezdőállás, majd minden lépés után)
A mezőszámok 1-alapúak, a zászlócsere a zászló→bábu irányban szerepel (pl. [1, 2]).
Az órák helyén 600.0/600.0 áll (az önjáték-napló időt nem rögzít).

Használat:
    python jatszma_export.py <jatszma_naplo.txt> --lista              # a napló játszmái
    python jatszma_export.py <jatszma_naplo.txt> --iter 50 --ep 100   # egy játszma kiírása
    python jatszma_export.py <jatszma_naplo.txt> --iter 50 --ep 100 --ki jatszma_50_100
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gepi_jatekos"))
sys.path.insert(0, HERE)

import fen72  # noqa: E402
import naplo_elemzes as ne  # noqa: E402
from game import NUM_TILES  # noqa: E402

G = fen72._game()


def export(g, forras):
    """Napló-játszma → (dict a JSON-hoz, FEN-lista). A lépéseket a szabálymotorral játssza vissza."""
    board, player = G.getInitBoard(), 1
    fens = [fen72.to_fen(board, player, 1)]
    moves = []
    for i, mv in enumerate(g["moves"]):
        a = mv["src"] * NUM_TILES + mv["dst"]
        if mv["pl"] != player or not G.getValidMoves(board, player)[a]:
            raise ValueError(f"Szabálytalan lépés a naplóban: {i + 1}. lépés {mv['src'] + 1}→{mv['dst'] + 1}")
        moves.append([mv["src"] + 1, mv["dst"] + 1])
        board, player = G.getNextState(board, player, a)
        fens.append(fen72.to_fen(board, player, i + 2))
    gy = {1: "vilagos", -1: "sotet", 0: None}[g.get("winner", 0)]
    data = dict(forras=forras, iter=g["it"], ep=g["ep"], lepesszam=len(moves), gyoztes=gy,
                mod="Zászló leütve" if gy else "döntetlen",
                kezdo_fen=fens[0], lepesek=moves, fenek=fens[1:])
    return data, fens


def main():
    ap = argparse.ArgumentParser(description="Naplózott játszma → FEN72 + lépéslista")
    ap.add_argument("naplo", help="jatszma_naplo.txt")
    ap.add_argument("--lista", action="store_true", help="a játszmák listája (iter, ep, hossz, győztes)")
    ap.add_argument("--iter", type=int)
    ap.add_argument("--ep", type=int)
    ap.add_argument("--ki", help="kimeneti fájlnév kiterjesztés nélkül (alap: jatszma_<iter>_<ep> a napló mellett)")
    a = ap.parse_args()

    games = [g for g in ne.parse(a.naplo) if "len" in g]
    if a.lista or a.iter is None:
        print("iter  ep   hossz  győztes")
        for g in games:
            w = {1: "világos", -1: "sötét", 0: "döntetlen"}[g.get("winner", 0)]
            print(f"{g['it']:4d} {g['ep']:4d}  {g['len']:5d}  {w}")
        return
    sel = [g for g in games if g["it"] == a.iter and (a.ep is None or g["ep"] == a.ep)]
    if not sel:
        sys.exit(f"Nincs ilyen játszma: iter {a.iter}, ep {a.ep}")
    g = sel[0]
    data, fens = export(g, os.path.basename(a.naplo))
    base = a.ki or os.path.join(os.path.dirname(os.path.abspath(a.naplo)), f"jatszma_{g['it']}_{g['ep']}")
    json.dump(data, open(base + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(base + ".fen", "w", encoding="utf-8").write("\n".join(fens) + "\n")
    print(f"{data['lepesszam']} lépés, győztes: {data['gyoztes']}")
    print(f"Kiírva: {base}.json és {base}.fen")
    print("Utolsó állás:", fens[-1])


if __name__ == "__main__":
    main()
