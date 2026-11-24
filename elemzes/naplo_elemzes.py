"""naplo_elemzes.py: az önjáték-napló (jatszma_naplo.txt) beolvasása és visszajátszása a szabálymotorral.

Minden állásban megvizsgálja, volt-e egylépéses zászlóütés és megtette-e a játékos, fenyegetett
állásban kivédte-e a fenyegetést, és elkerülhető baklövést követett-e el. A többi elemző program
modulként használja (parse, analyse, summarise, fmt, block).

Önálló futtatás:
    python naplo_elemzes.py <jatszma_naplo.txt> [--blokk 10]
"""
import re, sys, os, json, statistics as st
from collections import defaultdict, Counter
import numpy as np

TANITAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tanitas")
for _p in (TANITAS, os.path.join(TANITAS, "alpha-zero-general-master")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from game import FullerenSakkGame, NUM_TILES, CH_W_FLAG, CH_B_FLAG

G = FullerenSakkGame()

HDR = re.compile(r"^ITERÁCIÓ\s+(\d+) \| EPIZÓD\s+(\d+)\s+\[(\d\d:\d\d:\d\d)\]")
MV = re.compile(r"^\s+\[\s*(\d+)\] (FEHÉR|FEKETE)\s*\| (\S+)\s+(\d+)→\s*(\d+)\s+\[(.*?)\](?:\s+top: (.*))?$")
TOP = re.compile(r"(\d+)→(\d+)\(([\d.]+)\)")
RES = re.compile(r"Eredmény: (\S+) (GYŐZÖTT|DÖNTETLEN)?.*\((\d+) lépés\)")


def parse(path):
    games, cur = [], None
    for line in open(path, encoding="utf-8"):
        m = HDR.match(line)
        if m:
            cur = dict(it=int(m[1]), ep=int(m[2]), t=m[3], moves=[])
            games.append(cur); continue
        m = MV.match(line)
        if m and cur is not None:
            top = [(int(a) - 1, int(b) - 1, float(p)) for a, b, p in TOP.findall(m[7] or "")]
            cur["moves"].append(dict(pl=1 if m[2] == "FEHÉR" else -1, piece=m[3],
                                     src=int(m[4]) - 1, dst=int(m[5]) - 1, ev=m[6], top=top))
            continue
        m = RES.search(line)
        if m and cur is not None:
            cur["winner"] = {"FEHÉR": 1, "FEKETE": -1}.get(m[1], 0)
            cur["len"] = int(m[3])
    return games


def flag_captures(board, player):
    """Actions of `player` that capture the enemy flag (win in 1)."""
    v = G.getValidMoves(board, player)
    opp_flag = np.where(board[CH_B_FLAG if player == 1 else CH_W_FLAG] > 0)[0]
    if len(opp_flag) == 0:
        return set()
    f = int(opp_flag[0])
    return {a for a in np.nonzero(v)[0] if a % NUM_TILES == f}


def analyse(g):
    b, pl = G.getInitBoard(), 1
    r = dict(illegal=0, win_avail=0, win_taken=0, win_top1=0,
             threat_faced=0, threat_parried=0, blunder=0, blunder_avoidable=0,
             caps=0, swaps=0, top1p=[], chose_top1=0, n=0, end_by="?")
    for i, mv in enumerate(g["moves"]):
        a = mv["src"] * NUM_TILES + mv["dst"]
        valid = G.getValidMoves(b, pl)
        if mv["pl"] != pl or not valid[a]:
            r["illegal"] += 1
            break
        wins = flag_captures(b, pl)
        threatened = len(flag_captures(b, -pl)) > 0  # opponent could capture my flag if I pass
        if wins:
            r["win_avail"] += 1
            r["win_taken"] += a in wins
            if mv["top"] and (mv["top"][0][0] * NUM_TILES + mv["top"][0][1]) in wins:
                r["win_top1"] += 1
        nb, npl = G.getNextState(b, pl, a)
        if not wins:
            if threatened:
                r["threat_faced"] += 1
            opp_wins_after = len(flag_captures(nb, npl)) > 0 and G.getGameEnded(nb, npl) == 0
            if threatened and not opp_wins_after:
                r["threat_parried"] += 1
            if opp_wins_after:
                r["blunder"] += 1
                # was there a move that would not allow win-in-1?
                for alt in np.nonzero(valid)[0]:
                    ab, apl = G.getNextState(b, pl, alt)
                    if G.getGameEnded(ab, apl) != 0 or not flag_captures(ab, apl):
                        r["blunder_avoidable"] += 1
                        break
        if mv["ev"].startswith("ÜTÉS"):
            r["caps"] += 1
        if "CSERE" in mv["ev"]:
            r["swaps"] += 1
        if mv["top"]:
            r["top1p"].append(mv["top"][0][2])
            r["chose_top1"] += (mv["top"][0][0], mv["top"][0][1]) == (mv["src"], mv["dst"])
        r["n"] += 1
        if "GYŐZELEM" in mv["ev"]:
            r["end_by"] = mv["piece"].split("-")[1]
        b, pl = nb, npl
    return r


def block(it, size):
    lo = ((it - 1) // size) * size + 1
    return f"{lo:02d}-{lo + size - 1:02d}"


def summarise(games, keyf):
    groups = defaultdict(list)
    for g in games:
        groups[keyf(g)].append(g)
    rows = []
    for k in sorted(groups):
        gs = groups[k]
        L = [g["len"] for g in gs]
        A = [g["an"] for g in gs]
        s = lambda f: sum(a[f] for a in A)
        allp = [p for a in A for p in a["top1p"]]
        rows.append(dict(
            key=k, n=len(gs), white=sum(g["winner"] == 1 for g in gs), black=sum(g["winner"] == -1 for g in gs),
            draw=sum(g["winner"] == 0 for g in gs),
            mean=st.mean(L), med=st.median(L), mn=min(L), mx=max(L),
            caps=s("caps") / len(gs), swaps=s("swaps") / len(gs),
            top1p=st.mean(allp) if allp else 0,
            win_avail=s("win_avail"), win_taken=s("win_taken"), win_top1=s("win_top1"),
            threat=s("threat_faced"), parried=s("threat_parried"),
            blunder=s("blunder"), blunder_av=s("blunder_avoidable"),
            plies=s("n"), illegal=s("illegal"),
            end_by=Counter(a["end_by"] for a in A),
        ))
    return rows


def fmt(rows, title):
    out = [f"\n### {title}\n",
           "| Szakasz | N | Fehér nyer | Fekete nyer | Átl. hossz | Medián | Min–Max | Ütés/parti | Csere/parti | Átl. top‑1 π | Nyerő lépés elérhető → megtéve | MCTS top‑1 = nyerő | Fenyegetés → kivédve | Elkerülhető baklövés |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pct = lambda a, b: f"{a}/{b} ({100*a/b:.0f}%)" if b else "–"
        out.append(f"| {r['key']} | {r['n']} | {r['white']} ({100*r['white']/r['n']:.0f}%) | {r['black']} ({100*r['black']/r['n']:.0f}%) | "
                   f"{r['mean']:.1f} | {r['med']:.0f} | {r['mn']}–{r['mx']} | {r['caps']:.1f} | {r['swaps']:.1f} | {r['top1p']:.2f} | "
                   f"{pct(r['win_taken'], r['win_avail'])} | {pct(r['win_top1'], r['win_avail'])} | {pct(r['parried'], r['threat'])} | {r['blunder_av']} |")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Önjáték-napló taktikai elemzése")
    ap.add_argument("naplo", help="jatszma_naplo.txt")
    ap.add_argument("--blokk", type=int, default=10, help="iterációs blokkok mérete")
    a = ap.parse_args()
    games = [g for g in parse(a.naplo) if "len" in g]
    for g in games:
        g["an"] = analyse(g)
    print("szabálytalan lépések:", sum(g["an"]["illegal"] for g in games))
    print(fmt(summarise(games, lambda g: block(g["it"], a.blokk)), f"{a.blokk} iterációs blokkok"))
    print(fmt(summarise(games, lambda g: "ÖSSZES"), "Összesen"))
