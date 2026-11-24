"""
elemzes_kozos.py: az elemző programok közös segédfüggvényei (Wilson-intervallum, blokkosítás,
önjáték-napló összegzése).
"""
import math
import statistics as st

import naplo_elemzes as ne


def swap_loops(g):
    """A leghosszabb egymást követő zászlócsere-sorozat (ZÁSZLÓCSERE/VISSZACSERE) egy partiban."""
    best = cur = 0
    for mv in g["moves"]:
        if "CSERE" in mv["ev"]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def wilson(k, n, z=1.96):
    """95%-os Wilson-intervallum egy arányra."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def pct(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*p:.1f}% (95% CI {100*lo:.1f}–{100*hi:.1f}%)"


def I(r, k):
    return int(r[k]) if r[k] not in ("", None) else 0


def F(r, k):
    return float(r[k]) if r[k] not in ("", None) else None


def blocks(rows, size):
    out = {}
    for r in rows:
        b = (int(r["iter"]) - 1) // size
        out.setdefault(b, []).append(r)
    return [(f"{b*size+1:03d}–{b*size+size:03d}", v) for b, v in sorted(out.items())]


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def naplo_osszegzes(path, cim):
    games = [g for g in ne.parse(path) if "len" in g]
    for g in games:
        g["an"] = ne.analyse(g)
    s = lambda k: sum(g["an"][k] for g in games)  # noqa: E731
    print(f"\n### {cim}\n")
    print(f"- Játszmák: {len(games)}, szabálytalan: {s('illegal')}, döntetlen: {sum(g['winner'] == 0 for g in games)}, "
          f"leghosszabb: {max(g['len'] for g in games)} lépés, leghosszabb csere-sorozat: {max(swap_loops(g) for g in games)}")
    print(f"- Nyerő lépés elérhető → megtéve: {pct(s('win_taken'), s('win_avail'))}")
    print(f"- Fenyegetés → kivédve: {pct(s('threat_parried'), s('threat_faced'))}")
    print(f"- Elkerülhető baklövés: {s('blunder_avoidable')} ({s('blunder_avoidable')/len(games):.2f}/parti)")
    print(f"- Világos nyer: {pct(sum(g['winner'] == 1 for g in games), sum(g['winner'] != 0 for g in games))}")
    return games
