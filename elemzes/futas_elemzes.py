"""
futas_elemzes.py: egy tanítási futás (v5/v6 formátumú kimenet) összesítő elemzése, opcionálisan
összevetve egy másik futással.

A futás mappájában ezeket a fájlokat várja (a main.py ezeket írja a checkpoint-mappába):
    iteraciok.csv      iterációnkénti összesítő
    tanitas*.log       a tanítás konzolnaplója (csak az időtartamhoz, elhagyható)
    jatszma_naplo.txt  minden 10. önjáték-parti lépésenként (elhagyható)

Futtatás:
    python futas_elemzes.py <futas_mappa> --nev v6h [--osszevet <masik_mappa> --osszevet-nev v6] [--blokk 50]
A kimenet Markdown, pl. ... > elemzes_kimenet.md
"""
import argparse
import csv
import glob
import math
import os
import random
import re
import statistics as st
from collections import Counter
from datetime import datetime

import elemzes_kozos as ek

pct, I, blocks = ek.pct, ek.I, ek.blocks


def csv_be(mappa):
    return list(csv.DictReader(open(os.path.join(mappa, "iteraciok.csv"), encoding="utf-8")))


def arena_h0(n, q, alfa=0.10, hatar=0.4, parok=40, ismetles=20000, seed=3):
    """Várható átjutás, ha minden jelölt pontosan olyan erős, mint a legjobb (előjelpróbás aréna)."""
    rnd = random.Random(seed)
    hits = 0
    for _ in range(ismetles):
        b = a = 0
        for _ in range(parok):
            if rnd.random() < q:
                b, a = (b + 1, a) if rnd.random() < 0.5 else (b, a + 1)
        p = 1.0 if b + a == 0 else sum(math.comb(b + a, k) for k in range(b, b + a + 1)) / 2 ** (b + a)
        if b > a and p <= alfa:
            hits += 1
            continue
        if b > a and p <= hatar:
            for _ in range(parok):
                if rnd.random() < q:
                    b, a = (b + 1, a) if rnd.random() < 0.5 else (b, a + 1)
            p = sum(math.comb(b + a, k) for k in range(b, b + a + 1)) / 2 ** (b + a)
            hits += b > a and p <= alfa
    return hits / ismetles, n * hits / ismetles


def main():
    ap = argparse.ArgumentParser(description="Tanítási futás elemzése")
    ap.add_argument("mappa", help="a futás kimeneti mappája (iteraciok.csv, tanitas*.log, jatszma_naplo.txt)")
    ap.add_argument("--nev", default="futás", help="a futás neve a kimenetben")
    ap.add_argument("--osszevet", default=None, help="egy másik futás mappája összevetéshez")
    ap.add_argument("--osszevet-nev", default="összevetés")
    ap.add_argument("--blokk", type=int, default=10, help="iterációs blokkok mérete")
    a = ap.parse_args()

    rows = csv_be(a.mappa)
    mas = csv_be(a.osszevet) if a.osszevet else []
    n = len(rows)
    nev, mnev = a.nev, a.osszevet_nev
    print(f"# {nev} – elemzési kimenet\n")

    # ── idő ──────────────────────────────────────────────────────────────
    print("## Futás\n")
    tt = [sum(I(r, k) for k in ("onjatek_mp", "tanitas_mp", "arena_mp", "ertekeles_mp")) for r in rows]
    logok = sorted(glob.glob(os.path.join(a.mappa, "tanitas*.log")))
    ido = ""
    if logok:
        log = re.sub(r"\x1b\[[0-9;]*m", "", open(logok[0], encoding="utf-8", errors="replace").read())
        stamps = [datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
                  for m in re.finditer(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", log, re.M)]
        if stamps:
            ido = (f"; {stamps[0]:%Y-%m-%d %H:%M} – {stamps[-1]:%Y-%m-%d %H:%M} "
                   f"({(stamps[-1] - stamps[0]).total_seconds()/3600:.1f} óra)")
    print(f"- {n} iteráció{ido}; átl. {st.mean(tt)/60:.1f} perc/iteráció "
          f"(önjáték {st.mean(I(r,'onjatek_mp') for r in rows)/60:.1f} perc, tanítás {st.mean(I(r,'tanitas_mp') for r in rows):.0f} s, "
          f"aréna {st.mean(I(r,'arena_mp') for r in rows)/60:.1f} perc, kapu {st.mean(I(r,'ertekeles_mp') for r in rows):.0f} s)")
    acc = [r for r in rows if r["elfogadva"] == "1"]
    ref_rej = [r for r in rows if r["elutasitas_oka"] == "referencia"]
    passed = acc + ref_rej
    print(f"- Elfogadva: {len(acc)}/{n}: {', '.join(r['iter'] for r in acc)}")
    print(f"- Arénán átjutott: {len(passed)}/{n}; ebből a kapu elutasította: {len(ref_rej)} "
          f"({', '.join(r['iter'] for r in ref_rej)})")
    if mas:
        print(f"- ({mnev}: {sum(r['elfogadva'] == '1' for r in mas)}/{len(mas)} elfogadva)")

    # ── aréna ────────────────────────────────────────────────────────────
    print("\n## Előjelpróbás aréna\n")
    ext = [r for r in rows if I(r, "arena_extra_par") > 0]
    print(f"- Hosszabbítás (+40 pár): {len(ext)} iterációban; ebből átjutott {sum(r['ref_jelolt'] != '' for r in ext)}")
    ps = [float(r["arena_p"]) for r in rows if r["arena_p"] != ""]
    for th in (0.05, 0.10, 0.25, 0.5):
        print(f"- p ≤ {th:.2f}: {sum(p <= th for p in ps)} iteráció")
    up = [I(r, "arena_uj_par") for r in rows]
    ep = [I(r, "arena_elozo_par") for r in rows]
    tobb = ""
    if mas:
        mu = sum(I(r, "arena_uj_par") for r in mas)
        md = sum(I(r, "arena_uj_par") + I(r, "arena_elozo_par") for r in mas)
        tobb = f" ({mnev}: {pct(mu, md)})"
    print(f"- Új modell a döntött párokból összesen: {pct(sum(up), sum(up) + sum(ep))}{tobb}")
    npairs = [40 + I(r, "arena_extra_par") for r in rows]
    dec = [u + e for u, e in zip(up, ep)]
    print(f"- Átl. párok/iteráció {st.mean(npairs):.1f}; döntött pár arány {sum(dec)/sum(npairs):.3f}")
    aw = sum(I(r, "uj_vilagoskent") + I(r, "elozo_vilagoskent") for r in rows)
    ab = sum(I(r, "uj_sotetkent") + I(r, "elozo_sotetkent") for r in rows)
    print(f"- Világos nyer az arénában: {pct(aw, aw + ab)}")
    arany, vart = arena_h0(n, sum(dec) / sum(npairs))
    print(f"- H0 (minden jelölt pontosan olyan erős, mint a legjobb): várható átjutás {100*arany:.1f}% → "
          f"{vart:.1f} / {n}; mért: {len(passed)}")

    # ── kapu ─────────────────────────────────────────────────────────────
    if passed and passed[0]["ref_jelolt"] != "":
        print("\n## Újramérő referencia-kapu (jelölt és legjobb ugyanazokon a nyitáspárokon a referencia ellen)\n")
        print("| Iter | Aréna párok (új:előző:dönt.) | p | Jelölt | Legjobb (friss) | Különbség | Döntés |")
        print("|---|---|---|---|---|---|---|")
        for r in passed:
            c, b = float(r["ref_jelolt"]), float(r["ref_legjobb"])
            print(f"| {r['iter']} | {r['arena_uj_par']}:{r['arena_elozo_par']}:{r['arena_dontetlen_par']} | {r['arena_p']} | "
                  f"{c:.3f} | {b:.3f} | {c-b:+.3f} | {'ELFOGADVA' if r['elfogadva'] == '1' else 'elutasítva (kapu)'} |")
        cs = [float(r["ref_jelolt"]) for r in passed]
        bs = [float(r["ref_legjobb"]) for r in passed]
        if len(bs) > 1:
            print(f"\n- Jelölt átl. {st.mean(cs):.3f}, legjobb (friss) átl. {st.mean(bs):.3f} (szórás {st.stdev(bs):.3f}); "
                  f"min {min(bs):.3f}, max {max(bs):.3f}")

    # ── blokkok ──────────────────────────────────────────────────────────
    print(f"\n## {a.blokk} iterációs blokkok\n")
    print("| Iter | Elfog. | Arénán át | Hosszabbítás | Aréna párok új:előző | Önjáték világos nyer | Átl. hossz | Döntetlen | π-loss | v-loss |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, b in blocks(rows, a.blokk):
        print(f"| {name} | {sum(r['elfogadva'] == '1' for r in b)} | {sum(bool(r['ref_jelolt']) for r in b)} | "
              f"{sum(I(r,'arena_extra_par') > 0 for r in b)} | {sum(I(r,'arena_uj_par') for r in b)}:{sum(I(r,'arena_elozo_par') for r in b)} | "
              f"{sum(I(r,'onjatek_vilagos_nyer') for r in b)}/{sum(I(r,'onjatek_partik') for r in b)} | "
              f"{st.mean(float(r['onjatek_atl_hossz']) for r in b):.1f} | {sum(I(r,'onjatek_dontetlen') for r in b)} | "
              f"{st.mean(float(r['pi_loss']) for r in b):.3f} | {st.mean(float(r['v_loss']) for r in b):.3f} |")

    print(f"\n## Önjáték: {nev}" + (f" és {mnev}" if mas else "") + "\n")
    for key, lab in (("onjatek_atl_hossz", "átl. partihossz"), ("pi_loss", "π-loss"), ("v_loss", "v-loss")):
        x = [float(r[key]) for r in rows]
        sor = f"- {lab}: {nev} {st.mean(x):.3f} (első 20 iteráció {st.mean(x[:20]):.3f}, utolsó 20 {st.mean(x[-20:]):.3f})"
        if mas:
            sor += f"; {mnev} {st.mean(float(r[key]) for r in mas):.3f}"
        print(sor)
    for lab, rr in ((nev, rows), (mnev, mas)):
        if rr:
            w = sum(I(r, "onjatek_vilagos_nyer") for r in rr)
            d = sum(I(r, "onjatek_partik") - I(r, "onjatek_dontetlen") for r in rr)
            print(f"- {lab}: világos nyer (döntött partik) {pct(w, d)}; döntetlen "
                  f"{sum(I(r,'onjatek_dontetlen') for r in rr)} (ismétlés {sum(I(r,'onjatek_ismetles') for r in rr)})")

    # ── napló ────────────────────────────────────────────────────────────
    naplo = os.path.join(a.mappa, "jatszma_naplo.txt")
    if not os.path.exists(naplo):
        return
    print("\n## Önjáték-napló (minden 10. parti)\n")
    games = ek.naplo_osszegzes(naplo, nev)
    if a.osszevet and os.path.exists(os.path.join(a.osszevet, "jatszma_naplo.txt")):
        ek.naplo_osszegzes(os.path.join(a.osszevet, "jatszma_naplo.txt"), mnev)
    ne = ek.ne
    print(ne.fmt(ne.summarise(games, lambda g: ne.block(g["it"], a.blokk)), f"{nev} – {a.blokk} iterációs blokkok"))
    print("\n### Nyitás: a 3. féllépés változatossága\n")
    print("| Iter-blokk | Partik | Különböző 3. lépés | Leggyakoribb |")
    print("|---|---|---|---|")
    for lo in range(1, n + 1, a.blokk):
        sub = [g for g in games if lo <= g["it"] < lo + a.blokk and len(g["moves"]) >= 3]
        c = Counter((g["moves"][2]["src"] + 1, g["moves"][2]["dst"] + 1) for g in sub)
        print(f"| {lo:03d}–{lo+a.blokk-1:03d} | {len(sub)} | {len(c)} | " +
              ", ".join(f"{x}→{y} ×{k}" for (x, y), k in c.most_common(3)) + " |")
    h = Counter(min(g["len"] // 5 * 5, 60) for g in games)
    print("\nHossz-hisztogram: " + ", ".join(f"{k}–{k+4}: {v}" for k, v in sorted(h.items())))


if __name__ == "__main__":
    main()
