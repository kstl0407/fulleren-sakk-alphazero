"""
ai_jatekos.py — a betanított AlphaZero-modell gépi játékosként a GUI adapteréhez.

Protokoll (a kari játékfelület leírása szerint): soronként egy JSON a stdin-en, egy válaszsor a stdout-ra.
    {"tipus":"lepj","fen":"…","szin":"vilagos","lepesek":[[1,2],…],"idokorlat_ms":5000}
        → [honnan, hova]
    {"tipus":"vege",…}  → nincs válasz, a program kilép.
A stdout-ra CSAK a válasz megy, minden napló a stderr-re (vagy --log fájlba).

Lépésválasztás (sorrendben):
  1. Ha van zászlóütés (egylépéses nyerés) a listában → azonnal azt lépjük.
     Ez a betanított háló ismert vakfoltját fedi el (a keresés a nyerő lépést néha meg sem vizsgálja).
  2. MCTS a hálóval, időkorláttal (az idokorlat_ms egy része), zaj nélkül; alapértelmezésben a
     mcts_javitott.py változat (nyerés-ellenőrzés a fában + diszkontálás), --mcts eredeti a régi.
  3. Biztonsági szűrő: a legtöbbet látogatott lépések közül az elsőt választjuk,
     amely után az ellenfélnek NINCS egylépéses zászlóütése (ha van ilyen).
  4. Bármilyen hiba esetén: véletlen (de biztonságos, ha lehet) szabályos lépés.

Futtatás:
    python ai_jatekos.py                       # best.pth.tar ebből a mappából, ha nincs: ../modellek/v6_best.pth.tar
    python ai_jatekos.py --model best.pth.tar --max-sims 800 --ido-arany 0.6
    python ai_jatekos.py --halo-nelkul         # csak heurisztika (torch nélkül is fut)
Kézi próba:
    echo '{"tipus":"lepj","fen":"1Z0/2F/3F/4F/5F/6F/8B/14B/20B/26B/32B/38b/44b/50b/56b/62b/67z0/68f/69f/70f/71f/72f v 600.0/600.0 1","szin":"vilagos","lepesek":[[1,2],[8,9]],"idokorlat_ms":5000}' | python ai_jatekos.py
"""
import argparse
import json
import os
import random
import sys
import time

T_INDULAS = time.time()  # a folyamat indulása (az első lépés időkeretéhez)
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import fen72  # noqa: E402  (beállítja a tanitas/ elérési utat is)
from game import NUM_TILES  # noqa: E402


def _alap_modell():
    """Önálló csomagban a mappa best.pth.tar fájlja, a tárolóban a modellek/ mappa v6-os modellje."""
    helyi = os.path.join(HERE, "best.pth.tar")
    if os.path.exists(helyi):
        return helyi
    return os.path.join(os.path.dirname(HERE), "modellek", "v6_best.pth.tar")


def log(*a):
    print("[ai]", *a, file=sys.stderr, flush=True)


class Jatekos:
    def __init__(self, args):
        self.args = args
        self.game = fen72._game()
        self.nnet = None
        self.mcts = None
        self.rng = random.Random(args.seed)
        # A modellt induláskor, az üzenetek olvasása ELŐTT, szinkron töltjük be. (Háttérszálas betöltéssel
        # próbálkoztunk: a torch importja hosszan fogja a GIL-t, és a fő szál emiatt lépések közben
        # >1 s-ig nem tudott válaszolni.) Ha az első `lepj` már a betöltés alatt megérkezett, a betöltési
        # időt a main() levonja a keretből; elfogyott keretnél a szabályalapú réteg azonnal lép.
        if not args.halo_nelkul:
            self._load_net_safe()

    def _load_net_safe(self):
        try:
            self._load_net()
        except Exception as e:  # torch hiányzik / sérült modell → heurisztikus mód
            log("Háló betöltése sikertelen, heurisztikus mód:", repr(e))
            self.nnet, self.mcts = None, None

    def _load_net(self):
        import torch
        from MCTS import MCTS
        from nnet import DEFAULT_ARGS, FullerenNNet
        from utils import dotdict

        torch.set_num_threads(max(1, self.args.szalak))
        path = os.path.abspath(self.args.model)
        t0 = time.time()
        nnet = FullerenNNet(self.game, dict(DEFAULT_ARGS, cuda=False, checkpoint_dir=os.path.dirname(path)))
        # load_checkpoint print()-tel ír a stdout-ra: a stdout helyett közvetlenül a stderr-re irányított
        # print-et használunk (szálbiztosan, a sys.stdout cseréje nélkül)
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            nnet.load_checkpoint(os.path.dirname(path), os.path.basename(path))
        margs = dotdict(dict(numMCTSSims=self.args.max_sims, cpuct=self.args.cpuct,
                             mcts_gamma=self.args.gamma, mcts_win_check=True))
        mcts_cls = MCTS
        if self.args.mcts == "javitott":
            from mcts_javitott import MCTSJavitott as mcts_cls  # nyerés-ellenőrzés + diszkont
        self._uj_fa = lambda: mcts_cls(self.game, nnet, margs)
        self.mcts = self._uj_fa()
        self.nnet = nnet
        # a betöltött (nagy, tartós) objektumokat kivesszük a szemétgyűjtő figyeléséből → rövidebb GC-szünetek
        import gc
        gc.collect()
        gc.freeze()
        log(f"Modell betöltve: {path} ({time.time() - t0:.1f} s)")

    # ── segédfüggvények ──────────────────────────────────────────────────────
    def _flag(self, board, player):
        return fen72._flag_tile(board, player)

    def _opp_can_win(self, board, player, action):
        """Igaz, ha `action` után az ellenfélnek van egylépéses zászlóütése."""
        nb, npl = self.game.getNextState(board, player, action)
        if self.game.getGameEnded(nb, npl) != 0:
            return False
        my_flag = self._flag(nb, player)
        if my_flag < 0:
            return True
        v = self.game.getValidMoves(nb, npl)
        return bool(v[np.arange(NUM_TILES) * NUM_TILES + my_flag].any())

    # ── fő döntés ────────────────────────────────────────────────────────────
    def valassz(self, msg, kesés=0.0):
        # kesés: ennyi ideje várt már az üzenet, mire beolvastuk (pl. modellbetöltés alatt érkezett)
        t0 = time.time() - kesés
        gui = [list(m) for m in msg["lepesek"]]
        if not gui:
            raise ValueError("Üres lépéslista")
        board, player, move_no, _ = fen72.parse_fen(msg["fen"])
        exp = 1 if msg.get("szin", "vilagos") == "vilagos" else -1
        if exp != player:
            log(f"FIGYELEM: szin={msg.get('szin')} de a FEN szerint {'v' if player == 1 else 's'} lép")

        csak_gui, csak_motor = fen72.compare_moves(board, player, gui)
        if csak_gui or csak_motor:
            log(f"SZABÁLYELTÉRÉS (lépés {move_no}): csak GUI={csak_gui[:10]} csak motor={csak_motor[:10]}")

        acts = {}
        for m in gui:
            acts.setdefault(fen72.gui_to_action(board, player, m), m)

        # 1. egylépéses nyerés
        enemy_flag = self._flag(board, -player)
        for a, m in acts.items():
            if self.args.nyers:
                break
            if a % NUM_TILES == enemy_flag and a // NUM_TILES != self._flag(board, player):
                log(f"lépés {move_no}: zászlóütés {m}")
                return m

        limit = msg.get("idokorlat_ms", 5000) / 1000.0
        budget = limit * self.args.ido_arany
        deadline = t0 + max(0.05, budget - 0.15)
        self._hard = t0 + 0.8 * limit  # ezt semmilyen lépés (min_sims, szűrő) nem lépheti túl

        # 2. MCTS (vagy heurisztika). Ha a keret fele már elfogyott (pl. az első üzenet a modell
        # betöltése alatt várakozott), nem keresünk, hanem a szabályalapú réteg azonnal lép.
        keves_ido = time.time() > t0 + 0.5 * limit
        if keves_ido:
            log(f"lépés {move_no}: kevés idő maradt ({1000 * (time.time() - t0):.0f} ms eltelt) – szabályalapú lépés")
            # a biztonsági szűrő ilyenkor is kapjon egy rövid keretet, különben (lejárt _hard miatt)
            # azonnal kilépne, és egy véletlen, akár azonnal vesztő lépés menne ki
            self._hard = max(self._hard, time.time() + 0.25)
        if not keves_ido and self.nnet is not None:
            order, info = self._mcts_order(board, player, acts, deadline, move_no)
        else:
            order, info = list(acts), "heurisztika"
            self.rng.shuffle(order)

        self._t_kereses = time.time()
        # 3. biztonsági szűrő
        chosen = None
        if self.args.biztonsag and not self.args.nyers:
            for a in order[: self.args.szuro_mely]:
                if time.time() > self._hard:
                    break
                if not self._opp_can_win(board, player, a):
                    chosen = a
                    break
        if chosen is None:
            chosen = order[0]
        m = acts[chosen]
        vege = time.time()
        log(f"lépés {move_no}: {m} | {info} | {1000 * (vege - t0):.0f} ms")
        if vege > self._hard + 0.2:
            log(f"FIGYELEM: lassú lépés – keresés vége {1000 * (self._t_kereses - t0):.0f} ms, "
                f"szűrő vége {1000 * (vege - t0):.0f} ms, keret {1000 * (self._hard - t0):.0f} ms")
        return m

    def _mcts_order(self, board, player, acts, deadline, move_no=1):
        # minden lépésnél friss keresőfa: a régi referenciaszámlálással azonnal felszabadul, így nincs
        # nagy, sokáig élő objektumhalmaz, ami hosszú szemétgyűjtési szünetet okozhatna
        self.mcts = self._uj_fa()
        canon = self.game.getCanonicalForm(board, player)
        s = self.game.stringRepresentation(canon)
        n = 0
        while n < self.args.max_sims and (n < self.args.min_sims or time.time() < deadline) and time.time() < self._hard:
            self.mcts.search(canon)
            n += 1
        counts = {a: self.mcts.Nsa.get((s, a), 0) for a in acts}
        prior = self.mcts.Ps.get(s)
        key = (lambda a: (counts[a], prior[a] if prior is not None else 0.0))
        order = sorted(acts, key=key, reverse=True)
        # --veletlen-nyitas K: a saját első K lépésben a látogatásszámokkal arányos sorsolás
        # (két determinisztikus játékos különben mindig ugyanazt a partit játszaná)
        if (move_no + 1) // 2 <= self.args.veletlen_nyitas and sum(counts.values()) > 0:
            pick = self.rng.choices(list(counts), weights=list(counts.values()))[0]
            order.remove(pick)
            order.insert(0, pick)
        top = order[0]
        q = self.mcts.Qsa.get((s, top))
        return order, f"MCTS {n} szim, top N={counts[top]}" + (f" Q={q:+.2f}" if q is not None else "")


def main():
    ap = argparse.ArgumentParser(description="Fullerén-sakk gépi játékos (stdin/stdout JSON)")
    ap.add_argument("--model", default=_alap_modell())
    ap.add_argument("--max-sims", type=int, default=1000, help="MCTS szimulációk felső korlátja lépésenként")
    ap.add_argument("--min-sims", type=int, default=32, help="ennyit akkor is lefuttat, ha lejárt a keret")
    ap.add_argument("--ido-arany", type=float, default=0.6, help="az idokorlat_ms ekkora részét használja")
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--mcts", choices=["eredeti", "javitott"], default="javitott",
                    help="javitott = mcts_javitott.py (nyerés-ellenőrzés kibontáskor + diszkontálás)")
    ap.add_argument("--gamma", type=float, default=0.99, help="diszkont a javított MCTS-hez")
    ap.add_argument("--szalak", type=int, default=2, help="torch CPU-szálak")
    ap.add_argument("--szuro-mely", type=int, default=500, help="legfeljebb ennyi jelöltet vizsgál a biztonsági szűrő")
    ap.add_argument("--nincs-biztonsag", dest="biztonsag", action="store_false")
    ap.add_argument("--nyers", action="store_true",
                    help="csak a háló+MCTS (nincs nyerés-ellenőrzés és biztonsági szűrő) — összehasonlításhoz")
    ap.add_argument("--halo-nelkul", action="store_true", help="háló nélkül, csak heurisztika")
    ap.add_argument("--log", default=None, help="naplófájl (alapértelmezés: stderr)")
    ap.add_argument("--veletlen-nyitas", type=int, default=0, metavar="K",
                    help="a saját első K lépésben a látogatásszámokkal arányosan sorsol (méréshez)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.log:
        sys.stderr = open(args.log, "a", encoding="utf-8", buffering=1)

    # stdin/stdout UTF-8, soronként (Windows alatt is)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    jatekos = Jatekos(args)
    t_kesz = time.time()
    log(f"kész ({t_kesz - T_INDULAS:.1f} s alatt), várom az üzeneteket")
    elso = True
    # Az automatikus szemétgyűjtés egy partin át megőrzött nagy keresőfánál ritkán >1 s-os szünetet
    # okozhat, akár a lépés közepén is. Kikapcsoljuk, és a válasz elküldése után, az ellenfél
    # gondolkodási ideje alatt takarítunk.
    import gc
    gc.disable()
    for line in sys.stdin:
        t_olvas = time.time()
        # Ha az első üzenet már a betöltés alatt megérkezett (azonnal olvasható volt), az adapter
        # órája már a betöltés közben indult: ezt az időt levonjuk a gondolkodási keretből.
        kesés = (t_kesz - T_INDULAS) if elso and t_olvas - t_kesz < 0.05 else 0.0
        elso = False
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log("Nem JSON sor, kihagyom:", line[:200])
            continue
        tipus = msg.get("tipus")
        if tipus == "lepj":
            try:
                # az időt a sor beolvasásától mérjük (a JSON-feldolgozás is beleszámít)
                m = jatekos.valassz(msg, kesés + (time.time() - t_olvas))
            except Exception:
                log("HIBA a lépésválasztásban, véletlen lépés:\n" + traceback.format_exc())
                m = random.choice(msg.get("lepesek") or [[0, 0]])
            sys.stdout.write(json.dumps(m) + "\n")
            sys.stdout.flush()
            gc.collect()  # a válasz már elment – most nem késleltet
        elif tipus == "vege":
            log("vége:", json.dumps(msg, ensure_ascii=False))
            break
        else:
            log("Ismeretlen üzenettípus:", tipus)


if __name__ == "__main__":
    main()
