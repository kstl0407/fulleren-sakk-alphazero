import csv
import logging
import math
import os
import time
from collections import Counter, deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS

log = logging.getLogger(__name__)


def _arg(args, key, default):
    """args egy dotdict (dict): a getattr hiányzó kulcsnál KeyError-t dobna, ezért .get."""
    try:
        return args.get(key, default)
    except AttributeError:
        return getattr(args, key, default)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.

    v2 (2026-10-12) kapcsolható bővítések (alapértékkel az eredeti viselkedés):
      * visit_targets: a π-célérték mindig a látogatási eloszlás (temp=1), a tempThreshold
        után csak a *lépésválasztás* argmax — korábban a célérték is one-hot lett.
      * repetitionDraw (int, 0 = ki): ha ugyanaz az állás (bábuk + zászlóállapot + lépő fél)
        ennyiszer előfordul, az önjáték-parti döntetlen (zászlócsere-hurkok ellen).
      * arenaFreshTree / arenaTempMoves: aréna partinként friss fával, az első N féllépés
        sorsolva (független partik).
      * saveRejected: az elutasított modell is mentésre kerül (rejected_<i>.pth.tar).
      * evalGreedyGames / evalSims / evalEvery: a legjobb modell mérése egy rögzített mohó
        ellenfél ellen (ertekeles.py).
      * Iterációnkénti összesítő: <checkpoint>/iteraciok.csv.
    v3 (2026-10-14):
      * selfplayRandomPlies (int, 0 = ki): az önjáték első N féllépése egyenletesen véletlen
        szabályos lépés (a π-célérték ilyenkor is az MCTS-eloszlás) — a nyitás-összeomlás ellen.
      * Az aréna és a mérések eredménye színenként is a CSV-be kerül; az önjáték színaránya is.
      * evalRefModel / evalRefGames: a legjobb modell mérése egy rögzített referenciamodell ellen.
    v4 (2026-10-19):
      * arenaPaired / arenaPairs / arenaOpeningPlies: párosított aréna – minden sorsolt nyitásból
        két parti felcserélt színnel; az új modell akkor jut tovább, ha több párt nyer (2–0), mint
        az előző, és a döntött párok legalább updateThreshold részét. (A v3-ban a világos ~80%-ot
        nyert, így a nem párosított aréna főleg a színt mérte.)
      * refGate / refGateTol / evalRefPairs: az elfogadáshoz a jelöltnek a referenciamodell ellen
        (párosítva) legalább a jelenlegi legjobb pontszámát kell hoznia (−tolerancia). A v3-ban a
        82–89%-os modelleket (checkpoint_32/59/72) az aréna gyengébbekre cserélte.
    v5 (2026-10-21, a v4 futás elemzése alapján):
      * arenaSignAlpha (None = v4 szabály): az aréna egyoldali előjelpróbával dönt a döntött párokon
        (p <= alpha); határesetben (alpha < p <= arenaExtendP) még arenaExtendPairs pár, és az
        összesített eredményre újra próba. A v4 szabálya azonos erőnél 34%-ban átengedett, ez ~10%-ban.
      * refGateFresh: a referencia-kapunál a jelenlegi legjobbat is újramérjük, UGYANAZOKON a
        nyitásokon, mint a jelöltet (rögzített seed). A v4-ben a mérce az átengedett jelöltek
        szerencsés méréseire ugrott (győztes átka), és a jó jelöltek nagy részét elutasította.
    """

    def __init__(self, game, nnet, args, game_logger=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, self.nnet.args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.game_logger = game_logger  # GameLogger instance or None
        self._current_iter = 0          # aktuális iteráció sorszáma (learn() frissíti)
        self._current_ep   = 0          # epizód sorszáma az iteráción belül
        self.ep_stats = Counter()       # iterációnkénti önjáték-statisztika (döntetlen, ismétlés)
        self.ref_nnet = None
        ref = _arg(self.args, "evalRefModel", None)
        if ref:
            self.ref_nnet = self.nnet.__class__(self.game, self.nnet.args)
            self.ref_nnet.load_checkpoint(os.path.dirname(os.path.abspath(ref)), os.path.basename(ref))
            log.info("Referenciamodell a méréshez: %s", ref)

    @staticmethod
    def _position_key(board, player):
        """Ismétlés-kulcs: bábuk és zászlóállapot (csatornák 0–7) + lépő fél; a lépésszámláló nélkül."""
        return board[:8].tobytes() + (b"w" if player == 1 else b"b")

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0 (v2 visit_targets: only for move selection, not for the target).

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        visit_targets = bool(_arg(self.args, "visit_targets", False))
        rep_limit = int(_arg(self.args, "repetitionDraw", 0) or 0)
        rnd_plies = int(_arg(self.args, "selfplayRandomPlies", 0) or 0)
        seen = Counter({self._position_key(board, self.curPlayer): 1})

        log = self.game_logger
        if log:
            log.start_episode(self._current_iter, self._current_ep)

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            explore = episodeStep < self.args.tempThreshold

            if visit_targets:
                # célérték: mindig a látogatási eloszlás; a zajt az MCTS noise_all_game kapcsolója szabja meg
                pi = self.mcts.getActionProb(canonicalBoard, temp=1, add_dirichlet=True)
                if explore:
                    action = np.random.choice(len(pi), p=pi)
                else:
                    p = np.asarray(pi)
                    action = int(np.random.choice(np.flatnonzero(p == p.max())))
            else:
                temp = int(explore)
                pi = self.mcts.getActionProb(canonicalBoard, temp=temp, add_dirichlet=True)
                action = np.random.choice(len(pi), p=pi)

            # Valid moves mask — az MCTS search() már kiszámolta és Vs-ben cachelte
            s = self.game.stringRepresentation(canonicalBoard)
            valids = self.mcts.Vs.get(s, self.game.getValidMoves(canonicalBoard, 1))

            if episodeStep <= rnd_plies:
                # v3: véletlen nyitólépés (a célérték marad az MCTS-eloszlás)
                action = int(np.random.choice(np.flatnonzero(valids)))

            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, valids, None])

            if log:
                log.log_step(board, self.curPlayer, action, pi)

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r == 0 and rep_limit > 0:
                key = self._position_key(board, self.curPlayer)
                seen[key] += 1
                if seen[key] >= rep_limit:
                    r = 1e-4  # döntetlen (mint a MAX_MOVES-határnál)
                    self.ep_stats["ismetles"] += 1

            if r != 0:
                if abs(r) < 0.5:
                    self.ep_stats["dontetlen"] += 1
                elif (-self.curPlayer if r < 0 else self.curPlayer) == 1:  # r a lépésre jövő fél szemszögéből
                    self.ep_stats["vilagos_nyer"] += 1
                self.ep_stats["partik"] += 1
                self.ep_stats["lepesek"] += episodeStep
                if log:
                    log.finish_episode(r, board, self.curPlayer)
                # x: [board, curPlayer, pi, valids, None]
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)), x[3])
                        for x in trainExamples]

    def _arena_players(self, pmcts, nmcts):
        if _arg(self.args, "arenaFreshTree", False):
            from ertekeles import MCTSJatekos
            k = int(_arg(self.args, "arenaTempMoves", 0))
            return (MCTSJatekos(self.game, self.pnet, self.args, temp_moves=k),
                    MCTSJatekos(self.game, self.nnet, self.args, temp_moves=k))
        return (lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)))

    def _write_csv(self, row):
        path = os.path.join(self.args.checkpoint, "iteraciok.csv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header != list(row.keys()):  # más oszlopkészletű korábbi futás → külön fájl
                path = os.path.join(self.args.checkpoint, "iteraciok_uj.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def _ref_pontszam(self, nnet, seed=None):
        """v4: `nnet` pontszáma a referenciamodell ellen párosított partikban (0..1; döntetlen = ½).
        v5: `seed` megadásával a nyitások sorsolása rögzített, így két modell ugyanazokon a nyitásokon mérhető."""
        from ertekeles import MCTSJatekos, _eval_args, paros_parbaj
        eargs = _eval_args(self.args, int(_arg(self.args, "evalSims", 50)))
        pairs = int(_arg(self.args, "evalRefPairs", 20))
        rng = np.random.RandomState(seed) if seed is not None else np.random
        allapot = None
        if seed is not None:
            # az MCTS a holtversenyeket a globális generátorral dönti el: ezt is rögzítjük, majd visszaállítjuk
            allapot = np.random.get_state()
            np.random.seed(seed)
        try:
            r = paros_parbaj(self.game, MCTSJatekos(self.game, nnet, eargs), MCTSJatekos(self.game, self.ref_nnet, eargs),
                             pairs, int(_arg(self.args, "arenaOpeningPlies", 2)), rng=rng)
        finally:
            if allapot is not None:
                np.random.set_state(allapot)
        return (r["a_vilagos"] + r["a_sotet"] + 0.5 * r["dontetlen"]) / (2 * pairs), r

    @staticmethod
    def _elojel_p(b, n):
        """v5: egyoldali előjelpróba: P(X >= b), X ~ Bin(n, 1/2) (n = döntött párok)."""
        if n <= 0:
            return 1.0
        return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        # v3: a kiinduló modell legyen a best.pth.tar, ha még nincs – különben, ha az első
        # iterációk mind elutasítottak, a --load folytatás nem találna modellt
        if not os.path.isfile(os.path.join(self.args.checkpoint, 'best.pth.tar')):
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

        ref_gate = bool(_arg(self.args, "refGate", False)) and self.ref_nnet is not None
        self.best_ref = None
        ref_fresh = bool(_arg(self.args, "refGateFresh", False))
        if ref_gate and not ref_fresh:
            self.best_ref, r = self._ref_pontszam(self.nnet)
            log.info('REFERENCIA (kiinduló modell, párosítva): %.3f  [világosként %d, sötétként %d / %d pár]',
                     self.best_ref, r["a_vilagos"], r["a_sotet"], int(_arg(self.args, "evalRefPairs", 20)))

        start = int(_arg(self.args, "startIter", 1) or 1)  # v3: folytatásnál a számozás folytatódik
        last = start + self.args.numIters - 1
        for i in range(start, last + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            self._current_iter = i
            self.ep_stats = Counter()
            t0 = time.time()
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > start:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for ep_idx in tqdm(range(self.args.numEps), desc="Self Play"):
                    self._current_ep = ep_idx + 1
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
                st = self.ep_stats
                if st["partik"]:
                    log.info("ÖNJÁTÉK: %d parti, átl. hossz %.1f, döntetlen %d (ebből ismétlés %d)",
                             st["partik"], st["lepesek"] / st["partik"], st["dontetlen"], st["ismetles"])

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)
            t_selfplay = time.time() - t0

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            t1 = time.time()
            losses = self.nnet.train(trainExamples) or (float("nan"), float("nan"))
            t_train = time.time() - t1
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            t2 = time.time()
            paired = bool(_arg(self.args, "arenaPaired", False))
            if paired:
                from ertekeles import MCTSJatekos, paros_parbaj
                ar = paros_parbaj(self.game, MCTSJatekos(self.game, self.pnet, self.args),
                                  MCTSJatekos(self.game, self.nnet, self.args),
                                  int(_arg(self.args, "arenaPairs", 40)), int(_arg(self.args, "arenaOpeningPlies", 2)))
            else:
                prev_player, new_player = self._arena_players(pmcts, nmcts)
                from ertekeles import parbaj
                ar = parbaj(self.game, prev_player, new_player, self.args.arenaCompare)  # a = előző, b = új
            pwins, nwins, draws = ar["a_vilagos"] + ar["a_sotet"], ar["b_vilagos"] + ar["b_sotet"], ar["dontetlen"]
            t_arena = time.time() - t2

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            log.info('ARÉNA színenként: új világosként %d, sötétként %d | előző világosként %d, sötétként %d',
                     ar["b_vilagos"], ar["b_sotet"], ar["a_vilagos"], ar["a_sotet"])
            reason = ""
            arena_p, extra = "", 0
            if paired:
                dec = ar["b_par"] + ar["a_par"]
                log.info('ARÉNA párok: új nyert %d, előző nyert %d, döntetlen pár %d',
                         ar["b_par"], ar["a_par"], ar["dontetlen_par"])
                alpha = _arg(self.args, "arenaSignAlpha", None)
                if alpha is None:  # v4 szabály
                    accepted = ar["b_par"] > ar["a_par"] and ar["b_par"] / dec >= self.args.updateThreshold
                else:
                    # v5: egyoldali előjelpróba a döntött párokon; határesetben (p <= arenaExtendP) még
                    # arenaExtendPairs pár, majd az összesített eredményen újra a próba
                    arena_p = self._elojel_p(ar["b_par"], dec)
                    ext_p = float(_arg(self.args, "arenaExtendP", 0.4))
                    n_ext = int(_arg(self.args, "arenaExtendPairs", 40))
                    if ar["b_par"] > ar["a_par"] and alpha < arena_p <= ext_p and n_ext > 0:
                        log.info('ARÉNA határeset (p = %.3f): további %d pár', arena_p, n_ext)
                        ar2 = paros_parbaj(self.game, MCTSJatekos(self.game, self.pnet, self.args),
                                           MCTSJatekos(self.game, self.nnet, self.args),
                                           n_ext, int(_arg(self.args, "arenaOpeningPlies", 2)))
                        for k in ar:
                            ar[k] += ar2[k]
                        extra = n_ext
                        pwins, nwins, draws = ar["a_vilagos"] + ar["a_sotet"], ar["b_vilagos"] + ar["b_sotet"], ar["dontetlen"]
                        dec = ar["b_par"] + ar["a_par"]
                        arena_p = self._elojel_p(ar["b_par"], dec)
                        log.info('ARÉNA összesítve: új nyert %d, előző nyert %d, döntetlen pár %d',
                                 ar["b_par"], ar["a_par"], ar["dontetlen_par"])
                    accepted = ar["b_par"] > ar["a_par"] and arena_p <= alpha
                    log.info('ARÉNA előjelpróba: p = %.4f (küszöb %.2f) → %s', arena_p, alpha,
                             "átjut" if accepted else "nem jut át")
                t_arena = time.time() - t2
            else:
                accepted = not (pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold)
            if not accepted:
                reason = "arena"
            cand_ref = ""
            t_ref = time.time()
            if accepted and ref_gate:
                tol = float(_arg(self.args, "refGateTol", 0.05))
                if ref_fresh:
                    # v5: a jelenlegi legjobbat (pnet) UGYANAZOKON a nyitásokon újramérjük → nincs „győztes átka”
                    seed = 1000 + i
                    self.best_ref, rb = self._ref_pontszam(self.pnet, seed=seed)
                    cand_ref, r = self._ref_pontszam(self.nnet, seed=seed)
                else:
                    cand_ref, r = self._ref_pontszam(self.nnet)
                log.info('REFERENCIA (jelölt, párosítva): %.3f vs legjobb %.3f (tolerancia %.2f)  '
                         '[világosként %d, sötétként %d]', cand_ref, self.best_ref, tol, r["a_vilagos"], r["a_sotet"])
                if cand_ref < self.best_ref - tol:
                    accepted = False
                    reason = "referencia"
            t_ref = time.time() - t_ref
            if not accepted:
                log.info('REJECTING NEW MODEL')
                if _arg(self.args, "saveRejected", False):
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'rejected_{i}.pth.tar')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self._blokk_takaritas(i)
                if cand_ref != "" and not ref_fresh:
                    self.best_ref = cand_ref

            # független mérés: a jelenlegi legjobb modell a mohó alapvonal ellen
            ew = el = ed = mw_v = mw_s = ""
            rw = rl = rw_v = rw_s = ""
            t_eval = 0.0
            sims = int(_arg(self.args, "evalSims", 50))
            n_eval = int(_arg(self.args, "evalGreedyGames", 0) or 0)
            every = max(1, int(_arg(self.args, "evalEvery", 1) or 1))
            due = i % every == 0 or i == last
            t3 = time.time()
            if n_eval > 0 and due:
                from ertekeles import ertekel_moho_ellen
                m = ertekel_moho_ellen(self.game, self.nnet, self.args, n_games=n_eval, sims=sims)
                mw_v, mw_s = m["a_vilagos"], m["a_sotet"]
                ew, el, ed = mw_v + mw_s, m["b_vilagos"] + m["b_sotet"], m["dontetlen"]
                log.info('MOHÓ ELLEN (legjobb modell): %d győzelem / %d vereség / %d döntetlen '
                         '(világosként %d, sötétként %d)', ew, el, ed, mw_v, mw_s)
            n_ref = int(_arg(self.args, "evalRefGames", 0) or 0)
            if self.ref_nnet is not None and n_ref > 0 and due:
                from ertekeles import ertekel_modell_ellen
                m = ertekel_modell_ellen(self.game, self.nnet, self.ref_nnet, self.args, n_games=n_ref, sims=sims)
                rw_v, rw_s = m["a_vilagos"], m["a_sotet"]
                rw, rl = rw_v + rw_s, m["b_vilagos"] + m["b_sotet"]
                log.info('REFERENCIA ELLEN (legjobb modell): %d győzelem / %d vereség (világosként %d, sötétként %d)',
                         rw, rl, rw_v, rw_s)
            t_eval = time.time() - t3 + t_ref

            st = self.ep_stats
            self._write_csv(dict(
                iter=i, uj_gyozelem=nwins, elozo_gyozelem=pwins, arena_dontetlen=draws,
                uj_vilagoskent=ar["b_vilagos"], uj_sotetkent=ar["b_sotet"],
                elozo_vilagoskent=ar["a_vilagos"], elozo_sotetkent=ar["a_sotet"],
                arena_uj_par=ar.get("b_par", ""), arena_elozo_par=ar.get("a_par", ""),
                arena_dontetlen_par=ar.get("dontetlen_par", ""),
                arena_p=f"{arena_p:.4f}" if arena_p != "" else "", arena_extra_par=extra,
                elfogadva=int(accepted), elutasitas_oka=reason,
                ref_jelolt=f"{cand_ref:.3f}" if cand_ref != "" else "",
                ref_legjobb=f"{self.best_ref:.3f}" if self.best_ref is not None else "",
                moho_gyozelem=ew, moho_vereseg=el, moho_dontetlen=ed,
                moho_vilagoskent=mw_v, moho_sotetkent=mw_s,
                ref_gyozelem=rw, ref_vereseg=rl, ref_vilagoskent=rw_v, ref_sotetkent=rw_s,
                pi_loss=f"{losses[0]:.4f}", v_loss=f"{losses[1]:.4f}",
                onjatek_partik=st["partik"], onjatek_atl_hossz=f"{st['lepesek'] / st['partik']:.1f}" if st["partik"] else "",
                onjatek_dontetlen=st["dontetlen"], onjatek_ismetles=st["ismetles"],
                onjatek_vilagos_nyer=st["vilagos_nyer"],
                onjatek_mp=round(t_selfplay), tanitas_mp=round(t_train), arena_mp=round(t_arena), ertekeles_mp=round(t_eval),
            ))

    def _blokk_takaritas(self, i):
        """v6h: checkpointBlock = B > 0 esetén egy B iterációs blokkban (1–B, B+1–2B, …) csak a legutóbb
        elfogadott checkpoint_<i>.pth.tar marad meg; a blokk korábbi checkpointjai törlődnek.
        A best.pth.tar, a temp.pth.tar és a .examples fájlok érintetlenek."""
        import glob
        import re
        b = int(_arg(self.args, "checkpointBlock", 0) or 0)
        if b <= 0:
            return
        for p in glob.glob(os.path.join(self.args.checkpoint, "checkpoint_*.pth.tar")):
            m = re.fullmatch(r"checkpoint_(\d+)\.pth\.tar", os.path.basename(p))
            if m and int(m[1]) != i and (int(m[1]) - 1) // b == (i - 1) // b:
                try:
                    os.remove(p)
                    log.info("Blokk-takarítás: %s törölve (a blokk újabb modellje: checkpoint_%d)", os.path.basename(p), i)
                except OSError as e:
                    log.warning("Nem sikerült törölni: %s (%s)", p, e)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        # v3: a puffer minden fájlban teljes, ezért elég a legutóbbi (különben iterációnként több GB gyűlik)
        if not _arg(self.args, "keepAllExamples", False):
            import glob
            for old in glob.glob(os.path.join(folder, "checkpoint_*.pth.tar.examples")):
                if os.path.abspath(old) != os.path.abspath(filename):
                    try:
                        os.remove(old)
                    except OSError as e:
                        log.warning("Nem sikerült törölni: %s (%s)", old, e)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            # v2: a példák checkpoint_<i>.pth.tar.examples néven mentődnek → a legutóbbit használjuk
            import glob
            import re
            cands = glob.glob(os.path.join(self.args.load_folder_file[0], "checkpoint_*.pth.tar.examples"))
            num = lambda p: int(re.search(r"checkpoint_(\d+)\.pth", p).group(1))
            if cands:
                examplesFile = max(cands, key=num)
                log.warning(f'Nincs "{modelFile}.examples", a legutóbbi példafájlt használom: {examplesFile}')
        if not os.path.isfile(examplesFile):
            # v2: nem állunk meg input()-tal (tmux/tee alatt kényelmetlen) — példák nélkül folytatjuk
            log.warning(f'File "{examplesFile}" with trainExamples not found! Példák nélkül folytatom (önjátékkal kezd).')
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
