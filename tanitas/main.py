import logging
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "alpha-zero-general-master"))

import coloredlogs
from Coach import Coach
from utils import dotdict

import torch

from game import FullerenSakkGame
from nnet import FullerenNNet
from logger import GameLogger

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

HERE = os.path.dirname(os.path.abspath(__file__))

# v2 (2026-10-12): a régi futások a checkpoints/ mappában maradnak, a v2 alapértelmezése checkpoints_v2/
args = dotdict({
    "numIters":                       50,
    "numEps":                        100,
    "tempThreshold":                  20,
    "maxlenOfQueue":             200_000,
    "numMCTSSims":                   150,
    "updateThreshold":              0.55,
    "arenaCompare":                   60,
    "cpuct":                         1.5,
    "dirichlet_alpha":               0.1,   # v2: 0.3 → 0.1 (≈ 10 / átlagos lépésszám ~95)
    "dirichlet_epsilon":            0.25,
    "checkpoint": os.path.join(HERE, "checkpoints_v2"),
    # (a régi "load_model" kulcsot semmi nem olvasta – a betöltést a --load / --load-model kapcsoló intézi)
    "load_folder_file": (
        os.path.join(HERE, "checkpoints_v2"),
        "best.pth.tar",
    ),
    "numItersForTrainExamplesHistory": 20,

    # ── v2 bővítések (a 3. futás elemzése alapján) ─────────────────────
    "mcts_win_check":     True,   # egylépéses nyerés felismerése kibontáskor
    "mcts_gamma":         0.99,   # diszkont a visszaterjesztésben
    "visit_targets":      True,   # π-célérték mindig a látogatási eloszlás
    "noise_all_game":     True,   # Dirichlet-zaj a teljes partin
    "repetitionDraw":        3,   # háromszori ismétlés → döntetlen (csak önjáték)
    "arenaFreshTree":     True,   # aréna: friss fa partinként
    "arenaTempMoves":        4,   # aréna: az első 4 féllépés sorsolva
    "saveRejected":       True,   # elutasított modellek mentése
    "evalGreedyGames":      20,   # iterációnként 20 parti a mohó alapvonal ellen
    "evalSims":             50,
    "evalEvery":             1,

    # ── v3 bővítések (2026-10-14, a v2 futás elemzése alapján) – a v2 profilban kikapcsolva ──
    "selfplayRandomPlies":   0,   # önjáték: az első N féllépés véletlen (nyitás-összeomlás ellen)
    "evalRefModel":       None,   # rögzített referenciamodell a méréshez (--ref-model)
    "evalRefGames":         20,   # partik a referencia ellen iterációnként
    "keepAllExamples":   False,   # False: csak a legutóbbi checkpoint_*.examples marad (lemezhely!)

    # ── v4 bővítések (2026-10-19, a v3 futás elemzése alapján) – a v2/v3 profilban kikapcsolva ──
    "mcts_threat_check":  False,  # fenyegetés esetén a nem kivédő lépések priorja csökken: False | "root" | "all"
    "mcts_threat_factor":  0.05,
    "arenaPaired":        False,  # párosított aréna (nyitáspárok felcserélt színnel)
    "arenaPairs":            40,  # = 80 parti
    "arenaOpeningPlies":      2,  # sorsolt nyitó féllépések az arénában és a referencia-mérésben
    "refGate":            False,  # elfogadás csak ha a referencia ellen nem rosszabb a legjobbnál
    "refGateTol":          0.05,
    "evalRefPairs":          20,  # = 40 parti a referencia ellen (párosítva)

    # ── v5 bővítések (2026-10-21, a v4 futás elemzése alapján) – a korábbi profilokban kikapcsolva ──
    "arenaSignAlpha":      None,  # aréna: egyoldali előjelpróba a döntött párokon (None = v4 szabály)
    "arenaExtendP":         0.4,  # határeset: alpha < p <= ennyi → további párok
    "arenaExtendPairs":      40,
    "refGateFresh":       False,  # kapu: a legjobbat is újramérjük ugyanazokon a nyitásokon
})

nnet_args = {
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "dropout":        0.3,
    "epochs":          3,   # v2 P2.1: 10 → 3 (túltanulás a 20 iterációs pufferen)
    "batch_size":     64,
    "hidden_dim":     512,
    "num_resblocks":  6,
    "cuda":           torch.cuda.is_available(),
    "checkpoint_dir": os.path.join(HERE, "checkpoints_v2"),
}

# Profilok: a dolgozat egyes változatai és az ablációs kísérlet beállításai
PROFILOK = {
    # A: az eredeti (1–3. futás) beállítás – kontroll
    "regi": dict(dirichlet_alpha=0.3, mcts_win_check=False, mcts_gamma=1.0, visit_targets=False,
                 noise_all_game=False, repetitionDraw=0, arenaFreshTree=False, arenaTempMoves=0,
                 _epochs=10),
    # B: csak az MCTS-javítások (nyerés-ellenőrzés + diszkont), minden más régi
    "mcts": dict(dirichlet_alpha=0.3, mcts_win_check=True, mcts_gamma=0.99, visit_targets=False,
                 noise_all_game=False, repetitionDraw=0, arenaFreshTree=False, arenaTempMoves=0,
                 _epochs=10),
    # C: B + célértékek, zaj, ismétlés
    "mcts_cel": dict(mcts_win_check=True, mcts_gamma=0.99, visit_targets=True, noise_all_game=True,
                     repetitionDraw=3, arenaFreshTree=False, arenaTempMoves=0, _epochs=10),
    # D: minden v2 javítás (alapértelmezés)
    "v2": dict(),
    # v3: v2 + véletlen nyitólépések az önjátékban (+ ajánlott: --ref-model); kimenet: checkpoints_v3/
    "v3": dict(selfplayRandomPlies=2),
    # v4: v3 + fenyegetés-ellenőrzés + párosított aréna + referencia-kapu (--ref-model kötelező);
    # a mohó elleni mérés kikapcsolva (telítődött); kimenet: checkpoints_v4/
    "v4": dict(selfplayRandomPlies=2, mcts_threat_check="root", arenaPaired=True, refGate=True,
               evalGreedyGames=0, evalRefGames=0),
    # v5: v4 + előjelpróbás aréna (hosszabbítással) + újramérő referencia-kapu + tempThreshold 8
    # (a v4-ben a 20 nagyobb volt a partik ~10 lépéses hosszánál → minden önjáték-lépés sorsolt);
    # kimenet: checkpoints_v5/
    "v5": dict(selfplayRandomPlies=2, mcts_threat_check="root", arenaPaired=True, refGate=True,
               evalGreedyGames=0, evalRefGames=0, tempThreshold=8,
               arenaSignAlpha=0.10, refGateFresh=True),
    # v6 (2026-11-16, a v5 futás elemzése alapján): v5 + nagyobb lépés iterációnként és szigorúbb kapu.
    #   numEps 100 → 150: több friss önjáték-adat iterációnként (a v5-ben az egymást követő jelöltek
    #   jellemzően nem különböztek az előző legjobbtól);
    #   kapu: 20 → 40 pár, tolerancia 0,05 → 0 (a v5 8. iterációjában a −0,05-ös határon átjutott
    #   egy gyengébb modell). Kimenet: checkpoints_v6/
    "v6": dict(selfplayRandomPlies=2, mcts_threat_check="root", arenaPaired=True, refGate=True,
               evalGreedyGames=0, evalRefGames=0, tempThreshold=8,
               arenaSignAlpha=0.10, refGateFresh=True,
               numEps=150, evalRefPairs=40, refGateTol=0.0),
    # v6h (2026-11-17): a v6 beállításai hideg starthoz és hosszú (500 iterációs) futáshoz, takarékos
    #   mentéssel: elutasított modellek nem mentődnek (különben ~26 MB/iteráció), és 25 iterációs
    #   blokkonként csak az utolsó elfogadott checkpoint marad meg (max. 20 fájl). Kimenet: checkpoints_v6h/
    "v6h": dict(selfplayRandomPlies=2, mcts_threat_check="root", arenaPaired=True, refGate=True,
                evalGreedyGames=0, evalRefGames=0, tempThreshold=8,
                arenaSignAlpha=0.10, refGateFresh=True,
                numEps=150, evalRefPairs=40, refGateTol=0.0,
                saveRejected=False, checkpointBlock=25),
}


def main():
    """Parse CLI args, set up the Coach, and run the AlphaZero training loop."""
    parser = argparse.ArgumentParser(description="Fullerén-sakk AlphaZero tanítás (v2/v3/v4/v5)")
    parser.add_argument("--load",        action="store_true",
                        help="A checkpoint-mappa best.pth.tar-jának és tanítási példáinak betöltése, folytatás")
    parser.add_argument("--load-model",  metavar="PATH", default=None,
                        help="Csak a háló súlyainak betöltése (példák nélkül), pl. checkpoints/best.pth.tar")
    parser.add_argument("--checkpoint-dir", metavar="DIR", default=None,
                        help="Kimeneti mappa (alap: checkpoints_v2; a v3/v4 profilnál checkpoints_v3/v4)")
    parser.add_argument("--profil", choices=sorted(PROFILOK), default="v2",
                        help="Beállításcsomag: regi (A), mcts (B), mcts_cel (C), v2 (D, alap), v3, v4")
    parser.add_argument("--iters",       type=int, default=None,
                        help="Iterációk száma (felülírja az alapértelmezett 50-t)")
    parser.add_argument("--sims",        type=int, default=None,
                        help="MCTS-szimulációk lépésenként (alap 150)")
    parser.add_argument("--eval-games",  type=int, default=None,
                        help="Iterációnkénti partik a mohó ellen (0 = kikapcs, alap 20)")
    parser.add_argument("--ref-model",   metavar="PATH", default=None,
                        help="Rögzített referenciamodell: iterációnként partik ellene (színenként a CSV-ben)")
    parser.add_argument("--ref-games",   type=int, default=None,
                        help="Partik a referenciamodell ellen iterációnként (alap 20)")
    parser.add_argument("--random-plies", type=int, default=None,
                        help="Önjáték: az első N féllépés véletlen (a v3 profilban 2)")
    parser.add_argument("--kezdo-iter", type=int, default=1, metavar="N",
                        help="Az első iteráció sorszáma (--load folytatásnál pl. 51), hogy a fájlnevek ne ütközzenek")
    parser.add_argument("--fenyegetes", choices=["ki", "root", "all"], default=None,
                        help="Fenyegetés-ellenőrzés az MCTS-ben (v4 alap: root)")
    parser.add_argument("--log-every",   type=int, default=0, metavar="N",
                        help="Minden N-edik epizódot naplóz (0 = kikapcs)")
    parser.add_argument("--log-console", action="store_true",
                        help="A naplót a konzolra is kiírja (lassabb)")
    parser.add_argument("--gyors-teszt", action="store_true",
                        help="Pár perces próbafutás apró paraméterekkel (a kód ellenőrzésére)")
    cli = parser.parse_args()

    # NB: a dotdict-nek csak __getattr__-ja van, ezért mindig kulcsosan írunk (args["x"] = …),
    # különben az args.get(...) a régi értéket látná.
    prof = dict(PROFILOK[cli.profil])
    epochs = prof.pop("_epochs", None)
    args.update(prof)
    if epochs:
        nnet_args["epochs"] = epochs

    if not cli.checkpoint_dir and cli.profil in ("v3", "v4", "v5", "v6", "v6h"):
        cli.checkpoint_dir = os.path.join(HERE, f"checkpoints_{cli.profil}")
    if cli.checkpoint_dir:
        d = os.path.abspath(cli.checkpoint_dir)
        args["checkpoint"] = d
        args["load_folder_file"] = (d, "best.pth.tar")
        nnet_args["checkpoint_dir"] = d
    if cli.iters:
        args["numIters"] = cli.iters
    if cli.sims:
        args["numMCTSSims"] = cli.sims
    if cli.eval_games is not None:
        args["evalGreedyGames"] = cli.eval_games
    if cli.ref_model:
        if not os.path.isfile(cli.ref_model):
            parser.error(f"Nincs ilyen referenciamodell: {cli.ref_model}")
        args["evalRefModel"] = os.path.abspath(cli.ref_model)
    if cli.ref_games is not None:
        args["evalRefGames"] = cli.ref_games
    args["startIter"] = cli.kezdo_iter
    if cli.random_plies is not None:
        args["selfplayRandomPlies"] = cli.random_plies
    if cli.fenyegetes is not None:
        args["mcts_threat_check"] = False if cli.fenyegetes == "ki" else cli.fenyegetes
    if args.get("refGate") and not args.get("evalRefModel"):
        parser.error("A referencia-kapuhoz (v4/v5 profil) kell egy --ref-model, pl. checkpoints_v2/best.pth.tar")
    if cli.gyors_teszt:
        args.update(numIters=cli.iters or 2, numEps=2, numMCTSSims=8, arenaCompare=4,
                    evalGreedyGames=min(2, args.get("evalGreedyGames", 0)), evalRefGames=min(2, args.get("evalRefGames", 0)),
                    evalSims=4, numItersForTrainExamplesHistory=2, arenaPairs=2, evalRefPairs=2,
                    arenaExtendPairs=min(2, args.get("arenaExtendPairs", 0)))
        nnet_args["epochs"] = 1

    os.makedirs(args.checkpoint, exist_ok=True)
    log.info("Fullerén-sakk × AlphaZero — inicializálás (profil: %s)", cli.profil)
    log.info("Beállítások: %s", {k: v for k, v in args.items() if k not in ("load_folder_file",)})
    log.info("Háló: %s", {k: v for k, v in nnet_args.items() if k != "checkpoint_dir"})
    g = FullerenSakkGame()
    log.info("Tábla: %s  |  Akciótér: %d", g.getBoardSize(), g.getActionSize())

    nnet = FullerenNNet(g, args=nnet_args)

    if cli.load:
        folder, fname = args.load_folder_file
        log.info("Checkpoint betöltése: %s/%s", folder, fname)
        nnet.load_checkpoint(folder, fname)
    elif cli.load_model:
        p = os.path.abspath(cli.load_model)
        log.info("Csak a háló súlyainak betöltése: %s (tanítási példák nélkül)", p)
        nnet.load_checkpoint(os.path.dirname(p), os.path.basename(p))
    else:
        log.warning("Hideg start — nincs checkpoint betöltve.")

    game_logger = None
    if cli.log_every and cli.log_every > 0:
        log_path = os.path.join(args.checkpoint, "jatszma_naplo.txt")
        game_logger = GameLogger(
            path=log_path,
            log_every=cli.log_every,
            console=cli.log_console,
        )
        log.info("Játszma-napló: %s  (minden %d. epizód)", log_path, cli.log_every)

    coach = Coach(g, nnet, args, game_logger=game_logger)

    if cli.load:
        log.info("Korábbi tanítási példák betöltése...")
        coach.loadTrainExamples()

    log.info(
        "Tanítás indul — numIters=%d, numEps=%d, numMCTSSims=%d, kimenet: %s",
        args.numIters, args.numEps, args.numMCTSSims, args.checkpoint,
    )

    try:
        coach.learn()
    finally:
        if game_logger:
            game_logger.close()


if __name__ == "__main__":
    main()
