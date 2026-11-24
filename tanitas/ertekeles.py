"""
ertekeles.py — v2 (2026-10-12): aréna-játékos és független mérés rögzített ellenfél ellen.

Miért? Az eredeti arénában a két MCTS-fa mind a 60 partin át
megmaradt, zaj nélkül, így a partik nem függetlenek → az elfogadás zajos (1,8× túlszórás).
  * MCTSJatekos: partinként friss MCTS-fa (startGame), az első `temp_moves` féllépésben a
    látogatási eloszlásból sorsol (különböző partik), utána argmax.
  * moho_lepes: rögzített alapvonal — zászlót üt, ha tud; különben véletlen olyan lépés,
    amely után az ellenfélnek nincs egylépéses zászlóütése.
  * ertekel_moho_ellen: N parti (felcserélt színekkel) a mohó ellen → győzelmi arány.
  * v3: parbaj() színenként számol; ertekel_modell_ellen() egy rögzített referenciamodell ellen.
"""
import numpy as np

from Arena import Arena
from MCTS import MCTS


class MCTSJatekos:
    """Arena-kompatibilis játékos: friss fa minden partira, véletlenített nyitással."""

    def __init__(self, game, nnet, args, temp_moves=0, fresh_tree=True):
        self.game, self.nnet, self.args = game, nnet, args
        self.temp_moves = temp_moves
        self.fresh_tree = fresh_tree
        self.mcts = MCTS(game, nnet, args)
        self.ply = 0

    def startGame(self):
        if self.fresh_tree:
            self.mcts = MCTS(self.game, self.nnet, self.args)
        self.ply = 0

    def endGame(self):
        pass

    def __call__(self, canonical_board):
        self.ply += 1
        if self.ply <= self.temp_moves:
            pi = np.array(self.mcts.getActionProb(canonical_board, temp=1))
            return int(np.random.choice(len(pi), p=pi / pi.sum()))
        return int(np.argmax(self.mcts.getActionProb(canonical_board, temp=0)))


def moho_lepes(game, canonical_board, rng=np.random):
    """Mohó alapvonal kanonikus táblán (a soron lévő fél = +1)."""
    valids = game.getValidMoves(canonical_board, 1)
    wins = game.getWinningMoves(canonical_board, valids)
    if len(wins):
        return int(rng.choice(wins))
    legal = np.flatnonzero(valids)
    safe = []
    for a in legal:
        nb, npl = game.getNextState(canonical_board, 1, int(a))
        if game.getGameEnded(nb, npl) != 0:
            safe.append(int(a))
            continue
        if len(game.getWinningMoves(game.getCanonicalForm(nb, npl))) == 0:
            safe.append(int(a))
    return int(rng.choice(safe if safe else legal))


class MohoJatekos:
    def __init__(self, game, seed=None):
        self.game = game
        self.rng = np.random.RandomState(seed)

    def __call__(self, canonical_board):
        return moho_lepes(self.game, canonical_board, self.rng)


def parbaj(game, a, b, n_games):
    """n_games parti `a` és `b` között, fele-fele színnel (az első felében `a` a világos).

    Visszaad: dict(a_vilagos, a_sotet, b_vilagos, b_sotet, dontetlen) – győzelmek színenként.
    A játékosok Arena-kompatibilis függvények/objektumok (kanonikus tábla → akció).
    """
    half = n_games // 2
    r = dict(a_vilagos=0, a_sotet=0, b_vilagos=0, b_sotet=0, dontetlen=0)
    for _ in range(half):  # a = világos (player1)
        res = Arena(a, b, game).playGame()
        if res > 0.5:
            r["a_vilagos"] += 1
        elif res < -0.5:
            r["b_sotet"] += 1
        else:
            r["dontetlen"] += 1
    for _ in range(half):  # b = világos
        res = Arena(b, a, game).playGame()
        if res > 0.5:
            r["b_vilagos"] += 1
        elif res < -0.5:
            r["a_sotet"] += 1
        else:
            r["dontetlen"] += 1
    return r


def _eval_args(args, sims):
    from utils import dotdict
    eargs = dotdict(dict(args))
    eargs["numMCTSSims"] = sims
    return eargs


def ertekel_moho_ellen(game, nnet, args, n_games=20, sims=50, temp_moves=2):
    """A modell a mohó ellen; dict(a_vilagos, a_sotet, b_vilagos, b_sotet, dontetlen), a = modell."""
    ai = MCTSJatekos(game, nnet, _eval_args(args, sims), temp_moves=temp_moves, fresh_tree=True)
    return parbaj(game, ai, MohoJatekos(game), n_games)


def ertekel_modell_ellen(game, nnet, ref_nnet, args, n_games=20, sims=50, temp_moves=2):
    """A modell egy rögzített referenciamodell ellen (mindkettő azonos MCTS-beállítással)."""
    eargs = _eval_args(args, sims)
    ai = MCTSJatekos(game, nnet, eargs, temp_moves=temp_moves, fresh_tree=True)
    ref = MCTSJatekos(game, ref_nnet, eargs, temp_moves=temp_moves, fresh_tree=True)
    return parbaj(game, ai, ref, n_games)


# ── v4 (2026-10-19): párosított mérés – a kezdőelőny kiszűrése ───────────────────────────
# A v3 elemzése szerint egyforma erős modellek között a világos ~80%-ot nyer, ezért egy
# színre nem kiegyenlített mérés főleg a színt méri. Párosított mérésnél ugyanabból a
# (véletlen) nyitásból két parti megy, felcserélt színekkel; a párt az nyeri, aki mindkét
# partit megnyeri, az 1–1 (mindkét fél világossal nyer) nem dönt.

def sorsolt_nyitas(game, k, rng=np.random):
    """(board, player) k egyenletesen véletlen szabályos féllépés után (le nem zárult állás)."""
    for _ in range(100):
        board, player = game.getInitBoard(), 1
        ok = True
        for _ in range(k):
            a = int(rng.choice(np.flatnonzero(game.getValidMoves(board, player))))
            board, player = game.getNextState(board, player, a)
            if game.getGameEnded(board, player) != 0:
                ok = False
                break
        if ok:
            return board, player
    return game.getInitBoard(), 1


def jatek(game, feher, fekete, board=None, player=1):
    """Egy parti adott állásból. Visszaad: +1 világos nyert, -1 sötét nyert, 0 döntetlen."""
    if board is None:
        board, player = game.getInitBoard(), 1
    for p in (feher, fekete):
        if hasattr(p, "startGame"):
            p.startGame()
    while game.getGameEnded(board, player) == 0:
        canon = game.getCanonicalForm(board, player)
        a = (feher if player == 1 else fekete)(canon)
        if not game.getValidMoves(canon, 1)[a]:
            raise ValueError(f"Érvénytelen lépés: {a}")
        board, player = game.getNextState(board, player, a)
    r = game.getGameEnded(board, 1)  # világos szemszögéből
    return 1 if r > 0.5 else (-1 if r < -0.5 else 0)


def paros_parbaj(game, a, b, n_pairs, k_plies=2, rng=np.random):
    """n_pairs nyitáspár; minden nyitásból egy parti `a` világossal és egy `b` világossal.

    Visszaad: dict(a_par, b_par, dontetlen_par, a_vilagos, a_sotet, b_vilagos, b_sotet, dontetlen)
    – a *_par a megnyert párok száma (2–0), a többi a partik színenkénti eredménye.
    """
    r = dict(a_par=0, b_par=0, dontetlen_par=0, a_vilagos=0, a_sotet=0, b_vilagos=0, b_sotet=0, dontetlen=0)
    for _ in range(n_pairs):
        board, player = sorsolt_nyitas(game, k_plies, rng)
        g1 = jatek(game, a, b, board.copy(), player)   # a világos
        g2 = jatek(game, b, a, board.copy(), player)   # b világos
        a_pont = (g1 == 1) + (g2 == -1)
        b_pont = (g1 == -1) + (g2 == 1)
        r["a_vilagos"] += g1 == 1
        r["b_sotet"] += g1 == -1
        r["b_vilagos"] += g2 == 1
        r["a_sotet"] += g2 == -1
        r["dontetlen"] += (g1 == 0) + (g2 == 0)
        if a_pont > b_pont:
            r["a_par"] += 1
        elif b_pont > a_pont:
            r["b_par"] += 1
        else:
            r["dontetlen_par"] += 1
    return r
