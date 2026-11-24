import sys
import os
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "alpha-zero-general-master"))
from Game import Game

from grid import FullereneGrid, get_grid, PENTAGON_IDS, NUM_TILES

NUM_CHANNELS = 11
ACTION_SIZE  = NUM_TILES * NUM_TILES  # 5184

# Antipodal map: tile i ↔ tile _ANTIPODAL[i] (180° rotational symmetry)
# Formula (0-based): t∈[0,5]→t+66 | t∈[6,35]→t+30 | t∈[36,65]→t-30 | t∈[66,71]→t-66
_ANTIPODAL = np.empty(NUM_TILES, dtype=np.int32)
for _t in range(NUM_TILES):
    if   _t <=  5: _ANTIPODAL[_t] = _t + 66
    elif _t <= 35: _ANTIPODAL[_t] = _t + 30
    elif _t <= 65: _ANTIPODAL[_t] = _t - 30
    else:          _ANTIPODAL[_t] = _t - 66

MAX_MOVES = 300  # draw threshold

CH_W_FLAG      = 0
CH_W_ROOK      = 1
CH_W_BISHOP    = 2
CH_B_FLAG      = 3
CH_B_ROOK      = 4
CH_B_BISHOP    = 5
CH_W_FLAGSTATE = 6
CH_B_FLAGSTATE = 7
CH_TURN        = 8
CH_PENTA       = 9
CH_THREAT_OPP  = 10

INIT = {
    "white_flag":    [0],
    "white_rooks":   [7, 13, 19, 25, 31],
    "white_bishops": [1, 2, 3, 4, 5],
    "black_flag":    [66],
    "black_rooks":   [37, 43, 49, 55, 61],
    "black_bishops": [67, 68, 69, 70, 71],
}

_PENTA_MASK = np.zeros(NUM_TILES, dtype=np.float32)
for _p in PENTAGON_IDS:
    _PENTA_MASK[_p] = 1.0


def _board_state(board: np.ndarray) -> dict:
    """Map each occupied tile to its team (1=white, -1=black) for move generators."""
    state = {}
    for tile in range(NUM_TILES):
        if board[CH_W_FLAG, tile] or board[CH_W_ROOK, tile] or board[CH_W_BISHOP, tile]:
            state[tile] = 1
        elif board[CH_B_FLAG, tile] or board[CH_B_ROOK, tile] or board[CH_B_BISHOP, tile]:
            state[tile] = -1
    return state


def _piece_ch(board: np.ndarray, tile: int, player: int):
    """Return the channel index of player's piece on tile, or None if absent."""
    if player == 1:
        for ch in (CH_W_FLAG, CH_W_ROOK, CH_W_BISHOP):
            if board[ch, tile]:
                return ch
    else:
        for ch in (CH_B_FLAG, CH_B_ROOK, CH_B_BISHOP):
            if board[ch, tile]:
                return ch
    return None


def _opp_channels(player: int) -> Tuple[int, int, int]:
    """Return (flag, rook, bishop) channel indices for the opponent of player."""
    if player == 1:
        return CH_B_FLAG, CH_B_ROOK, CH_B_BISHOP
    return CH_W_FLAG, CH_W_ROOK, CH_W_BISHOP


def _move_count(board: np.ndarray) -> int:
    """Decode the normalised move counter stored in CH_TURN tile 71."""
    return round(float(board[CH_TURN, 71]) * MAX_MOVES)


def _threat_channel(board: np.ndarray, opponent: int, grid) -> np.ndarray:
    """Build a binary mask of all tiles reachable in one move by opponent's rooks and bishops."""
    rook_ch   = CH_W_ROOK   if opponent == 1 else CH_B_ROOK
    bishop_ch = CH_W_BISHOP if opponent == 1 else CH_B_BISHOP
    state     = _board_state(board)
    threat    = np.zeros(NUM_TILES, dtype=np.float32)
    for src in np.where(board[rook_ch] > 0)[0]:
        for dst in grid.get_rook_moves(int(src), state):
            threat[dst] = 1.0
    for src in np.where(board[bishop_ch] > 0)[0]:
        for dst in grid.get_bishop_moves(int(src), state):
            threat[dst] = 1.0
    return threat


class FullerenSakkGame(Game):
    """AlphaZero Game interface for Fullerene Chess."""

    def __init__(self):
        super().__init__()
        self.grid: FullereneGrid = get_grid()

    def getBoardSize(self) -> Tuple[int, int]:
        """Return (NUM_CHANNELS, NUM_TILES) as required by the AlphaZero interface."""
        return (NUM_CHANNELS, NUM_TILES)

    def getActionSize(self) -> int:
        """Return the flat action space size (72 × 72)."""
        return ACTION_SIZE

    def getInitBoard(self) -> np.ndarray:
        """Return the starting board with pieces in their initial positions."""
        board = np.zeros((NUM_CHANNELS, NUM_TILES), dtype=np.float32)

        board[CH_W_FLAG,   INIT["white_flag"]]    = 1.0
        board[CH_W_ROOK,   INIT["white_rooks"]]   = 1.0
        board[CH_W_BISHOP, INIT["white_bishops"]] = 1.0
        board[CH_B_FLAG,   INIT["black_flag"]]    = 1.0
        board[CH_B_ROOK,   INIT["black_rooks"]]   = 1.0
        board[CH_B_BISHOP, INIT["black_bishops"]] = 1.0

        board[CH_W_FLAGSTATE, :] = 0.0
        board[CH_B_FLAGSTATE, :] = 0.0
        board[CH_PENTA, :]       = _PENTA_MASK
        board[CH_TURN, :]        = 0.0

        board[CH_THREAT_OPP, :] = _threat_channel(board, -1, self.grid)
        return board

    def getNextState(self, board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
        """Apply action and return (next_board, next_player)."""
        src = action // NUM_TILES
        dst = action %  NUM_TILES

        b = board.copy()
        b = self._apply_move(b, player, src, dst)

        count = _move_count(board) + 1
        b[CH_TURN, 71]  = float(count) / MAX_MOVES
        b[CH_TURN, :71] = 0.0 if (-player == 1) else 1.0

        # Threat mask reflects the player who just moved, since they are now the opponent.
        b[CH_THREAT_OPP, :] = _threat_channel(b, player, self.grid)
        return b, -player

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        """Return a binary array of all legal moves for player in the current position."""
        valids = np.zeros(ACTION_SIZE, dtype=np.int8)
        state  = _board_state(board)

        flag_ch      = CH_W_FLAG      if player == 1 else CH_B_FLAG
        rook_ch      = CH_W_ROOK      if player == 1 else CH_B_ROOK
        bishop_ch    = CH_W_BISHOP    if player == 1 else CH_B_BISHOP
        flagstate_ch = CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE

        flag_tiles   = np.where(board[flag_ch]   > 0)[0]
        rook_tiles   = np.where(board[rook_ch]   > 0)[0]
        bishop_tiles = np.where(board[bishop_ch] > 0)[0]

        flag_pos   = int(flag_tiles[0]) if len(flag_tiles) > 0 else -1
        flag_state = int(board[flagstate_ch, 0])

        for src in rook_tiles:
            src = int(src)
            for dst in self.grid.get_rook_moves(src, state):
                valids[src * NUM_TILES + dst] = 1

            # Rook can swap back to the flag when flagState=1 and they are edge-adjacent.
            if flag_state == 1 and flag_pos >= 0:
                if flag_pos in self.grid.edge_neighbors[src]:
                    valids[src * NUM_TILES + flag_pos] = 1

        for src in bishop_tiles:
            src = int(src)
            for dst in self.grid.get_bishop_moves(src, state):
                valids[src * NUM_TILES + dst] = 1

        # Flag swap: flag moves onto an adjacent friendly bishop (flagState=0)
        # or onto an adjacent friendly rook (flagState=1).
        if flag_pos >= 0:
            for nbr in self.grid.edge_neighbors[flag_pos]:
                if nbr == -1:
                    continue
                if state.get(nbr) != player:
                    continue
                nbr_ch = _piece_ch(board, nbr, player)
                if flag_state == 0 and nbr_ch == bishop_ch:
                    valids[flag_pos * NUM_TILES + nbr] = 1
                elif flag_state == 1 and nbr_ch == rook_ch:
                    valids[flag_pos * NUM_TILES + nbr] = 1

        return valids

    def getWinningMoves(self, board: np.ndarray, valids: np.ndarray = None) -> np.ndarray:
        """Action indices with which player 1 (canonical board) captures the enemy flag (win in 1).

        Used by MCTS when args.mcts_win_check is on (v2). `valids` may be passed to avoid
        recomputing getValidMoves.
        """
        flag = np.flatnonzero(board[CH_B_FLAG])
        if len(flag) == 0:
            return np.empty(0, dtype=np.int64)
        if valids is None:
            valids = self.getValidMoves(board, 1)
        idx = np.arange(NUM_TILES) * NUM_TILES + int(flag[0])
        return idx[valids[idx] > 0]

    def _flag_attackers(self, board: np.ndarray):
        """Kanonikus táblán: az ellenfél (−1) mely bábui ütik a soron lévő fél zászlóját.

        Visszaad: (zászló mező, [(támadó mező, elérhető mezők halmaza, futó-e), …]).
        """
        flag = np.flatnonzero(board[CH_W_FLAG])
        if len(flag) == 0:
            return -1, []
        t = int(flag[0])
        state = _board_state(board)
        out = []
        for src in np.flatnonzero(board[CH_B_ROOK]):
            reach = self.grid.get_rook_moves(int(src), state)
            if t in reach:
                out.append((int(src), set(reach), False))
        for src in np.flatnonzero(board[CH_B_BISHOP]):
            reach = self.grid.get_bishop_moves(int(src), state)
            if t in reach:
                out.append((int(src), set(reach), True))
        return t, out

    def getParryingMoves(self, board: np.ndarray, valids: np.ndarray = None):
        """v4: kanonikus táblán, ha az ellenfél most le tudná ütni a soron lévő fél zászlóját,
        a fenyegetést megszüntető lépések indexei (üres tömb, ha nincs ilyen); None, ha nincs fenyegetés.

        A zászló sosem mozdul, ezért kivédeni csak a támadó leütésével, egy bábu útba
        állításával, vagy futó ellen a sorompó lezárásával (az átló melletti mezőre lépve) lehet.
        Ezeket a jelölteket (célmező = a támadó mezője, az általa elérhető mező, futónál ezek
        élszomszédai is) egzaktul ellenőrizzük a lépés végrehajtásával.
        """
        # gyors előszűrés: a CH_THREAT_OPP csatorna (getNextState tölti ki) épp az ellenfél által
        # támadott mezőket jelöli; ha a saját zászló mezője nincs köztük, nincs fenyegetés
        flag = np.flatnonzero(board[CH_W_FLAG])
        if len(flag) and board[CH_THREAT_OPP, int(flag[0])] == 0:
            return None
        t, attackers = self._flag_attackers(board)
        if not attackers:
            return None
        if valids is None:
            valids = self.getValidMoves(board, 1)
        zone = set()
        for src, reach, futo in attackers:
            tiles = reach | {src}
            zone |= tiles
            if futo:  # sorompó: a futó útja melletti mezők
                for x in tiles:
                    zone.update(n for n in self.grid.edge_neighbors[x] if n != -1)
        parry = []
        for a in np.flatnonzero(valids):
            src, dst = divmod(int(a), NUM_TILES)
            if dst not in zone:
                continue
            nb = self._apply_move(board, 1, src, dst)
            if not self._flag_attackers(nb)[1]:
                parry.append(int(a))
        return np.array(parry, dtype=np.int64)

    def getGameEnded(self, board: np.ndarray, player: int) -> float:
        """Return 1 if player won, -1 if lost, ~0 for draw, 0 if game continues."""
        opp_flag_ch = CH_B_FLAG if player == 1 else CH_W_FLAG
        if board[opp_flag_ch].sum() == 0:
            return 1.0

        own_flag_ch = CH_W_FLAG if player == 1 else CH_B_FLAG
        if board[own_flag_ch].sum() == 0:
            return -1.0

        if self.getValidMoves(board, player).sum() == 0:
            return -1.0

        if _move_count(board) >= MAX_MOVES:
            return 1e-4

        return 0.0

    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        """Flip board to white's perspective so the network always sees the same orientation."""
        if player == 1:
            return board
        b = board.copy()
        b[CH_W_FLAG],      b[CH_B_FLAG]      = board[CH_B_FLAG].copy(),      board[CH_W_FLAG].copy()
        b[CH_W_ROOK],      b[CH_B_ROOK]      = board[CH_B_ROOK].copy(),      board[CH_W_ROOK].copy()
        b[CH_W_BISHOP],    b[CH_B_BISHOP]    = board[CH_B_BISHOP].copy(),    board[CH_W_BISHOP].copy()
        b[CH_W_FLAGSTATE], b[CH_B_FLAGSTATE] = board[CH_B_FLAGSTATE].copy(), board[CH_W_FLAGSTATE].copy()
        b[CH_TURN, :71] = 1.0 - board[CH_TURN, :71]
        return b

    def getSymmetries(self, board: np.ndarray, pi: np.ndarray):
        """Return the original and its 180° rotated (antipodal) augmentation."""
        idx = _ANTIPODAL
        b_rot = np.empty_like(board)
        # Swap colours and permute tile indices under the antipodal map.
        b_rot[CH_W_FLAG]      = board[CH_B_FLAG][idx]
        b_rot[CH_W_ROOK]      = board[CH_B_ROOK][idx]
        b_rot[CH_W_BISHOP]    = board[CH_B_BISHOP][idx]
        b_rot[CH_B_FLAG]      = board[CH_W_FLAG][idx]
        b_rot[CH_B_ROOK]      = board[CH_W_ROOK][idx]
        b_rot[CH_B_BISHOP]    = board[CH_W_BISHOP][idx]
        b_rot[CH_W_FLAGSTATE] = board[CH_B_FLAGSTATE]
        b_rot[CH_B_FLAGSTATE] = board[CH_W_FLAGSTATE]
        b_rot[CH_PENTA]       = board[CH_PENTA][idx]
        b_rot[CH_TURN, :71]   = 1.0 - board[CH_TURN, :71]
        b_rot[CH_TURN, 71]    = board[CH_TURN, 71]
        b_rot[CH_THREAT_OPP]  = _threat_channel(b_rot, -1, self.grid)

        # Permute policy vector: action (src→dst) becomes (σ(src)→σ(dst)).
        pi_rot = np.zeros_like(pi)
        for a in np.nonzero(pi)[0]:
            src, dst = divmod(a, NUM_TILES)
            pi_rot[idx[src] * NUM_TILES + idx[dst]] = pi[a]

        return [(board, pi), (b_rot, pi_rot)]

    def stringRepresentation(self, board: np.ndarray) -> bytes:
        """Return a hashable bytes key for the board state (used by MCTS cache)."""
        return board.tobytes()

    @staticmethod
    def display(board: np.ndarray, move_num: int = None):
        """Print a compact text representation of the board to stdout."""
        sym = {
            CH_W_FLAG:   "F",
            CH_W_ROOK:   "R",
            CH_W_BISHOP: "B",
            CH_B_FLAG:   "f",
            CH_B_ROOK:   "r",
            CH_B_BISHOP: "b",
        }
        line = []
        for tile in range(NUM_TILES):
            ch_found = "."
            for ch, s in sym.items():
                if board[ch, tile] > 0:
                    ch_found = s
                    break
            line.append(ch_found)

        prefix = f"[{move_num:3d}] " if move_num is not None else "      "
        print(f"\n{prefix}Tábla állapota (tile 0–71):")
        for row in range(12):
            tiles = line[row * 6:(row + 1) * 6]
            ids   = [f"{row*6+i:2d}" for i in range(6)]
            print(f"  {'  '.join(ids)}")
            print(f"  {'  '.join(tiles)}")

    def _apply_move(self, board: np.ndarray, player: int, src: int, dst: int) -> np.ndarray:
        """Execute src→dst for player and return the updated board."""
        b = board.copy()

        flag_ch      = CH_W_FLAG      if player == 1 else CH_B_FLAG
        rook_ch      = CH_W_ROOK      if player == 1 else CH_B_ROOK
        bishop_ch    = CH_W_BISHOP    if player == 1 else CH_B_BISHOP
        flagstate_ch = CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE

        src_ch = _piece_ch(b, src, player)
        if src_ch is None:
            return b

        flag_state = int(b[flagstate_ch, 0])

        # 1. Flag → adjacent bishop (flagState 0→1): bishop becomes a rook.
        if src_ch == flag_ch and flag_state == 0 and b[bishop_ch, dst]:
            b[bishop_ch, dst]  = 0.0
            b[rook_ch,   dst]  = 1.0
            b[flagstate_ch, :] = 1.0
            return b

        # 2. Flag → adjacent rook (flagState 1→0): rook becomes a bishop.
        if src_ch == flag_ch and flag_state == 1 and b[rook_ch, dst]:
            b[rook_ch,   dst]  = 0.0
            b[bishop_ch, dst]  = 1.0
            b[flagstate_ch, :] = 0.0
            return b

        # 3. Rook → own flag (reverse swap, flagState 1→0): rook becomes a bishop.
        if src_ch == rook_ch and flag_state == 1 and b[flag_ch, dst]:
            b[rook_ch,   src]  = 0.0
            b[bishop_ch, src]  = 1.0
            b[flagstate_ch, :] = 0.0
            return b

        # 4. Normal move: remove any enemy piece on dst, then slide src piece there.
        opp_flag_ch, opp_rook_ch, opp_bishop_ch = _opp_channels(player)
        for ch in (opp_flag_ch, opp_rook_ch, opp_bishop_ch):
            b[ch, dst] = 0.0

        b[src_ch, src] = 0.0
        b[src_ch, dst] = 1.0
        return b


if __name__ == "__main__":
    game = FullerenSakkGame()
    board = game.getInitBoard()

    print(f"Tábla méret:  {game.getBoardSize()}")
    print(f"Akciótér:     {game.getActionSize()}")
    print(f"Fehér zászló: {np.where(board[CH_W_FLAG])[0].tolist()}")
    print(f"Fehér bástyák:{np.where(board[CH_W_ROOK])[0].tolist()}")
    print(f"Fehér futók:  {np.where(board[CH_W_BISHOP])[0].tolist()}")

    valids = game.getValidMoves(board, 1)
    print(f"\nFehér érvényes lépések: {valids.sum()}")

    flag_swaps = [(a // NUM_TILES, a % NUM_TILES) for a in range(ACTION_SIZE)
                  if valids[a] and a // NUM_TILES == 0]
    print(f"Zászlócsere lépések (src=0): {[(s, d) for s, d in flag_swaps]}")

    b2, p2 = game.getNextState(board, 1, 0 * NUM_TILES + 1)
    print(f"\nZaszlocsere utan (0->1):")
    print(f"  Fehér futók:  {np.where(b2[CH_W_BISHOP])[0].tolist()}")
    print(f"  Fehér bástyák:{np.where(b2[CH_W_ROOK])[0].tolist()}")
    print(f"  Fehér flagState: {int(b2[CH_W_FLAGSTATE, 0])}")
