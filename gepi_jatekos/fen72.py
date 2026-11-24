"""
fen72.py — átjáró a GUI (sakk.itk.ppke.hu) FEN72 formátuma és a game.py belső táblája között.

GUI-konvenciók:
  * mezők 1-alapúak (1..72), a belső motor 0-alapú (0..71);
  * `Z/B/F` = zászló/bástya/futó, nagybetű = világos (= belső player +1), kisbetű = sötét (-1);
  * a zászló után egy számjegy: állapot (0/1) = belső CH_*_FLAGSTATE;
  * `v`/`s` = ki lép; órák; lépésszám (1-től számolt féllépés).

Sánc-eltérés: a GUI a sáncot mindkét irányból felsorolja ([1,2] és [2,1]), a belső
motor 0-s zászlóállapotnál csak a zászló→futó irányt generálja. A két lépés hatása
azonos, ezért a `gui_to_action` a futó→zászló irányt a zászló→futó akcióra képezi le.

Önálló teszt:  python fen72.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Önálló csomagban (csomagol.py) a game.py ugyanebben a mappában van;
# a tárolóban a szomszédos tanitas/ mappát használjuk.
CODE = HERE if os.path.exists(os.path.join(HERE, "game.py")) else os.path.join(os.path.dirname(HERE), "tanitas")
if CODE not in sys.path:
    sys.path.insert(0, CODE)
    sys.path.insert(0, os.path.join(CODE, "alpha-zero-general-master"))

from game import (  # noqa: E402
    CH_B_BISHOP, CH_B_FLAG, CH_B_FLAGSTATE, CH_B_ROOK, CH_PENTA, CH_THREAT_OPP, CH_TURN,
    CH_W_BISHOP, CH_W_FLAG, CH_W_FLAGSTATE, CH_W_ROOK, MAX_MOVES, NUM_CHANNELS, NUM_TILES,
    FullerenSakkGame, _PENTA_MASK, _threat_channel,
)

START_FEN = ("1Z0/2F/3F/4F/5F/6F/8B/14B/20B/26B/32B/38b/44b/50b/56b/62b/"
             "67z0/68f/69f/70f/71f/72f v 600.0/600.0 1")

_PIECE_CH = {
    "Z": CH_W_FLAG, "B": CH_W_ROOK, "F": CH_W_BISHOP,
    "z": CH_B_FLAG, "b": CH_B_ROOK, "f": CH_B_BISHOP,
}
_CH_PIECE = {v: k for k, v in _PIECE_CH.items()}
_ENTRY = re.compile(r"^(\d+)([ZBFzbf])(\d?)$")

# A belső motor a lépésszámból döntetlent hirdet MAX_MOVES-nál; a GUI-ban nincs
# döntetlen, ezért a hálónak adott számlálót ez alatt tartjuk.
_COUNT_CAP = MAX_MOVES - 50


def parse_fen(fen: str):
    """FEN72 → (board, player, move_no, clocks). player: +1 világos, -1 sötét."""
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Hibás FEN: {fen!r}")
    tiles = parts[0]
    player = 1 if parts[1] == "v" else -1
    clocks = tuple(float(x) for x in parts[2].split("/")) if len(parts) > 2 else (None, None)
    move_no = int(parts[3]) if len(parts) > 3 else 1

    board = np.zeros((NUM_CHANNELS, NUM_TILES), dtype=np.float32)
    for e in tiles.split("/"):
        if not e:
            continue
        m = _ENTRY.match(e)
        if not m:
            raise ValueError(f"Hibás FEN-bejegyzés: {e!r}")
        tile, ch = int(m[1]) - 1, _PIECE_CH[m[2]]
        if not 0 <= tile < NUM_TILES:
            raise ValueError(f"Mezőszám tartományon kívül: {e!r}")
        board[ch, tile] = 1.0
        if m[2] in "Zz":
            state = int(m[3] or 0)
            board[CH_W_FLAGSTATE if m[2] == "Z" else CH_B_FLAGSTATE, :] = float(state)

    board[CH_PENTA, :] = _PENTA_MASK
    # getNextState konvenció: CH_TURN[:71] = 0, ha világos jön, 1, ha sötét.
    board[CH_TURN, :71] = 0.0 if player == 1 else 1.0
    board[CH_TURN, 71] = min(move_no - 1, _COUNT_CAP) / MAX_MOVES
    # A fenyegetés-csatorna a lépésre NEM jövő fél támadott mezőit tartalmazza.
    board[CH_THREAT_OPP, :] = _threat_channel(board, -player, _game().grid)
    return board, player, move_no, clocks


def to_fen(board: np.ndarray, player: int, move_no: int = 1, clocks=(600.0, 600.0)) -> str:
    """Belső tábla → FEN72 (teszteléshez / szimulált adapterhez)."""
    entries = []
    for t in range(NUM_TILES):
        for ch, sym in _CH_PIECE.items():
            if board[ch, t] > 0:
                s = f"{t + 1}{sym}"
                if sym == "Z":
                    s += str(int(board[CH_W_FLAGSTATE, 0]))
                elif sym == "z":
                    s += str(int(board[CH_B_FLAGSTATE, 0]))
                entries.append(s)
                break
    return f"{'/'.join(entries)} {'v' if player == 1 else 's'} {clocks[0]:.1f}/{clocks[1]:.1f} {move_no}"


_G = None


def _game() -> FullerenSakkGame:
    global _G
    if _G is None:
        _G = FullerenSakkGame()
    return _G


def _flag_tile(board, player):
    t = np.where(board[CH_W_FLAG if player == 1 else CH_B_FLAG] > 0)[0]
    return int(t[0]) if len(t) else -1


def gui_to_action(board: np.ndarray, player: int, move) -> int:
    """GUI [honnan, hova] (1-alapú) → belső akcióindex (src*72+dst)."""
    src, dst = int(move[0]) - 1, int(move[1]) - 1
    flag = _flag_tile(board, player)
    own_bishop = CH_W_BISHOP if player == 1 else CH_B_BISHOP
    state = int(board[CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE, 0])
    # futó → saját zászló (0-s állapot): a motorban ez zászló → futó
    if dst == flag and state == 0 and board[own_bishop, src] > 0:
        return flag * NUM_TILES + src
    return src * NUM_TILES + dst


def action_to_gui(board: np.ndarray, player: int, action: int, gui_moves) -> list:
    """Belső akció → a GUI listájának megfelelő eleme (vagy None, ha nincs megfelelője)."""
    gm = {tuple(m) for m in gui_moves}
    src, dst = divmod(int(action), NUM_TILES)
    cand = (src + 1, dst + 1)
    if cand in gm:
        return list(cand)
    if (dst + 1, src + 1) in gm and gui_to_action(board, player, (dst + 1, src + 1)) == action:
        return [dst + 1, src + 1]
    return None


def engine_gui_moves(board: np.ndarray, player: int):
    """A belső motor szabályos lépései GUI-formában, a GUI-hoz hasonlóan mindkét sáncirány."""
    g = _game()
    valid = g.getValidMoves(board, player)
    out = set()
    flag = _flag_tile(board, player)
    state = int(board[CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE, 0])
    for a in np.nonzero(valid)[0]:
        s, d = divmod(int(a), NUM_TILES)
        out.add((s + 1, d + 1))
        if s == flag and state == 0:  # zászló→futó sánc: a GUI a fordítottját is adja
            out.add((d + 1, s + 1))
    return sorted(out)


def compare_moves(board, player, gui_moves):
    """(csak_gui, csak_motor) halmazpár — szabályeltérés-figyeléshez."""
    gm = {tuple(m) for m in gui_moves}
    em = set(engine_gui_moves(board, player))
    return sorted(gm - em), sorted(em - gm)


if __name__ == "__main__":
    g = _game()
    b, p, n, c = parse_fen(START_FEN)
    init = g.getInitBoard()
    same = all(np.array_equal(b[ch], init[ch]) for ch in range(NUM_CHANNELS))
    print("Kezdő FEN == getInitBoard():", same)
    print("Visszaalakítva:", to_fen(b, p, n, c) == START_FEN)
    gm = engine_gui_moves(b, p)
    print("Világos lépései GUI-formában:", len(gm), "(a játékfelület leírása szerint 115)")
    print("[2,1] →", divmod(gui_to_action(b, p, [2, 1]), NUM_TILES), "(várt: (0, 1) = zászló→futó)")
    print("Sötét lépései a kezdőállásban:", len(engine_gui_moves(b, -1)))
