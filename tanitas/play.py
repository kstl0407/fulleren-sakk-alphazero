import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "alpha-zero-general-master"))

from MCTS import MCTS
from utils import dotdict

from game import (
    FullerenSakkGame,
    NUM_TILES,
    CH_W_FLAG, CH_W_ROOK, CH_W_BISHOP,
    CH_B_FLAG, CH_B_ROOK, CH_B_BISHOP,
    CH_W_FLAGSTATE, CH_B_FLAGSTATE,
    _piece_ch, _opp_channels,
)
from nnet import FullerenNNet
from grid import PENTAGON_IDS

PIECE_NAMES = {
    CH_W_FLAG:   "Zászló",
    CH_W_ROOK:   "Bástya",
    CH_W_BISHOP: "Futó",
    CH_B_FLAG:   "Zászló",
    CH_B_ROOK:   "Bástya",
    CH_B_BISHOP: "Futó",
}

TEAM_NAMES = {1: "FEHÉR", -1: "FEKETE"}


def _to0(tile_1based: int) -> int:
    """Convert a 1-based tile number to 0-based."""
    return tile_1based - 1


def _to1(tile_0based: int) -> int:
    """Convert a 0-based tile index to 1-based for display."""
    return tile_0based + 1


def print_neighbors(grid, tile_0: int):
    """Print edge and vertex neighbors of a tile in 1-based indexing."""
    shape = "Ötszög" if tile_0 in PENTAGON_IDS else "Hatszög"
    edge  = [_to1(n) for n in grid.edge_neighbors[tile_0] if n != -1]
    vert  = [_to1(n) for n in grid.vertex_neighbors[tile_0] if n != -1]
    print(f"Mező {_to1(tile_0)} ({shape})")
    print(f"  Lapszomszédok:   {' '.join(map(str, edge))}")
    print(f"  Csúcsszomszédok: {' '.join(map(str, vert))}")
    print()


def print_board(board: np.ndarray):
    """Print a compact summary of both players' piece positions."""
    def tiles(ch):
        return sorted(_to1(int(i)) for i in np.where(board[ch] > 0)[0])

    w_flag_state = int(board[CH_W_FLAGSTATE, 0])
    b_flag_state = int(board[CH_B_FLAGSTATE, 0])

    print("  Fehér : Zászló" + str(tiles(CH_W_FLAG)) +
          f"[{'F' if w_flag_state == 0 else 'B'}-mód]"
          f"  Futó{tiles(CH_W_BISHOP)}  Bástya{tiles(CH_W_ROOK)}")
    print("  Fekete: Zászló" + str(tiles(CH_B_FLAG)) +
          f"[{'F' if b_flag_state == 0 else 'B'}-mód]"
          f"  Futó{tiles(CH_B_BISHOP)}  Bástya{tiles(CH_B_ROOK)}")


def get_valid_dsts(game: FullerenSakkGame, board: np.ndarray,
                   player: int, src_0: int) -> list:
    """Return 0-based destination tiles reachable from src_0 for player."""
    valids = game.getValidMoves(board, player)
    return [dst for dst in range(NUM_TILES) if valids[src_0 * NUM_TILES + dst]]


def read_tile(prompt: str) -> int:
    """Read a 1-based tile number from stdin; return 0 to quit, -1 on bad input."""
    try:
        val = int(input(prompt))
    except (ValueError, EOFError):
        return -1
    return val


def load_ai(checkpoint_path: str, num_sims: int, game: FullerenSakkGame) -> MCTS:
    """Load a trained checkpoint and return a ready MCTS instance."""
    folder, fname = os.path.split(checkpoint_path)
    print(f"  AI betöltése: {fname}  ({num_sims} szimuláció/lépés)")
    nnet = FullerenNNet(game)
    nnet.load_checkpoint(folder, fname)
    mcts_args = dotdict({"numMCTSSims": num_sims, "cpuct": 1.5})
    return MCTS(game, nnet, mcts_args)


def ai_turn(game: FullerenSakkGame, board: np.ndarray,
            player: int, mcts: MCTS) -> int:
    """Let the AI pick and announce its move; return the action index."""
    print(f"\n  AI ({TEAM_NAMES[player]}) gondolkodik...")
    canonical = game.getCanonicalForm(board, player)
    pi        = mcts.getActionProb(canonical, temp=0)
    action    = int(np.argmax(pi))

    src       = action // NUM_TILES
    dst       = action % NUM_TILES
    src_ch    = _piece_ch(board, src, player)
    piece_name = PIECE_NAMES.get(src_ch, "?")

    opp_flag_ch = CH_B_FLAG if player == 1 else CH_W_FLAG
    opp_ch_list = list(_opp_channels(player))

    if board[opp_flag_ch, dst]:
        print(f"  AI lép: {piece_name} {_to1(src)} → {_to1(dst)}  !!! ÜTÉS (ellenfél zászlója!) !!!")
    elif any(board[ch, dst] for ch in opp_ch_list):
        victim_ch   = _piece_ch(board, dst, -player)
        victim_name = PIECE_NAMES.get(victim_ch, "?")
        print(f"  AI lép: {piece_name} {_to1(src)} → {_to1(dst)}  (ütés: {victim_name})")
    elif src_ch in (CH_W_FLAG, CH_B_FLAG):
        flagstate_ch = CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE
        flag_state   = int(board[flagstate_ch, 0])
        dst_ch       = _piece_ch(board, dst, player)
        if flag_state == 0 and dst_ch in (CH_W_BISHOP, CH_B_BISHOP):
            print(f"  AI lép: Zászlócsere {_to1(src)} → {_to1(dst)}  (Futó→Bástya, mód 0→1)")
        elif flag_state == 1 and dst_ch in (CH_W_ROOK, CH_B_ROOK):
            print(f"  AI lép: Visszacsere {_to1(src)} → {_to1(dst)}  (Bástya→Futó, mód 1→0)")
        else:
            print(f"  AI lép: Zászló {_to1(src)} → {_to1(dst)}")
    else:
        print(f"  AI lép: {piece_name} {_to1(src)} → {_to1(dst)}")

    return action


def main():
    """Run the interactive game loop (human vs. human or human vs. AI)."""
    parser = argparse.ArgumentParser(description="Interaktív Fullerén-sakk")
    parser.add_argument(
        "--ai-color", choices=["white", "black"], default=None,
        help="AI által játszott szín. Ha nincs megadva: ember vs. ember."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best.pth.tar"),
        help="Checkpoint fájl elérési útja (alapértelmezett: checkpoints/best.pth.tar)"
    )
    parser.add_argument(
        "--sims", type=int, default=50,
        help="MCTS szimulációk száma lépésenként (alapértelmezett: 50)"
    )
    cli = parser.parse_args()

    game   = FullerenSakkGame()
    board  = game.getInitBoard()
    player = 1  # white moves first

    ai_side = None
    mcts    = None
    if cli.ai_color is not None:
        ai_side = 1 if cli.ai_color == "white" else -1
        mcts    = load_ai(cli.checkpoint, cli.sims, game)

    print()
    print("=" * 60)
    print("  Interaktív Fullerén-sakk")
    if ai_side is not None:
        print(f"  AI: {TEAM_NAMES[ai_side]}   —   Te: {TEAM_NAMES[-ai_side]}")
    else:
        print("  Mód: Ember vs. Ember")
    print("  (0 megadása bármelyik promptnál kilép)")
    print("=" * 60)

    while True:
        print()
        print_board(board)
        print()
        print(f" SORON KÖVETKEZIK: {TEAM_NAMES[player]}")
        print()

        if mcts is not None and player == ai_side:
            action = ai_turn(game, board, player, mcts)
            board, player = game.getNextState(board, player, action)
            ended = game.getGameEnded(board, -player)
            if ended != 0.0:
                print()
                print_board(board)
                prev = -player
                if ended > 0:
                    print(f"=== JÁTÉK VÉGE: {TEAM_NAMES[prev]} nyert! ===")
                else:
                    print("=== DÖNTETLEN (körkorlát elérve) ===")
                break
            continue

        raw_src = read_tile("Melyik mezőről akarsz lépni? (Kilépés: 0): ")
        if raw_src == 0:
            break
        if raw_src < 1 or raw_src > NUM_TILES:
            print(">>> HIBA: Érvénytelen mezőszám.")
            continue

        src    = _to0(raw_src)
        src_ch = _piece_ch(board, src, player)
        opp_ch = _piece_ch(board, src, -player)

        if src_ch is None:
            if opp_ch is not None:
                print(f">>> HIBA: Ez nem a te bábudj! Válassz {TEAM_NAMES[player]} bábot.")
            else:
                print(f">>> HIBA: A(z) {raw_src}. mezőn nincs bábu!")
            continue

        print_neighbors(game.grid, src)

        dsts = get_valid_dsts(game, board, player, src)
        if not dsts:
            print(f">>> A(z) {raw_src} mezőn lévő {PIECE_NAMES[src_ch]} nem tud lépni.")
            continue
        print(f"Érvényes célmezők a(z) {raw_src} mezőről ({PIECE_NAMES[src_ch]}): "
              f"{sorted(_to1(d) for d in dsts)}")
        print()

        raw_dst = read_tile(f"Hova szeretnél lépni a(z) {raw_src}. mezőről?: ")
        if raw_dst == 0:
            break
        if raw_dst < 1 or raw_dst > NUM_TILES:
            print(">>> HIBA: Érvénytelen mezőszám.")
            continue

        dst    = _to0(raw_dst)
        action = src * NUM_TILES + dst

        valids = game.getValidMoves(board, player)
        if not valids[action]:
            print(">>> Szabálytalan lépés!")
            continue

        opp_flag_ch  = CH_B_FLAG if player == 1 else CH_W_FLAG
        opp_ch_list  = list(_opp_channels(player))
        flagstate_ch = CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE
        flag_ch      = CH_W_FLAG   if player == 1 else CH_B_FLAG
        bishop_ch    = CH_W_BISHOP if player == 1 else CH_B_BISHOP
        rook_ch      = CH_W_ROOK   if player == 1 else CH_B_ROOK

        if board[opp_flag_ch, dst]:
            board, _ = game.getNextState(board, player, action)
            print(f"!!! ÜTÉS !!! → {TEAM_NAMES[player]} ütötte az ellenfél zászlóját!")
            print(f"=== GYŐZELEM! {TEAM_NAMES[player]} nyert! ===")
            break
        elif any(board[ch, dst] for ch in opp_ch_list):
            victim_ch = _piece_ch(board, dst, -player)
            print(f"!!! ÜTÉS !!! {PIECE_NAMES[src_ch]}({raw_src}) üti"
                  f" {PIECE_NAMES[victim_ch]}({raw_dst})")
        elif src_ch == flag_ch:
            flag_state = int(board[flagstate_ch, 0])
            dst_ch     = _piece_ch(board, dst, player)
            if flag_state == 0 and dst_ch == bishop_ch:
                print(f">>> Zászlócsere! Futó({raw_dst}) → Bástya, mód 0→1")
            elif flag_state == 1 and dst_ch == rook_ch:
                print(f">>> Visszacsere! Bástya({raw_dst}) → Futó, mód 1→0")

        board, player = game.getNextState(board, player, action)

        ended = game.getGameEnded(board, -player)
        if ended != 0.0:
            print()
            print_board(board)
            prev = -player
            if ended > 0:
                print(f"=== JÁTÉK VÉGE: {TEAM_NAMES[prev]} nyert! ===")
            elif ended < 0:
                print(f"=== JÁTÉK VÉGE: {TEAM_NAMES[prev]} veszített (holtpont)! ===")
            else:
                print("=== DÖNTETLEN (körkorlát elérve) ===")
            break

    print("\nKöszönjük a játékot!")


if __name__ == "__main__":
    main()
