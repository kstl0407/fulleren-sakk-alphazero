import os
import time
import numpy as np

from game import (
    NUM_TILES,
    CH_W_FLAG, CH_W_ROOK, CH_W_BISHOP,
    CH_B_FLAG, CH_B_ROOK, CH_B_BISHOP,
    CH_W_FLAGSTATE, CH_B_FLAGSTATE,
)

_CH_NAMES = {
    CH_W_FLAG:   "W-Zászló",
    CH_W_ROOK:   "W-Bástya",
    CH_W_BISHOP: "W-Futó",
    CH_B_FLAG:   "B-Zászló",
    CH_B_ROOK:   "B-Bástya",
    CH_B_BISHOP: "B-Futó",
}


def _piece_name(board: np.ndarray, tile: int, player: int) -> str:
    """Return the display name of the player's piece on tile."""
    chs = ([CH_W_FLAG, CH_W_ROOK, CH_W_BISHOP]
           if player == 1 else [CH_B_FLAG, CH_B_ROOK, CH_B_BISHOP])
    for ch in chs:
        if board[ch, tile] > 0:
            return _CH_NAMES.get(ch, "?")
    return "?"


def _action_event(board: np.ndarray, player: int, src: int, dst: int) -> str:
    """Classify the action as flag-swap, recapture, capture, or plain move."""
    flag_ch      = CH_W_FLAG      if player == 1 else CH_B_FLAG
    rook_ch      = CH_W_ROOK      if player == 1 else CH_B_ROOK
    bishop_ch    = CH_W_BISHOP    if player == 1 else CH_B_BISHOP
    flagstate_ch = CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE
    opp_chs      = ([CH_B_FLAG, CH_B_ROOK, CH_B_BISHOP]
                    if player == 1 else [CH_W_FLAG, CH_W_ROOK, CH_W_BISHOP])

    src_ch = None
    for ch in (flag_ch, rook_ch, bishop_ch):
        if board[ch, src] > 0:
            src_ch = ch
            break
    if src_ch is None:
        return "ISMERETLEN"

    flag_state = int(board[flagstate_ch, 0])

    if src_ch == flag_ch:
        if flag_state == 0 and board[bishop_ch, dst] > 0:
            return "ZÁSZLÓCSERE futóval (FS:0→1)"
        if flag_state == 1 and board[rook_ch, dst] > 0:
            return "VISSZACSERE bástyával (FS:1→0)"

    if src_ch == rook_ch and board[flag_ch, dst] > 0 and flag_state == 1:
        return "VISSZACSERE (bástya→zászló) (FS:1→0)"

    for ch in opp_chs:
        if board[ch, dst] > 0:
            name = _CH_NAMES.get(ch, "?")
            win  = "  *** GYŐZELEM ***" if ch in (CH_B_FLAG, CH_W_FLAG) else ""
            return f"ÜTÉS: {name}{win}"

    return "lépés"


class GameLogger:
    """Writes per-move game logs to a text file during self-play."""

    def __init__(self, path: str, log_every: int = 1, console: bool = False):
        self.path      = path
        self.log_every = log_every
        self.console   = console

        self._iter   = 0
        self._ep     = 0
        self._step   = 0
        self._active = False

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def start_episode(self, iter_num: int, ep_num: int):
        """Mark the beginning of an episode; activate logging if ep_num % log_every == 0."""
        self._iter   = iter_num
        self._ep     = ep_num
        self._step   = 0
        self._active = (self.log_every > 0 and ep_num % self.log_every == 0)
        if self._active:
            self._write(f"\n{'='*65}")
            self._write(
                f"ITERÁCIÓ {iter_num:3d} | EPIZÓD {ep_num:3d}  "
                f"[{time.strftime('%H:%M:%S')}]"
            )
            self._write(f"{'='*65}")

    def log_step(self, board: np.ndarray, player: int, action: int, pi):
        """Log one move with piece info, event type, and top-3 MCTS probabilities."""
        if not self._active:
            return
        self._step += 1
        src = action // NUM_TILES
        dst = action %  NUM_TILES

        piece      = _piece_name(board, src, player)
        event      = _action_event(board, player, src, dst)
        who        = "FEHÉR" if player == 1 else "FEKETE"
        fs_ch      = CH_W_FLAGSTATE if player == 1 else CH_B_FLAGSTATE
        flag_state = int(board[fs_ch, 0])

        pi_arr   = np.array(pi)
        top3_idx = np.argsort(pi_arr)[::-1][:3]
        top3 = [
            f"{t//NUM_TILES+1}→{t%NUM_TILES+1}({pi_arr[t]:.2f})"
            for t in top3_idx if pi_arr[t] > 0
        ]
        top3_str = "  top: " + ", ".join(top3) if top3 else ""

        line = (
            f"  [{self._step:3d}] {who:6s} | {piece:12s} "
            f"{src+1:2d}→{dst+1:2d}  [{event}]{top3_str}"
        )
        self._write(line)

    def finish_episode(self, result: float, board: np.ndarray, player: int):
        """Write the game outcome and total move count."""
        if not self._active:
            return
        # result is from the perspective of the player who is about to move (not who just moved).
        if abs(result) > 0.5:
            winner  = "FEKETE" if player == 1 else "FEHÉR"
            outcome = f"{winner} GYŐZÖTT"
        else:
            outcome = "DÖNTETLEN"
        self._write(
            f"  {'─'*60}\n"
            f"  Eredmény: {outcome}  ({self._step} lépés)\n"
        )

    def _write(self, msg: str):
        """Write a line to file (and optionally to console)."""
        self._file.write(msg + "\n")
        self._file.flush()
        if self.console:
            print(msg)

    def close(self):
        """Flush and close the log file."""
        self._file.close()
