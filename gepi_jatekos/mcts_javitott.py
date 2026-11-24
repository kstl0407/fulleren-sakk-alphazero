"""
mcts_javitott.py — az MCTS két javítása a vakfolt ellen (javaslat, az eredeti kód érintetlen).

1. Nyerés-ellenőrzés kibontáskor (`win_check`): ha egy új csúcsban a soron lévő fél egy lépéssel
   le tudja ütni az ellenfél zászlóját, a csúcs értéke azonnal +1, és a prior a nyerő lépés(ek)re
   koncentrálódik. Így a nyerő lépés kis háló-prior mellett sem marad látogatatlan, és a
   tanítási π-célérték is tartalmazza.
2. Diszkontálás (`gamma`): a visszaterjesztett érték lépésenként gamma-val szorzódik, így a
   "most nyerek" többet ér, mint a "később nyerek". Ez megszünteti a nyerő állásban való
   halogatást (zászlócsere-hurkok).

A felület megegyezik az alpha-zero-general MCTS-ével, a Coach-ban egy sor cseréjével használható:
    from mcts_javitott import MCTSJavitott as MCTS
args-ban opcionális: `mcts_gamma` (alap 0.99), `mcts_win_check` (alap True).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = HERE if os.path.exists(os.path.join(HERE, "game.py")) else os.path.join(os.path.dirname(HERE), "tanitas")
for p in (CODE, os.path.join(CODE, "alpha-zero-general-master")):
    if p not in sys.path:
        sys.path.insert(0, p)

from MCTS import EPS, MCTS  # noqa: E402
from game import CH_B_FLAG, NUM_TILES  # noqa: E402


class MCTSJavitott(MCTS):
    def __init__(self, game, nnet, args):
        super().__init__(game, nnet, args)
        # az args egy dotdict (dict), ezért .get – a getattr hiányzó kulcsnál KeyError-t dobna
        self.gamma = float(args.get("mcts_gamma", 0.99))
        self.win_check = bool(args.get("mcts_win_check", True))

    def _winning_actions(self, canonicalBoard, valids):
        """Kanonikus táblán (a soron lévő fél = +1) a zászlóütő lépések indexei."""
        flag = np.where(canonicalBoard[CH_B_FLAG] > 0)[0]
        if len(flag) == 0:
            return np.array([], dtype=int)
        idx = np.arange(NUM_TILES) * NUM_TILES + int(flag[0])
        return idx[valids[idx] > 0]

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            wins = self._winning_actions(canonicalBoard, valids) if self.win_check else []
            if len(wins):
                # ismert nyerés: a prior a nyerő lépésekre, az érték +1 (net-hívás nélkül)
                p = np.zeros(self.game.getActionSize())
                p[wins] = 1.0 / len(wins)
                self.Ps[s], v = p, 1.0
            else:
                p, v = self.nnet.predict(canonicalBoard)
                p = p * valids
                ssum = np.sum(p)
                p = p / ssum if ssum > 0 else (p + valids) / np.sum(p + valids)
                self.Ps[s] = p
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best, best_act = -float("inf"), -1
        sq = math.sqrt(self.Ns[s] + EPS)
        for a in np.nonzero(valids)[0]:
            a = int(a)
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * sq
            if u > cur_best:
                cur_best, best_act = u, a
        a = best_act
        if a == -1:
            self.Ns[s] += 1
            return 1.0

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.gamma * self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v
