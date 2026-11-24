import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


def _arg(args, key, default):
    """args egy dotdict (dict): a getattr hiányzó kulcsnál KeyError-t dobna, ezért .get."""
    try:
        return args.get(key, default)
    except AttributeError:
        return getattr(args, key, default)


class MCTS():
    """
    This class handles the MCTS tree.

    v2 (2026-10-12) bővítések, mind kapcsolható az args-ban (a vakfolt-elemzés alapján):
      * mcts_win_check (bool, alap False): kibontáskor, ha a játék megadja a
        getWinningMoves(canonicalBoard) metódust és van egylépéses nyerés, a csúcs értéke +1,
        a prior a nyerő lépésekre koncentrálódik (nincs háló-hívás).
      * mcts_gamma (float, alap 1.0): diszkont a visszaterjesztésben (v ← gamma·v lépésenként),
        hogy a gyorsabb nyerés többet érjen (halogatás / zászlócsere-hurok ellen).
      * noise_all_game (bool, alap False): a Dirichlet-zaj temp=0 mellett is (a parti egészén).
    v4 (2026-10-19):
      * mcts_threat_check (False | "root" | "all", alap False) / mcts_threat_factor (alap 0.05): ha
        az ellenfél a következő lépésben leütné a zászlót, a fenyegetést nem kivédő lépések priorja
        threat_factor-szorosára csökken (game.getParryingMoves, egzakt ellenőrzés). "root": csak a
        gyökérben (lépésenként egy számítás); "all": minden új csúcsban (~2 ms/csúcs többlet).
    Az alapértékekkel a viselkedés megegyezik az eredetivel.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.gamma = float(_arg(args, "mcts_gamma", 1.0))
        self.win_check = bool(_arg(args, "mcts_win_check", False)) and hasattr(game, "getWinningMoves")
        self.noise_all_game = bool(_arg(args, "noise_all_game", False))
        # v4: fenyegetés-ellenőrzés – ha az ellenfél a következő lépésben ütné a zászlót, a nem
        # kivédő lépések priorja `mcts_threat_factor`-szorosára csökken (ha van kivédés)
        # mcts_threat_check: False | "root" (csak a gyökérben, olcsó) | "all" (minden új csúcsban, lassabb)
        tc = _arg(args, "mcts_threat_check", False)
        tc = "all" if tc is True else (tc or False)
        self.threat_check = tc if hasattr(game, "getParryingMoves") else False
        self.threat_factor = float(_arg(args, "mcts_threat_factor", 0.05))
        self._threat_done = set()  # gyökerek, amelyeken a súlyozás már megtörtént

    def getActionProb(self, canonicalBoard, temp=1, add_dirichlet=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        add_dirichlet: ha True, Dirichlet-zajt kever a gyökércsúcs policy priorjába
                       az első szimuláció után (önjáték közben; temp>0 esetén, vagy
                       noise_all_game=True mellett mindig).

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # Az első szimuláció inicializálja a gyökér Ps[s] priorját
        self.search(canonicalBoard)

        # v4: fenyegetés-súlyozás a gyökérben (a teljes fában a search() végzi, ha "all")
        if self.threat_check == "root":
            s = self.game.stringRepresentation(canonicalBoard)
            if s in self.Ps and s in self.Vs and s not in self._threat_done:
                self._threat_done.add(s)
                self._threat_reweight(canonicalBoard, s)

        # Dirichlet-zaj a gyökér priorjához (AlphaZero-papír, 2017).
        # Arena-értékelésnél add_dirichlet=False marad → zajmentes.
        if add_dirichlet and (temp > 0 or self.noise_all_game):
            s = self.game.stringRepresentation(canonicalBoard)
            if s in self.Ps and s in self.Vs:
                alpha = _arg(self.args, 'dirichlet_alpha', 0.3)
                eps = _arg(self.args, 'dirichlet_epsilon', 0.25)
                valids = self.Vs[s]
                valid_idx = np.where(valids)[0]
                n_valid = len(valid_idx)
                if n_valid > 0:
                    noise = np.zeros(self.game.getActionSize())
                    noise[valid_idx] = np.random.dirichlet([alpha] * n_valid)
                    self.Ps[s] = (1 - eps) * self.Ps[s] + eps * noise
                    s_sum = np.sum(self.Ps[s])
                    if s_sum > 0:
                        self.Ps[s] /= s_sum

        # A maradék szimulációk már a (zajjal terhelt) priorral futnak
        for i in range(self.args.numMCTSSims - 1):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if sum(counts) == 0:
            # Biztonsági háló: egyenletes eloszlás az érvényes lépések felett.
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if valids.sum() > 0:
                probs = valids.astype(float)
                return (probs / probs.sum()).tolist()
            return [1.0 / self.game.getActionSize()] * self.game.getActionSize()

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def _threat_reweight(self, canonicalBoard, s):
        """v4: ha az ellenfél a következő lépésben ütné a zászlót, a nem kivédő lépések priorja csökken."""
        parry = self.game.getParryingMoves(canonicalBoard, self.Vs[s])
        if parry is not None and len(parry):
            w = np.full(len(self.Ps[s]), self.threat_factor)
            w[parry] = 1.0
            self.Ps[s] = self.Ps[s] * w
            self.Ps[s] /= np.sum(self.Ps[s])

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            valids = self.game.getValidMoves(canonicalBoard, 1)
            wins = self.game.getWinningMoves(canonicalBoard, valids) if self.win_check else ()
            if len(wins):
                # v2: ismert egylépéses nyerés — érték +1, prior a nyerő lépés(ek)en, háló-hívás nélkül
                self.Ps[s] = np.zeros(self.game.getActionSize())
                self.Ps[s][wins] = 1.0 / len(wins)
                v = 1.0
            else:
                self.Ps[s], v = self.nnet.predict(canonicalBoard)
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s  # renormalize
                else:
                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])
                if self.threat_check == "all":
                    self.Vs[s] = valids
                    self._threat_reweight(canonicalBoard, s)

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        sqrt_ns = math.sqrt(self.Ns[s])
        sqrt_ns_eps = math.sqrt(self.Ns[s] + EPS)
        for a in np.flatnonzero(valids):
            a = int(a)
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * sqrt_ns / (
                        1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * sqrt_ns_eps  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        if a == -1:
            # Nincs érvényes lépés – biztonsági fallback: a jelenlegi játékos veszít.
            self.Ns[s] += 1
            return 1.0

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # v2: diszkont (gamma=1.0 → eredeti viselkedés)
        v = self.gamma * self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
