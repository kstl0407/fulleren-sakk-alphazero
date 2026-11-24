import json
import os
from typing import List

NUM_TILES = 72

PENTAGON_IDS = [i for i in range(NUM_TILES) if i % 6 == 0 and i <= 66]

_P_LINKS_1BASED = {
    1:  [7,  13, 19, 25, 31],
    7:  [1,  31, 49, 43, 13],
    13: [1,  7,  43, 37, 19],
    19: [1,  13, 37, 61, 25],
    25: [1,  19, 61, 55, 31],
    31: [1,  25, 55, 49,  7],
    67: [37, 43, 49, 55, 61],
    37: [67, 61, 19, 13, 43],
    43: [67, 37, 13,  7, 49],
    49: [67, 43,  7, 31, 55],
    55: [67, 49, 31, 25, 61],
    61: [67, 55, 25, 19, 37],
}
P_LINKS = {(k - 1): [v - 1 for v in vals] for k, vals in _P_LINKS_1BASED.items()}


def _is_pentagon_1based(id_: int) -> bool:
    """Return True if the 1-based tile id is a pentagon."""
    return (id_ - 1) % 6 == 0 and id_ <= 67


class FullereneGrid:
    """Graph encoding the 72-tile fullerene surface (edge and vertex adjacency)."""

    def __init__(self):
        self.edge_neighbors:   List[List[int]] = [[] for _ in range(NUM_TILES)]
        self.vertex_neighbors: List[List[int]] = [[] for _ in range(NUM_TILES)]
        self._initialize()

    def _initialize(self):
        """Build edge and vertex neighbor lists from the pentagon link table."""
        for i in range(NUM_TILES):
            size = 5 if self.is_pentagon(i) else 6
            self.edge_neighbors[i]   = [-1] * size
            self.vertex_neighbors[i] = [-1] * size

        # Each pentagon anchors a ring of 5 hexagons; wire their 6 edge-slots.
        for p_1 in range(1, 68, 6):
            if not _is_pentagon_1based(p_1):
                continue
            p = p_1 - 1

            for i in range(5):
                hex_1   = p_1 + 1 + i
                hex_    = hex_1 - 1

                next_hex_1 = p_1 + 1 + (i + 1) % 5
                prev_hex_1 = p_1 + 1 + (i + 4) % 5
                next_hex   = next_hex_1 - 1
                prev_hex   = prev_hex_1 - 1

                target_p_1      = _P_LINKS_1BASED[p_1][i]
                next_target_p_1 = _P_LINKS_1BASED[p_1][(i + 1) % 5]

                self.edge_neighbors[hex_][0] = p
                self.edge_neighbors[hex_][1] = prev_hex

                # Find which slot of target_p_1 points back to our pentagon.
                j = -1
                for k in range(5):
                    if _P_LINKS_1BASED[target_p_1][k] == p_1:
                        j = k
                        break

                self.edge_neighbors[hex_][2] = target_p_1 + j
                self.edge_neighbors[hex_][3] = target_p_1 + (j + 4) % 5

                k_idx = -1
                for k in range(5):
                    if _P_LINKS_1BASED[next_target_p_1][k] == p_1:
                        k_idx = k
                        break

                self.edge_neighbors[hex_][4] = next_target_p_1 + k_idx
                self.edge_neighbors[hex_][5] = next_hex

                self.edge_neighbors[p][i] = hex_

        # A vertex neighbor of tile p at slot k is the unique tile shared by
        # edge-neighbors k and k+1 of p (the tile touching the corner between them).
        for p in range(NUM_TILES):
            p_size = len(self.edge_neighbors[p])
            for k in range(p_size):
                n1 = self.edge_neighbors[p][k]
                n2 = self.edge_neighbors[p][(k + 1) % p_size]
                if n1 == -1 or n2 == -1:
                    continue
                n1_size = len(self.edge_neighbors[n1])
                n2_size = len(self.edge_neighbors[n2])
                for ii in range(n1_size):
                    for jj in range(n2_size):
                        a = self.edge_neighbors[n1][ii]
                        b = self.edge_neighbors[n2][jj]
                        if a != -1 and a != p and a == b:
                            self.vertex_neighbors[p][k] = a

    @staticmethod
    def is_pentagon(tile_id: int) -> bool:
        """Return True if the 0-based tile id is a pentagon."""
        return tile_id % 6 == 0 and tile_id <= 66

    def get_opposite_edge_neighbor(self, current: int, arrival: int) -> int:
        """Return the tile directly opposite to the arrival direction (straight rook continuation)."""
        if self.is_pentagon(current):
            return -1
        nbrs = self.edge_neighbors[current]
        try:
            idx = nbrs.index(arrival)
        except ValueError:
            return -1
        return nbrs[(idx + 3) % 6]

    def get_opposite_vertex_neighbor(self, current: int, arrival: int) -> int:
        """Return the tile directly opposite across a vertex (straight bishop continuation)."""
        if self.is_pentagon(current):
            return -1
        nbrs = self.vertex_neighbors[current]
        try:
            idx = nbrs.index(arrival)
        except ValueError:
            return -1
        return nbrs[(idx + 3) % 6]

    def get_rook_moves(self, tile_id: int, board_state: dict) -> List[int]:
        """Return all tiles reachable by a rook from tile_id, sliding along edges."""
        moves = []
        current_team = board_state.get(tile_id)
        for start_dir in self.edge_neighbors[tile_id]:
            if start_dir == -1:
                continue
            prev_id, curr_id = tile_id, start_dir
            while curr_id != -1:
                if curr_id in board_state:
                    if board_state[curr_id] != current_team:
                        moves.append(curr_id)  # enemy piece is capturable
                    break
                moves.append(curr_id)
                if self.is_pentagon(curr_id):
                    break  # rooks stop on pentagons
                next_id = self.get_opposite_edge_neighbor(curr_id, prev_id)
                prev_id, curr_id = curr_id, next_id
        return moves

    def get_bishop_moves(self, tile_id: int, board_state: dict) -> List[int]:
        """Return all tiles reachable by a bishop from tile_id, sliding along vertices."""
        moves = []
        current_team = board_state.get(tile_id)
        opp = -current_team

        tile_edge_nbrs = self.edge_neighbors[tile_id]
        tile_edge_size  = len(tile_edge_nbrs)

        for i, start_dir in enumerate(self.vertex_neighbors[tile_id]):
            if start_dir == -1:
                continue

            # Two enemy pieces flanking the departure vertex block this direction.
            s1 = tile_edge_nbrs[i]
            s2 = tile_edge_nbrs[(i + 1) % tile_edge_size]
            if s1 != -1 and s2 != -1 and board_state.get(s1) == opp and board_state.get(s2) == opp:
                continue

            prev_id, curr_id = tile_id, start_dir

            while curr_id != -1:
                if curr_id in board_state:
                    if board_state[curr_id] != current_team:
                        moves.append(curr_id)
                    break

                moves.append(curr_id)
                if self.is_pentagon(curr_id):
                    break  # bishops stop on pentagons

                vtx_nbrs = self.vertex_neighbors[curr_id]
                try:
                    arrival_idx = vtx_nbrs.index(prev_id)
                except ValueError:
                    break

                next_dir = (arrival_idx + 3) % 6
                prev_id = curr_id

                # Two enemy pieces flanking the transit vertex block further travel.
                edge_nbrs = self.edge_neighbors[curr_id]
                s1 = edge_nbrs[next_dir]
                s2 = edge_nbrs[(next_dir + 1) % 6]
                if s1 != -1 and s2 != -1 and board_state.get(s1) == opp and board_state.get(s2) == opp:
                    break

                curr_id = vtx_nbrs[next_dir]

        return moves

    def to_dict(self) -> dict:
        """Serialize adjacency lists to a plain dict for JSON export."""
        return {
            "edge_neighbors":   self.edge_neighbors,
            "vertex_neighbors": self.vertex_neighbors,
        }

    def save_json(self, path: str):
        """Save the grid to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, path: str) -> "FullereneGrid":
        """Load a grid from a JSON file without re-running the expensive _initialize."""
        g = cls.__new__(cls)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        g.edge_neighbors   = data["edge_neighbors"]
        g.vertex_neighbors = data["vertex_neighbors"]
        return g


_GRID_CACHE_PATH = os.path.join(os.path.dirname(__file__), "grid_cache.json")


def get_grid() -> FullereneGrid:
    """Return a cached FullereneGrid, loading from JSON if available."""
    if os.path.exists(_GRID_CACHE_PATH):
        return FullereneGrid.from_json(_GRID_CACHE_PATH)
    grid = FullereneGrid()
    grid.save_json(_GRID_CACHE_PATH)
    return grid


if __name__ == "__main__":
    g = FullereneGrid()
    g.save_json(_GRID_CACHE_PATH)
    print(f"Pentagons (0-alapú): {PENTAGON_IDS}")
    print(f"Tile 0 él-szomszédok: {g.edge_neighbors[0]}")
    print(f"Tile 0 csúcs-szomszédok: {g.vertex_neighbors[0]}")
    print(f"Tile 1 él-szomszédok: {g.edge_neighbors[1]}")
    missing = [i for i in range(NUM_TILES) if -1 in g.edge_neighbors[i]]
    print(f"Hiányos él-szomszéd-listák: {missing}")
