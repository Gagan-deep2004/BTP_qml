"""Grid road network G = (V, E): intersections on an R x C lattice, bidirectional links.

Direction indices are used everywhere (routing probabilities, approaches, qubit outcomes):
    0 = N, 1 = E, 2 = S, 3 = W
A link leaving node u in direction d arrives at node v on approach (d + 2) % 4,
e.g. a southbound link enters v from its North approach.
"""
import numpy as np

N, E, S, W = 0, 1, 2, 3
DIR_NAMES = ["N", "E", "S", "W"]
DIR_DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NS_APPROACHES = (N, S)
EW_APPROACHES = (E, W)


def opposite(d):
    return (d + 2) % 4


class Grid:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.n_nodes = rows * cols

        # neighbour[v, d] = node reached from v going in direction d, or -1
        self.neighbour = -np.ones((self.n_nodes, 4), dtype=int)
        for v in range(self.n_nodes):
            r, c = divmod(v, cols)
            for d, (dr, dc) in enumerate(DIR_DELTA):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    self.neighbour[v, d] = rr * cols + cc

        # Directed links. out_link[v, d] = link leaving v in direction d.
        # in_link[v, a] = link arriving at v on approach a.
        self.link_from, self.link_to, self.link_dir = [], [], []
        self.out_link = -np.ones((self.n_nodes, 4), dtype=int)
        self.in_link = -np.ones((self.n_nodes, 4), dtype=int)
        for v in range(self.n_nodes):
            for d in range(4):
                u = self.neighbour[v, d]
                if u < 0:
                    continue
                lid = len(self.link_from)
                self.link_from.append(v)
                self.link_to.append(u)
                self.link_dir.append(d)
                self.out_link[v, d] = lid
                self.in_link[u, opposite(d)] = lid
        self.link_from = np.array(self.link_from)
        self.link_to = np.array(self.link_to)
        self.link_dir = np.array(self.link_dir)
        self.n_links = len(self.link_from)

        self.valid_mask = (self.neighbour >= 0).astype(float)  # (n_nodes, 4)

    def rc(self, v):
        return divmod(v, self.cols)

    def manhattan(self, a, b):
        ra, ca = divmod(a, self.cols)
        rb, cb = divmod(b, self.cols)
        return abs(ra - rb) + abs(ca - cb)

    def productive_dirs(self, v, dest):
        """Directions from v that reduce the Manhattan distance to dest (1 or 2 of them)."""
        r, c = divmod(v, self.cols)
        rd, cd = divmod(dest, self.cols)
        dirs = []
        if rd < r:
            dirs.append(N)
        if rd > r:
            dirs.append(S)
        if cd > c:
            dirs.append(E)
        if cd < c:
            dirs.append(W)
        return dirs
