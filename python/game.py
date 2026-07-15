"""Othello / Reversi rules, NumPy only. Python twin of game.cpp.

Board size is configurable (6 by default for fast CPU self-play, 8 for real
Othello). When encoded for the net the board is always seen from the side to
move, so the policy/value net only ever learns "my stones vs your stones".
"""

import numpy as np

BLACK = +1   # moves first
WHITE = -1
EMPTY = 0

# 8 directions
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Othello:
    def __init__(self, size=6):
        self.n = size
        self.b = np.zeros(size * size, np.int8)   # row-major, values in {EMPTY,BLACK,WHITE}
        self.player = BLACK
        self.reset()

    def A(self):
        return self.n * self.n + 1               # actions: cells + one "pass"

    def PASS(self):
        return self.n * self.n

    def clone(self):
        g = Othello.__new__(Othello)
        g.n, g.b, g.player = self.n, self.b.copy(), self.player
        return g

    def reset(self):
        self.b[:] = EMPTY
        n, c = self.n, self.n // 2
        self.b[(c - 1) * n + (c - 1)] = WHITE
        self.b[(c - 1) * n + c] = BLACK
        self.b[c * n + (c - 1)] = BLACK
        self.b[c * n + c] = WHITE
        self.player = BLACK

    # Does placing `who` at idx bracket a line of opponents in direction (dr,dc)?
    def _flips_in_dir(self, idx, who, dr, dc):
        n = self.n
        r, c = idx // n + dr, idx % n + dc
        seen = 0
        while 0 <= r < n and 0 <= c < n:
            v = self.b[r * n + c]
            if v == -who:
                seen += 1; r += dr; c += dc
            elif v == who:
                return seen > 0                  # own disc closes the bracket
            else:
                break                            # empty -> no bracket
        return False

    def has_move(self, who):
        n = self.n
        for idx in range(n * n):
            if self.b[idx] != EMPTY:
                continue
            for dr, dc in _DIRS:
                if self._flips_in_dir(idx, who, dr, dc):
                    return True
        return False

    def is_terminal(self):
        return not self.has_move(BLACK) and not self.has_move(WHITE)

    def legal_actions(self):
        n = self.n
        out = []
        for idx in range(n * n):
            if self.b[idx] != EMPTY:
                continue
            for dr, dc in _DIRS:
                if self._flips_in_dir(idx, self.player, dr, dc):
                    out.append(idx); break
        if not out and self.has_move(-self.player):
            out.append(self.PASS())              # must pass
        return out                               # empty => terminal

    def apply(self, action):
        if action == self.PASS():
            self.player = -self.player; return
        n, who = self.n, self.player
        self.b[action] = who
        r0, c0 = action // n, action % n
        for dr, dc in _DIRS:
            if not self._flips_in_dir(action, who, dr, dc):
                continue
            r, c = r0 + dr, c0 + dc
            while self.b[r * n + c] == -who:      # flip the bracketed opponents
                self.b[r * n + c] = who
                r += dr; c += dc
        self.player = -self.player

    def disc_diff(self, who):
        return int(np.sum(self.b == who) - np.sum(self.b == -who))

    def terminal_value(self):
        d = self.disc_diff(self.player)          # from side-to-move's perspective
        return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)

    def encode(self):
        """3 planes (my stones, opponent stones, all-ones bias) as (3,n,n)."""
        n = self.n
        board = self.b.reshape(n, n)
        planes = np.zeros((3, n, n), np.float64)
        planes[0] = (board == self.player)
        planes[1] = (board == -self.player)
        planes[2] = 1.0
        return planes
