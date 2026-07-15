"""PUCT Monte-Carlo Tree Search guided by the policy/value net (AlphaZero).

Python twin of mcts.cpp. Each simulation descends by maximizing Q + U, expands
one leaf via a single net evaluation, and backs the value up the path flipping
sign each ply. Root priors get Dirichlet noise during self-play.
"""

import math
import numpy as np

import autograd as ag


class Node:
    __slots__ = ("state", "terminal", "expanded", "legal", "P", "N", "W", "child", "sumN")

    def __init__(self, state):
        self.state = state
        self.terminal = False
        self.expanded = False
        self.legal = []
        self.P = None
        self.N = None
        self.W = None
        self.child = None
        self.sumN = 0


class MCTS:
    def __init__(self, sims=100, c_puct=1.5, dir_alpha=0.6, dir_eps=0.25):
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps

    # Softmax the net's logits over legal actions into nd.P; return net value.
    def _expand(self, nd, net):
        A = nd.state.A()
        nd.P = np.zeros(A)
        nd.N = np.zeros(A, np.int64)
        nd.W = np.zeros(A)
        nd.child = [None] * A
        nd.legal = nd.state.legal_actions()

        logits, value = net.eval_state(nd.state)
        idx = np.array(nd.legal, dtype=int)
        z = logits[idx]
        z = np.exp(z - z.max())
        nd.P[idx] = z / (z.sum() + 1e-12)
        nd.expanded = True
        return value

    def _select(self, nd):
        sq = math.sqrt(1 + nd.sumN)
        best, best_score = nd.legal[0], -1e30
        for a in nd.legal:
            q = nd.W[a] / nd.N[a] if nd.N[a] > 0 else 0.0
            u = self.c_puct * nd.P[a] * sq / (1 + nd.N[a])
            score = q + u
            if score > best_score:
                best_score, best = score, a
        return best

    def _simulate(self, nd, net):
        if nd.terminal:
            return nd.state.terminal_value()
        if not nd.expanded:
            return self._expand(nd, net)
        a = self._select(nd)
        if nd.child[a] is None:
            c = Node(nd.state.clone())
            c.state.apply(a)
            c.terminal = c.state.is_terminal()
            nd.child[a] = c
        v = -self._simulate(nd.child[a], net)     # flip to this node's view
        nd.N[a] += 1
        nd.W[a] += v
        nd.sumN += 1
        return v

    def run(self, root_state, net, add_noise):
        root = Node(root_state.clone())
        root.terminal = root_state.is_terminal()
        if root.terminal:
            return np.zeros(root_state.A(), np.int64)

        self._expand(root, net)

        if add_noise and root.legal:
            noise = ag.rng().dirichlet([self.dir_alpha] * len(root.legal))
            for i, a in enumerate(root.legal):
                root.P[a] = (1 - self.dir_eps) * root.P[a] + self.dir_eps * noise[i]

        for _ in range(self.sims):
            self._simulate(root, net)
        return root.N
