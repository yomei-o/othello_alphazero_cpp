"""Self-play, replay buffer, training loop, and evaluation. Python twin of
pipeline.cpp -- the AlphaZero outer loop around game/net/mcts.
"""

import time
from collections import deque
import numpy as np

import autograd as ag
from autograd import Tensor
from game import Othello, BLACK, WHITE
from net import Net
from mcts import MCTS


class Config:
    def __init__(self, **kw):
        self.size = 6
        self.channels, self.blocks, self.vhid = 32, 3, 32
        self.sims = 60
        self.iters = 20
        self.games_per_iter = 15
        self.temp_moves = 8
        self.train_steps = 200
        self.batch = 64
        self.buffer_games = 400
        self.arena_games = 20
        self.lr, self.momentum, self.weight_decay = 0.01, 0.9, 1e-4
        self.seed = 1
        self.out = "best.pkl"
        self.__dict__.update(kw)


class Sample:
    __slots__ = ("planes", "pi", "player", "z")

    def __init__(self, planes, pi, player):
        self.planes, self.pi, self.player, self.z = planes, pi, player, 0.0


def _argmax(v):
    return int(np.argmax(v))


def self_play_game(net, mcts, cfg, out):
    g = Othello(cfg.size)
    local = []
    move = 0
    while not g.is_terminal():
        visits = mcts.run(g, net, add_noise=True)
        s = visits.sum()
        if s <= 0:
            g.apply(g.legal_actions()[0]); move += 1; continue
        pi = visits / s                                  # proportional to visit counts
        local.append(Sample(g.encode(), pi.copy(), g.player))
        if move < cfg.temp_moves:                        # temp 1 -> sample, else greedy
            a = int(ag.rng().choice(len(pi), p=pi))
        else:
            a = _argmax(visits)
        g.apply(a)
        move += 1
    # outcome from BLACK's perspective, projected onto each sample's player
    bd = g.disc_diff(BLACK)
    zb = 1.0 if bd > 0 else (-1.0 if bd < 0 else 0.0)
    for smp in local:
        smp.z = zb if smp.player == BLACK else -zb
        out.append(smp)


def _play_match(A, B, cfg, a_is_black, sims):
    g = Othello(cfg.size)
    m = MCTS(sims=sims)
    while not g.is_terminal():
        mover = A if ((g.player == BLACK) == a_is_black) else B
        v = m.run(g, mover, add_noise=False)
        g.apply(_argmax(v))
    return g.disc_diff(BLACK if a_is_black else WHITE)


def arena(cand, opp, cfg, games):
    wins = 0.0
    for i in range(games):
        ad = _play_match(cand, opp, cfg, i % 2 == 0, cfg.sims)
        wins += 1.0 if ad > 0 else (0.5 if ad == 0 else 0.0)
    return wins / games if games else 0.0


def eval_vs_random(net, cfg, games):
    m = MCTS(sims=cfg.sims)
    wins = 0.0
    for i in range(games):
        net_black = (i % 2 == 0)
        g = Othello(cfg.size)
        while not g.is_terminal():
            if (g.player == BLACK) == net_black:
                g.apply(_argmax(m.run(g, net, add_noise=False)))
            else:
                la = g.legal_actions()
                g.apply(la[int(ag.randf() * len(la)) % len(la)])
        d = g.disc_diff(BLACK if net_black else WHITE)
        wins += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    return wins / games


def run_train(cfg):
    ag.seed(cfg.seed)
    net = Net(cfg.size, cfg.channels, cfg.blocks, cfg.vhid)
    best = Net(cfg.size, cfg.channels, cfg.blocks, cfg.vhid)   # best-by-winrate checkpoint
    init = Net(cfg.size, cfg.channels, cfg.blocks, cfg.vhid)   # frozen starting net
    best.copy_from(net)
    init.copy_from(net)

    params = net.params()
    vel = [np.zeros_like(p.data) for p in params]
    buffer = deque()
    buf_cap = cfg.buffer_games * cfg.size * cfg.size

    print(f"== AlphaZero Othello {cfg.size}x{cfg.size} (from-scratch autograd on NumPy) ==")
    print(f"   ch={cfg.channels} blocks={cfg.blocks} sims={cfg.sims} | iters={cfg.iters} "
          f"games/iter={cfg.games_per_iter} train/iter={cfg.train_steps} batch={cfg.batch}")

    sp = MCTS(sims=cfg.sims)
    best_wr = eval_vs_random(net, cfg, 30)
    best.save(cfg.out)
    print(f"   [init] win vs random = {100*best_wr:.0f}%", flush=True)

    A, hw = cfg.size * cfg.size + 1, cfg.size * cfg.size
    for it in range(1, cfg.iters + 1):
        t0 = time.time()
        # self-play with the LATEST net (AlphaZero uses the newest weights)
        net.training = False
        fresh = []
        for _ in range(cfg.games_per_iter):
            self_play_game(net, sp, cfg, fresh)
        buffer.extend(fresh)
        while len(buffer) > buf_cap:
            buffer.popleft()

        # train
        net.training = True
        loss_acc, nb = 0.0, 0
        buf = list(buffer)
        for step in range(cfg.train_steps):
            if len(buf) < cfg.batch:
                break
            B = cfg.batch
            x = np.zeros((B, 3, cfg.size, cfg.size))
            tpi = np.zeros((B, A))
            tz = np.zeros((B, 1))
            for b in range(B):
                smp = buf[int(ag.randf() * len(buf)) % len(buf)]
                x[b] = smp.planes
                tpi[b] = smp.pi
                tz[b, 0] = smp.z
            for p in params:
                p.zero_grad()
            loss = net.loss(Tensor(x), Tensor(tpi), Tensor(tz))
            loss.backward()
            for i, p in enumerate(params):
                g = p.grad + cfg.weight_decay * p.data
                vel[i] = cfg.momentum * vel[i] - cfg.lr * g
                p.data += vel[i]
            loss_acc += loss.item(); nb += 1

        # evaluate: vs random (learning signal) and vs the frozen starting net
        net.training = False
        vr = eval_vs_random(net, cfg, 30)
        wi = arena(net, init, cfg, cfg.arena_games)
        tag = ""
        if vr >= best_wr:
            best_wr = vr; best.copy_from(net); best.save(cfg.out); tag = " *saved"
        print(f"iter {it:2d} | loss {loss_acc/nb if nb else 0:.3f} | vs random {100*vr:.0f}% "
              f"(best {100*best_wr:.0f}%) | vs init-net {100*wi:.0f}% | buf {len(buffer)} "
              f"| {time.time()-t0:.1f}s{tag}", flush=True)

    print(f"best (vs-random {100*best_wr:.0f}%) saved -> {cfg.out}")
    return 0
