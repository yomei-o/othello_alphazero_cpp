"""Entry point: selftest (default) / train / play. Python twin of main.cpp.

    python main.py                      # gradient checks + game/MCTS sanity
    python main.py train --size 6       # train (see options below)
    python main.py play  --size 6 --weights best6.pkl
"""

import sys
import numpy as np

import autograd as ag
from autograd import Tensor
from game import Othello, BLACK, EMPTY
from net import Net
from mcts import MCTS
from pipeline import Config, run_train
from play import run_play


# ---- numeric gradient check (central differences) --------------------------
def gradcheck(inputs, build, max_idx=40, eps=1e-4):
    for t in inputs:
        t.zero_grad()
    build().backward()
    analytic = [t.grad.copy() for t in inputs]
    max_err = 0.0
    for ti, t in enumerate(inputs):
        flat = t.data.reshape(-1)
        af = analytic[ti].reshape(-1)
        step = max(1, flat.size // max_idx)
        for i in range(0, flat.size, step):
            orig = flat[i]
            flat[i] = orig + eps; fp = build().item()
            flat[i] = orig - eps; fm = build().item()
            flat[i] = orig
            num = (fp - fm) / (2 * eps)
            denom = max(1.0, abs(af[i]) + abs(num))
            max_err = max(max_err, abs(af[i] - num) / denom)
    return max_err


def check(name, err, tol=1e-2):
    print(f"  {name:<20} max rel err = {err:.2e}   {'OK' if err < tol else '*** FAIL ***'}")


def selftest():
    ag.seed(42)
    print("== game logic ==")
    for sz in (6, 8):
        g = Othello(sz)
        print(f"  {sz}x{sz} start: {len(g.legal_actions())} legal moves (expect 4), "
              f"black to move={g.player == BLACK}")
        r = Othello(sz); moves = 0
        while not r.is_terminal() and moves < sz * sz + 10:
            la = r.legal_actions(); r.apply(la[int(ag.randf() * len(la)) % len(la)]); moves += 1
        filled = int(np.sum(r.b != EMPTY))
        print(f"  {sz}x{sz} random playout: terminal={r.is_terminal()} in {moves} moves, discs={filled}")

    print("\n== autograd new-op gradient checks ==")
    a = Tensor.randn((3, 5), 1, True)
    check("tanh", gradcheck([a], lambda: ag.sum(ag.tanh_(a))))
    a = Tensor.randn((2, 6), 1, True); b = Tensor.randn((6,), 1, True)
    check("add_bias_2d", gradcheck([a, b], lambda: ag.sum(ag.mul(ag.add_bias_2d(a, b), ag.add_bias_2d(a, b)))))
    a = Tensor.randn((2, 3, 2, 2), 1, True)
    check("reshape", gradcheck([a], lambda: ag.sum(ag.mul(ag.reshape(a, (2, 12)), ag.reshape(a, (2, 12))))))
    a = Tensor.randn((3, 4), 1, True)
    pi = Tensor.from_data([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]])
    check("log_softmax", gradcheck([a], lambda: ag.mul_scalar(ag.sum(ag.mul(pi, ag.log_softmax_rows(a))), -1.0)))

    print("\n== net forward + loss gradient check (tiny net) ==")
    net = Net(6, channels=4, nblocks=1, vhid=8)
    x = Tensor.randn((2, 3, 6, 6), 1, False)
    logits, v = net.forward(x)
    print(f"  forward: policy {logits.shape} value {v.shape}, v0={v.data[0,0]:.3f} in (-1,1)")
    pi = Tensor.zeros((2, net.A), False)
    pi.data[0, 0] = 1.0; pi.data[1, 1] = 1.0
    z = Tensor.from_data([[0.5], [-0.5]])
    net.training = True
    check("net loss d/pb", gradcheck([net.pb], lambda: net.loss(x, pi, z), 20), 3e-2)
    check("net loss d/vb2", gradcheck([net.vb2], lambda: net.loss(x, pi, z), 20), 3e-2)

    print("\n== MCTS sanity ==")
    net = Net(6); net.training = False
    v = MCTS(sims=80).run(Othello(6), net, add_noise=False)
    g = Othello(6)
    chosen = int(np.argmax(v))
    print(f"  sims=80 -> sum visits={int(v.sum())} (expect 80), chosen legal={chosen in g.legal_actions()}")
    return 0


def parse(argv):
    c = Config()
    i = 0
    while i + 1 < len(argv):
        k, val = argv[i], argv[i + 1]
        m = {"--size": ("size", int), "--iters": ("iters", int), "--sims": ("sims", int),
             "--games": ("games_per_iter", int), "--train-steps": ("train_steps", int),
             "--batch": ("batch", int), "--ch": ("channels", int), "--blocks": ("blocks", int),
             "--arena-games": ("arena_games", int), "--lr": ("lr", float),
             "--seed": ("seed", int), "--out": ("out", str)}
        if k in m:
            attr, cast = m[k]; setattr(c, attr, cast(val))
        i += 2
    return c


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "selftest":
        sys.exit(selftest())
    if sys.argv[1] == "train":
        sys.exit(run_train(parse(sys.argv[2:])))
    if sys.argv[1] == "play":
        size, color, sims, w = 6, BLACK, 200, "best6.pkl"
        args = sys.argv[2:]
        for i in range(0, len(args) - 1, 2):
            k, val = args[i], args[i + 1]
            if k == "--size": size = int(val)
            elif k == "--weights": w = val
            elif k == "--sims": sims = int(val)
        if "--white" in args:
            from game import WHITE; color = WHITE
        sys.exit(run_play(size, w, color, sims))
    print("usage: python main.py [selftest | train <opts> | play <opts>]")
    sys.exit(1)
