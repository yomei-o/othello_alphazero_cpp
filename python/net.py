"""The AlphaZero policy/value network on the self-made NumPy autograd.

Python twin of net.cpp: a small conv ResNet with two heads (policy logits over
A = n*n+1 actions incl. pass; value scalar in (-1,1) via tanh). Same Conv->BN->ReLU
block reused from the from-scratch YOLO autograd.
"""

import math
import pickle
import numpy as np

import autograd as ag
from autograd import Tensor
from game import Othello


class ConvBN:
    """Conv (no bias) -> BatchNorm -> (ReLU applied by `block`). He-style init."""

    def __init__(self, cin, cout, k, stride, pad):
        self.w = Tensor.randn((cout, cin, k, k), math.sqrt(2.0 / (cin * k * k)), True)
        self.bias = Tensor.zeros((cout,), False)     # fixed zero; BN.beta shifts
        self.gamma = Tensor(np.ones(cout), requires_grad=True)
        self.beta = Tensor.zeros((cout,), requires_grad=True)
        self.run_mean = np.zeros(cout)               # BN buffers (no grad)
        self.run_var = np.ones(cout)
        self.stride, self.pad = stride, pad

    def params(self):
        return [self.w, self.gamma, self.beta]


def _cbn(net, c, x, act):
    h = ag.conv2d(x, c.w, c.bias, c.stride, c.pad)
    h = ag.batchnorm2d(h, c.gamma, c.beta, c.run_mean, c.run_var, net.training)
    return ag.relu(h) if act else h


def _fc(x, w, b):
    return ag.add_bias_2d(ag.matmul(x, w), b)        # (N,in)@(in,out)+ (out)


def _fc_weight(fin, fout):
    return Tensor.randn((fin, fout), math.sqrt(2.0 / fin), True)


class ResBlock:
    def __init__(self, ch):
        self.a = ConvBN(ch, ch, 3, 1, 1)
        self.b = ConvBN(ch, ch, 3, 1, 1)


class Net:
    def __init__(self, boardn, channels=32, nblocks=3, vhid=32):
        self.n, self.ch, self.nblocks, self.vhid = boardn, channels, nblocks, vhid
        self.hw = boardn * boardn
        self.A = self.hw + 1
        self.training = True

        self.stem = ConvBN(3, channels, 3, 1, 1)
        self.blocks = [ResBlock(channels) for _ in range(nblocks)]
        self.phead = ConvBN(channels, 2, 1, 1, 0)
        self.pw = _fc_weight(2 * self.hw, self.A)
        self.pb = Tensor.zeros((self.A,), True)
        self.vhead = ConvBN(channels, 1, 1, 1, 0)
        self.vw1 = _fc_weight(self.hw, vhid)
        self.vb1 = Tensor.zeros((vhid,), True)
        self.vw2 = _fc_weight(vhid, 1)
        self.vb2 = Tensor.zeros((1,), True)

    def forward(self, x):
        N = x.shape[0]
        h = _cbn(self, self.stem, x, True)
        for blk in self.blocks:
            y = _cbn(self, blk.a, h, True)
            y = _cbn(self, blk.b, y, False)
            h = ag.relu(ag.add(h, y))                 # residual
        # policy head
        p = _cbn(self, self.phead, h, True)           # (N,2,n,n)
        p = ag.reshape(p, (N, 2 * self.hw))
        logits = _fc(p, self.pw, self.pb)             # (N,A)
        # value head
        v = _cbn(self, self.vhead, h, True)           # (N,1,n,n)
        v = ag.reshape(v, (N, self.hw))
        v = ag.relu(_fc(v, self.vw1, self.vb1))       # (N,vhid)
        v = ag.tanh_(_fc(v, self.vw2, self.vb2))      # (N,1)
        return logits, v

    def params(self):
        p = []
        p += self.stem.params()
        for blk in self.blocks:
            p += blk.a.params() + blk.b.params()
        p += self.phead.params() + [self.pw, self.pb]
        p += self.vhead.params() + [self.vw1, self.vb1, self.vw2, self.vb2]
        return p

    def buffers(self):
        b = []
        b += [self.stem.run_mean, self.stem.run_var]
        for blk in self.blocks:
            b += [blk.a.run_mean, blk.a.run_var, blk.b.run_mean, blk.b.run_var]
        b += [self.phead.run_mean, self.phead.run_var, self.vhead.run_mean, self.vhead.run_var]
        return b

    def loss(self, x, target_pi, target_z):
        N = x.shape[0]
        logits, v = self.forward(x)
        logp = ag.log_softmax_rows(logits)
        # policy cross-entropy: -(1/N) sum_n sum_a pi * log_softmax
        pol = ag.mul_scalar(ag.sum(ag.mul(target_pi, logp)), -1.0 / N)
        diff = ag.sub(v, target_z)                    # value MSE
        val = ag.mean(ag.mul(diff, diff))
        return ag.add(pol, val)

    def eval_state(self, s):
        """Batch-1 inference for MCTS: returns (logits over A, value scalar)."""
        prev = self.training
        self.training = False
        x = Tensor(s.encode()[None])                  # (1,3,n,n)
        logits, v = self.forward(x)
        self.training = prev
        return logits.data[0].copy(), float(v.data[0, 0])

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p.data[...] = q.data
        for b, c in zip(self.buffers(), other.buffers()):
            b[...] = c

    def config(self):
        return dict(n=self.n, ch=self.ch, nblocks=self.nblocks, vhid=self.vhid)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"config": self.config(),
                         "params": [p.data for p in self.params()],
                         "buffers": [b for b in self.buffers()]}, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        for p, arr in zip(self.params(), d["params"]):
            p.data[...] = arr
        for b, arr in zip(self.buffers(), d["buffers"]):
            b[...] = arr

    @staticmethod
    def from_file(path):
        with open(path, "rb") as f:
            cfg = pickle.load(f)["config"]
        net = Net(cfg["n"], cfg["ch"], cfg["nblocks"], cfg["vhid"])
        net.load(path)
        net.training = False
        return net
