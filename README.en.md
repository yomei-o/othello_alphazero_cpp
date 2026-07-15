# othello-alphazero-cpp — AlphaZero for Othello, from scratch in C++

*[日本語](README.md) | English*

**AlphaZero built with ZERO external libraries (C++ standard library only, CPU only)**,
on top of a hand-written automatic-differentiation engine, learning Othello (Reversi)
by self-play. No LibTorch, no BLAS. A teaching artifact for *understanding why AlphaZero
gets strong, by building every piece yourself*.

Board size is configurable: **6x6 (default — self-play is tractable on CPU and you can
watch it get stronger)** and **8x8 (real Othello)**.

> **A Python version is also available** → [`python/`](python/) (the same AlphaZero ported
> to NumPy only; readable side-by-side with the C++, though the C++ trains much faster).

## The three loops of AlphaZero (the whole picture)

```
   ┌─────────────┐   self-play games      ┌──────────────┐
   │  self-play   │ ──(state, π, z)──▶  │ replay buffer │
   │ MCTS + net   │                      └──────────────┘
   └─────────────┘                             │ sample
         ▲                                      ▼
         │ save if stronger              ┌──────────────┐
   ┌─────────────┐   eval vs random /     │  train net    │
   │  evaluate    │ ◀──── vs init-net ──  │ policy CE +   │
   └─────────────┘                        │ value MSE     │
                                          └──────────────┘
```

1. **Self-play**: the latest net guides an **MCTS** that plays itself, collecting for
   every position the **MCTS visit distribution π** and the **final result z** as targets.
2. **Train**: push the net's policy toward π and its value toward z
   (**policy = cross-entropy, value = MSE**).
3. **Evaluate**: measure the net vs a random player (learning signal) and vs the frozen
   starting net (improvement over where we began); checkpoint the best-by-win-rate.
   With nothing but self-play, playing strength climbs.

## Contents

| File | What |
|------|------|
| `autograd.h/.cpp` | **Self-made autograd** (reused from the scratch engine of [mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp)): tensor + graph + backprop, conv2d/batchnorm/relu, **plus new ops**: tanh, log_softmax, reshape, FC bias |
| `game.h/.cpp` | Othello rules (legal moves, pass, flips, terminal, result), configurable board size, net-input encoding |
| `net.h/.cpp` | Policy/value net (**conv+BN+Residual** blocks → policy head + value head w/ tanh), loss, weight save/load |
| `mcts.h/.cpp` | **PUCT search** (Q+U selection, expand via one net eval, value backup) + root Dirichlet noise |
| `pipeline.h/.cpp` | Self-play, replay buffer, training loop, evaluation |
| `play.cpp` | CLI to **play against** a trained net |
| `main.cpp` | `selftest` / `train` / `play` |

## Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
# => build/Release/othello.exe
```

On Linux/mac: `cmake -S . -B build && cmake --build build` (`-O3`, pthread). Compiler only.

## Run

```powershell
# 1) self-test (game logic, gradient checks of the new autograd ops, net, MCTS)
build/Release/othello.exe selftest

# 2) train on 6x6
build/Release/othello.exe train --size 6 --iters 40 --out best6.bin

# 3) play the trained net (you are black X; enter moves as "row col", or "pass")
#    the repo ships best6.bin so you can play right away (6x6, lightly trained)
build/Release/othello.exe play --size 6 --weights best6.bin

# real 8x8 Othello (long training on CPU)
build/Release/othello.exe train --size 8 --iters 100 --ch 64 --blocks 5 --out best8.bin
```

### Key `train` options

| flag | default | meaning |
|------|---------|---------|
| `--size` | 6 | board size (6 or 8) |
| `--iters` | 20 | outer AlphaZero iterations |
| `--sims` | 60 | MCTS simulations per move (strength ↔ speed) |
| `--games` | 15 | self-play games per iteration |
| `--train-steps` | 200 | gradient steps per iteration |
| `--ch` / `--blocks` | 32 / 3 | net channels / residual blocks |
| `--lr` | 0.01 | learning rate (SGD + momentum + weight decay) |

Each iteration logs the **loss**, the **win rate vs a random player** (and best-so-far),
and the **win rate vs the frozen starting net**. Rising win rates = learning is working.

## How it works

### 1. MCTS (PUCT)
Each node keeps, per action, `N (visits), W (value sum), P (prior)`. A simulation walks
down by maximizing **Q + U** (`U = c_puct·P·√ΣN /(1+N)`); at an unexpanded leaf it calls
the **net once** for priors and a value, then backs the value up the path **flipping sign
each ply** (good for the opponent = bad for you). The root gets **Dirichlet noise** so
self-play exploration doesn't collapse onto the current favorite.

### 2. Policy/value net
Residual `conv+BN+ReLU` blocks feeding two heads: a **policy head** (logits over
A = n²+1 actions incl. pass) and a **value head** (tanh → expected result in (-1,1) for the
side to move). The input is always **from the side-to-move's view** (my stones / your
stones / all-ones), so the net never has to reason about which color it is.

### 3. Loss
`policy = -Σ π·log_softmax(logits)` + `value = (v − z)²`. `log_softmax` and `tanh` are
added as new autograd ops and **gradient-checked** against finite differences (`selftest`).

### 4. Self-play → train → checkpoint
The first `temp_moves` moves are sampled from π (exploration), the rest greedy. We keep the
**latest** net for self-play (that's AlphaZero — the 55%-gate is AlphaGo Zero's and can
deadlock) and checkpoint whichever net has the best win rate vs random.

## About the series

One of a series that builds AI from scratch in C++/CPU to understand it. The **autograd is
reused verbatim from [mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp)** — the
same hand-made automatic differentiation drives both object detection and AlphaZero.
Related: [nanoGPT-cpp](https://github.com/yomei-o/nanoGPT-cpp) /
[nanochat-cpp](https://github.com/yomei-o/nanochat-cpp) /
[lecun1989-cpp](https://github.com/yomei-o/lecun1989-cpp).

## Notes

- **6x6 is recommended** (trains fast, you can feel it improve). 8x8 on CPU alone takes a
  long time to reach strength.
- Cost is dominated by MCTS net evaluations. Lower `--sims` = faster/weaker, higher =
  slower/stronger.
- Deterministic: fix `--seed` for reproducibility (self-made xorshift RNG).
