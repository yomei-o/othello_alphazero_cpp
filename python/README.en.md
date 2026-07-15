# python — AlphaZero for Othello in NumPy (Python port)

*[日本語](README.md) | English*

A **Python port of the parent C++ AlphaZero (Othello)**. No deep-learning framework
(PyTorch/TensorFlow) at all — a hand-written autograd engine on top of **NumPy array
arithmetic** builds AlphaZero and learns Othello by self-play. Readable side-by-side
with the C++ version, line for line.

The autograd is reused verbatim from the `scratch_py` (NumPy autograd) of
[mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp), with only the AlphaZero
ops added (tanh / log_softmax / reshape / FC bias). The **same hand-made automatic
differentiation drives both object detection and AlphaZero**.

## Contents (one-to-one with the C++ version)

| File | What |
|------|------|
| `autograd.py` | self-made autograd; conv2d(im2col+einsum)/batchnorm/relu/matmul **+ tanh, log_softmax_rows, reshape, add_bias_2d** |
| `game.py` | Othello rules (legal moves, pass, flips, terminal, encoding), configurable board size |
| `net.py` | policy/value net (conv+BN+residual → policy head + value head w/ tanh), loss, save/load |
| `mcts.py` | PUCT search + root Dirichlet noise |
| `pipeline.py` | self-play, replay buffer, training loop, evaluation |
| `play.py` | play against a trained net |
| `main.py` | `selftest` / `train` / `play` |

Only dependency: NumPy.

## Run

```bash
cd python
python main.py                                  # self-test (gradient checks, etc.)
python main.py train --size 6 --iters 40 --out best6.pkl
python main.py play  --size 6 --weights best6.pkl
```

`selftest` checks the game logic, **gradient-checks the new autograd ops (all ~1e-9)**,
the net's loss backprop, and MCTS sanity.

### Key `train` options

`--size`(6/8) `--iters` `--sims` `--games` `--train-steps` `--ch` `--blocks`
`--arena-games` `--lr` `--seed` `--out`. Each iteration prints **loss / win rate vs
random / win rate vs the frozen starting net**. Rising win rates = learning works.

## Differences from the C++ version (mind the speed)

| | C++ (parent dir) | Python (here) |
|---|---|---|
| deps | standard library only | **NumPy only** |
| precision | float32 | float64 (crisp gradient checks) |
| conv impl | hand im2col+GEMM (+AVX2/threads) | im2col + `np.einsum` |
| speed | fast (6x6 trained in ~9 min) | **much slower** (MCTS does many net evals) |
| goal | practical + learning | **readability / understanding** |

The algorithm is identical. **Python is far slower for training** (MCTS evaluates the net
dozens of times per move, and each becomes a NumPy forward — tens of times slower than
the C++). Use this port to *read every piece of AlphaZero line by line*; use the C++
version to actually train strong models.

## About the series

The Python companion in a series that builds AI from scratch in C++/CPU. Related:
[mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp) (YOLO Python port in `scratch_py`) /
[nanoGPT-cpp](https://github.com/yomei-o/nanoGPT-cpp) /
[nanochat-cpp](https://github.com/yomei-o/nanochat-cpp) /
[lecun1989-cpp](https://github.com/yomei-o/lecun1989-cpp).
