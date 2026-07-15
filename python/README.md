# python — NumPyだけで作る AlphaZero（オセロ）Python版

*日本語 | [English](README.en.md)*

親ディレクトリの **C++版 AlphaZero（オセロ）の Python 移植**。ディープラーニングの
フレームワーク（PyTorch/TensorFlow）を一切使わず、**NumPyの配列演算だけ**を土台に、
自作の自動微分エンジンから AlphaZero を組み上げてオセロを自己対戦で強くする。
C++版と1行ずつ読み比べられる。

autograd は [mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp) の
`scratch_py`（NumPy自作autograd）をそのまま流用し、AlphaZero用の op
（tanh / log_softmax / reshape / FC bias）だけ追加している。
**同じ手作りの自動微分が、物体検出にもAlphaZeroにも使える**ことを示す。

## 中身（C++版と一対一対応）

| ファイル | 内容 |
|----------|------|
| `autograd.py` | 自作autograd（Tensor＋計算グラフ＋逆伝播）。conv2d(im2col+einsum)/batchnorm/relu/matmul ＋ **tanh・log_softmax_rows・reshape・add_bias_2d** |
| `game.py` | オセロ規則（合法手・パス・反転・終局・勝敗・エンコード）、盤面サイズ可変 |
| `net.py` | 方策/価値ネット（conv+BN+Residual → policyヘッド＋valueヘッド(tanh)）、損失・保存/読込 |
| `mcts.py` | PUCT探索＋ルートDirichletノイズ |
| `pipeline.py` | 自己対戦・リプレイ・学習ループ・評価 |
| `play.py` | 学習済みネットと人間が対戦 |
| `main.py` | `selftest` / `train` / `play` |

依存は **NumPyのみ**。

## 実行

```bash
cd python
python main.py                                  # 自己テスト（勾配チェック等）
python main.py train --size 6 --iters 40 --out best6.pkl
python main.py play  --size 6 --weights best6.pkl
```

`selftest` は、ゲームロジック・**自作opの数値勾配チェック（全て ~1e-9）**・
ネット損失の逆伝播・MCTSのサニティを確認する。

### 主なオプション（`train`）

`--size`(6/8) `--iters` `--sims` `--games` `--train-steps` `--ch` `--blocks`
`--arena-games` `--lr` `--seed` `--out`。各イテレーションで **loss / 対ランダム勝率 /
対「初期ネット」勝率** を表示。対ランダム勝率が上がれば学習が効いている証拠。

## C++版との違い（速度に注意）

| | C++版（親ディレクトリ） | Python版（ここ） |
|---|---|---|
| 依存 | 標準ライブラリのみ | **NumPyのみ** |
| 数値精度 | float32 | float64（勾配チェックが綺麗に通る） |
| conv実装 | 手書き im2col+GEMM(+AVX2/スレッド) | im2col + `np.einsum` |
| 速度 | 速い（6x6を約9分で学習実証済み） | **かなり遅い**（MCTSが大量のネット評価をするため） |
| 目的 | 実用＋学習 | **読みやすさ・理解に全振り** |

アルゴリズムは完全に同一。**Python版は学習速度では不利**（MCTSは1手ごとに数十回
ネットを評価し、それがNumPyのforwardになるため、C++版の数十倍遅い）。
「AlphaZeroの各部品を1行ずつ読んで理解する」用途向け。実際に強いモデルを回すなら
C++版、仕組みを追うならこのPython版、という住み分け。

## シリーズについて

C++/CPUだけでAIを一から作って理解するシリーズの Python 補完。関連：
[mini-yolov5-cpp](https://github.com/yomei-o/mini-yolov5-cpp)（`scratch_py` にYOLOのPython版）/
[nanoGPT-cpp](https://github.com/yomei-o/nanoGPT-cpp) /
[nanochat-cpp](https://github.com/yomei-o/nanochat-cpp) /
[lecun1989-cpp](https://github.com/yomei-o/lecun1989-cpp)。
