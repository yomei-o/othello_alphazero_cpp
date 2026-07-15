"""Play against a trained net from the terminal. Python twin of play.cpp."""

import numpy as np

from game import Othello, BLACK, WHITE, EMPTY
from net import Net
from mcts import MCTS


def _print_board(g):
    print("\n   " + " ".join(str(c) for c in range(g.n)))
    board = g.b.reshape(g.n, g.n)
    for r in range(g.n):
        row = " ".join("X" if v == BLACK else ("O" if v == WHITE else ".") for v in board[r])
        print(f" {r} {row}")
    nb = int(np.sum(g.b == BLACK)); nw = int(np.sum(g.b == WHITE))
    print(f"   X(black)={nb}  O(white)={nw}")


def run_play(size, weights, human_color, sims):
    net = Net.from_file(weights)
    m = MCTS(sims=sims)
    g = Othello(size)
    print(f'You are {"X (black)" if human_color == BLACK else "O (white)"}. '
          f'Enter moves as "row col" (or "pass"). X=black moves first.')

    while not g.is_terminal():
        _print_board(g)
        legal = g.legal_actions()
        if g.player == human_color:
            if len(legal) == 1 and legal[0] == g.PASS():
                print("You have no move -> pass."); g.apply(g.PASS()); continue
            try:
                line = input(f'Your move ({"X" if g.player == BLACK else "O"}): ').split()
            except EOFError:
                print("\n(input closed) bye"); return 0
            if not line:
                print('enter "row col" or "pass".'); continue
            if line[0] == "pass":
                action = g.PASS()
            else:
                try:
                    action = int(line[0]) * g.n + int(line[1])
                except (ValueError, IndexError):
                    print('could not parse, try "row col".'); continue
            if action not in legal:
                print("illegal move, try again."); continue
            g.apply(action)
        else:
            v = m.run(g, net, add_noise=False)
            best = int(np.argmax(v))
            print("Engine passes." if best == g.PASS() else f"Engine plays {best // g.n} {best % g.n}")
            g.apply(best)

    _print_board(g)
    d = g.disc_diff(human_color)
    print("\n" + ("You win!" if d > 0 else ("Engine wins." if d < 0 else "Draw.")))
    return 0
