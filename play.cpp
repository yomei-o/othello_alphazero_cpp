#include "game.h"
#include "net.h"
#include "mcts.h"
#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace oth {

static void print_board(const Othello& g) {
    std::printf("\n   ");
    for (int c = 0; c < g.n; ++c) std::printf("%d ", c);
    std::printf("\n");
    for (int r = 0; r < g.n; ++r) {
        std::printf(" %d ", r);
        for (int c = 0; c < g.n; ++c) {
            int v = g.b[r * g.n + c];
            std::printf("%c ", v == BLACK ? 'X' : (v == WHITE ? 'O' : '.'));
        }
        std::printf("\n");
    }
    int nb = 0, nw = 0;
    for (int8_t v : g.b) { nb += (v == BLACK); nw += (v == WHITE); }
    std::printf("   X(black)=%d  O(white)=%d\n", nb, nw);
}

int run_play(int size, const std::string& weights, int human_color, int sims) {
    Net net(size);
    if (!net.load(weights)) { std::printf("could not load weights: %s\n", weights.c_str()); return 1; }
    net.training = false;
    MCTS m; m.sims = sims;

    Othello g(size);
    std::printf("You are %s. Enter moves as \"row col\" (or \"pass\"). X=black moves first.\n",
                human_color == BLACK ? "X (black)" : "O (white)");
    while (!g.is_terminal()) {
        print_board(g);
        auto legal = g.legal_actions();
        if (g.player == human_color) {
            // if the only move is pass, take it automatically
            if (legal.size() == 1 && legal[0] == g.PASS()) {
                std::printf("You have no move -> pass.\n"); g.apply(g.PASS()); continue;
            }
            std::printf("Your move (%s): ", g.player == BLACK ? "X" : "O");
            std::string line;
            if (!std::getline(std::cin, line)) { std::printf("\n(input closed) bye\n"); return 0; }
            std::istringstream ss(line); std::string tok;
            if (!(ss >> tok)) { std::printf("enter \"row col\" or \"pass\".\n"); continue; }
            int action;
            if (tok == "pass") action = g.PASS();
            else {
                int c;
                try { int r = std::stoi(tok); if (!(ss >> c)) throw std::exception(); action = r * g.n + c; }
                catch (...) { std::printf("could not parse, try \"row col\".\n"); continue; }
            }
            bool ok = false; for (int a : legal) if (a == action) ok = true;
            if (!ok) { std::printf("illegal move, try again.\n"); continue; }
            g.apply(action);
        } else {
            auto v = m.run(g, net, false);
            int best = 0; for (int a = 1; a < (int)v.size(); ++a) if (v[a] > v[best]) best = a;
            if (best == g.PASS()) std::printf("Engine passes.\n");
            else std::printf("Engine plays %d %d\n", best / g.n, best % g.n);
            g.apply(best);
        }
    }
    print_board(g);
    int d = g.disc_diff(human_color);
    std::printf("\n%s\n", d > 0 ? "You win!" : (d < 0 ? "Engine wins." : "Draw."));
    return 0;
}

}  // namespace oth
