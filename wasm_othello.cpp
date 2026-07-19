// WebAssembly bindings for AlphaZero Othello. Runs the policy/value net + PUCT
// MCTS in the browser so a human can play against the trained agent. The net is
// embedded at build time as net.bin (best6/best8), the board size is fixed by
// -DBOARD_N to match it. Strength at play time is set by the MCTS sim count that
// JS passes in (more sims -> stronger), independent of how far training got.
#include "game.h"
#include "net.h"
#include "mcts.h"
#include "autograd.h"
#include <emscripten.h>
#include <vector>
#include <algorithm>

using namespace ag;
using namespace oth;

#ifndef BOARD_N
#define BOARD_N 8
#endif

static Net*      g_net = nullptr;
static Othello   g_game(BOARD_N);
static std::vector<int> g_board;   // n*n ints in {+1 black, -1 white, 0 empty}
static std::vector<int> g_mask;    // A ints: 1 if action legal for side to move

static void refresh() {
    int hw = BOARD_N * BOARD_N, A = hw + 1;
    g_board.assign(hw, 0);
    for (int i = 0; i < hw; ++i) g_board[i] = g_game.b[i];
    g_mask.assign(A, 0);
    for (int a : g_game.legal_actions()) g_mask[a] = 1;
}

extern "C" {

// Construct the net and load the embedded weights. Returns the board side n.
EMSCRIPTEN_KEEPALIVE
int oth_init() {
    if (!g_net) {
        seed(12345u);
        g_net = new Net(BOARD_N, 32, 3, 32);   // must match the trained arch
        g_net->load("net.bin");
        g_net->training = false;               // eval mode for inference
    }
    g_game.reset();
    refresh();
    return BOARD_N;
}

EMSCRIPTEN_KEEPALIVE int oth_size()      { return BOARD_N; }
EMSCRIPTEN_KEEPALIVE int oth_pass()      { return BOARD_N * BOARD_N; }
EMSCRIPTEN_KEEPALIVE int oth_player()    { return g_game.player; }        // +1 / -1
EMSCRIPTEN_KEEPALIVE int oth_terminal()  { return g_game.is_terminal() ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE int oth_count(int who) {
    int c = 0; for (int8_t v : g_game.b) c += (v == who); return c;
}

EMSCRIPTEN_KEEPALIVE void oth_reset() { g_game.reset(); refresh(); }

// Pointer to the n*n board (int32 per cell). Call after any state change.
EMSCRIPTEN_KEEPALIVE int* oth_board() { return g_board.data(); }
// Pointer to the A-length legality mask for the side to move.
EMSCRIPTEN_KEEPALIVE int* oth_mask()  { return g_mask.data(); }

// Is `action` legal for the side to move? (action in [0, n*n], n*n == pass)
EMSCRIPTEN_KEEPALIVE int oth_is_legal(int action) {
    if (action < 0 || action >= (int)g_mask.size()) return 0;
    return g_mask[action];
}

// Apply a move for the side to move. Returns 1 if legal & applied, else 0.
EMSCRIPTEN_KEEPALIVE int oth_apply(int action) {
    if (!oth_is_legal(action)) return 0;
    g_game.apply(action);
    refresh();
    return 1;
}

// Let the net+MCTS pick and play a move for the side to move using `sims`
// simulations. Returns the action played, or -1 if there was nothing to do.
EMSCRIPTEN_KEEPALIVE
int oth_ai_move(int sims) {
    if (g_game.is_terminal() || !g_net) return -1;
    MCTS m; m.sims = sims > 0 ? sims : 100;
    auto visits = m.run(g_game, *g_net, /*add_noise=*/false);
    int best = -1, bestv = -1;
    for (int a = 0; a < (int)visits.size(); ++a)
        if (visits[a] > bestv) { bestv = visits[a]; best = a; }
    if (best < 0) return -1;
    g_game.apply(best);
    refresh();
    return best;
}

// Net's own evaluation of the current position, in [-1,1] from side-to-move's
// view (>0 means the mover is favoured). Useful for a strength/hint readout.
EMSCRIPTEN_KEEPALIVE
float oth_value() {
    if (!g_net) return 0.f;
    std::vector<float> logits; float v = 0.f;
    g_net->eval_state(g_game, logits, v);
    return v;
}

}  // extern "C"
