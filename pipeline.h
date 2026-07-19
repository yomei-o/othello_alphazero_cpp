#pragma once
// Self-play data generation, the training loop, and evaluation (arena vs a
// frozen best, and a sanity match vs a random player). This is the AlphaZero
// outer loop wrapped around game.h / net.h / mcts.h.
#include "game.h"
#include "net.h"
#include "mcts.h"
#include <vector>
#include <deque>

namespace oth {

struct Config {
    int size = 6;              // board side (6 fast / 8 real)
    int channels = 32, blocks = 3, vhid = 32;
    int sims = 60;             // MCTS simulations per move
    int iters = 20;            // outer AlphaZero iterations
    int games_per_iter = 15;   // self-play games each iteration
    int temp_moves = 8;        // opening moves sampled at temperature 1
    int train_steps = 200;     // gradient steps per iteration
    int batch = 64;
    int buffer_games = 400;    // replay buffer capacity (in games)
    int arena_games = 20;      // candidate vs best matches
    float arena_thresh = 0.55f;
    float lr = 0.01f, momentum = 0.9f, weight_decay = 1e-4f;
    unsigned seed = 1;
    std::string out = "best.bin";
    std::string resume = "";   // if set, warm-start net/best/init from this weights file
    std::string baseline = ""; // if set, a fixed net used as the arena gate: the
                               // champion is only adopted when it beats this net by
                               // arena_thresh, and this net is shipped as the floor.
    int minutes = 0;           // if >0, stop after roughly this many wall-clock minutes
};

// One training sample: encoded state, MCTS policy target, and (filled in after
// the game ends) the game outcome z from that state's side-to-move perspective.
struct Sample {
    std::vector<float> planes;   // 3*size*size
    std::vector<float> pi;       // A
    int player;                  // side to move when this state occurred
    float z = 0.f;
};

// Play one self-play game with the net; appends samples to `out`.
void self_play_game(Net& net, MCTS& mcts, const Config& cfg, std::vector<Sample>& out);

// Win rate of `cand` vs `opp` over arena_games (alternating colors), greedy MCTS.
float arena(Net& cand, Net& opp, const Config& cfg, int games);

// Win rate of the net (greedy MCTS) vs a uniformly-random legal player.
float eval_vs_random(Net& net, const Config& cfg, int games);

int run_train(const Config& cfg);

}  // namespace oth
