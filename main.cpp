#include "autograd.h"
#include "game.h"
#include "net.h"
#include "mcts.h"
#include "pipeline.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <functional>

using namespace ag;
using namespace oth;

namespace oth { int run_play(int size, const std::string& weights, int human_color, int sims); }

// ---- numeric gradient check (central differences) for the new autograd ops --
static float gradcheck(const std::vector<Tensor>& inputs,
                       const std::function<Tensor()>& build, int max_idx = 40, float eps = 1e-3f) {
    for (auto& t : inputs) t.zero_grad();
    Tensor loss = build();
    loss.backward();
    std::vector<std::vector<float>> analytic;
    for (auto& t : inputs) analytic.push_back(t.grad());

    float max_err = 0.f;
    for (size_t ti = 0; ti < inputs.size(); ++ti) {
        auto& d = inputs[ti].data();
        int step = std::max(1, (int)d.size() / max_idx);
        for (size_t i = 0; i < d.size(); i += step) {
            float orig = d[i];
            d[i] = orig + eps; float fp = build().item();
            d[i] = orig - eps; float fm = build().item();
            d[i] = orig;
            float num = (fp - fm) / (2 * eps), a = analytic[ti][i];
            float denom = std::max(1.0f, std::fabs(a) + std::fabs(num));
            max_err = std::max(max_err, std::fabs(a - num) / denom);
        }
    }
    return max_err;
}
static void check(const char* name, float err, float tol = 1e-2f) {
    std::printf("  %-20s max rel err = %.2e   %s\n", name, err, err < tol ? "OK" : "*** FAIL ***");
}

static int selftest() {
    seed(42);
    std::printf("== game logic ==\n");
    for (int sz : {6, 8}) {
        Othello g(sz);
        auto la = g.legal_actions();
        std::printf("  %dx%d start: %zu legal moves (expect 4), black to move=%d\n",
                    sz, sz, la.size(), g.player == BLACK);
        // random playout terminates
        int moves = 0; Othello r(sz);
        while (!r.is_terminal() && moves < sz * sz + 10) {
            auto l = r.legal_actions(); r.apply(l[(int)(randf() * l.size()) % l.size()]); ++moves;
        }
        int filled = 0; for (int8_t v : r.b) filled += (v != EMPTY);
        std::printf("  %dx%d random playout: terminal=%d in %d moves, discs=%d\n",
                    sz, sz, r.is_terminal(), moves, filled);
    }

    std::printf("\n== autograd new-op gradient checks ==\n");
    { auto a = Tensor::randn({3, 5}, 1, true);
      check("tanh", gradcheck({a}, [&] { return sum(tanh_(a)); })); }
    { auto a = Tensor::randn({2, 6}, 1, true), b = Tensor::randn({6}, 1, true);
      check("add_bias_2d", gradcheck({a, b}, [&] { return sum(mul(add_bias_2d(a, b), add_bias_2d(a, b))); })); }
    { auto a = Tensor::randn({2, 3, 2, 2}, 1, true);
      check("reshape", gradcheck({a}, [&] { return sum(mul(reshape(a, {2, 12}), reshape(a, {2, 12}))); })); }
    { // log_softmax against a fixed target distribution: cross-entropy style loss
      auto a = Tensor::randn({3, 4}, 1, true);
      auto pi = Tensor::from({0.1f,0.2f,0.3f,0.4f, 0.25f,0.25f,0.25f,0.25f, 0.7f,0.1f,0.1f,0.1f}, {3, 4}, false);
      check("log_softmax", gradcheck({a}, [&] { return mul_scalar(sum(mul(pi, log_softmax_rows(a))), -1.f); })); }

    std::printf("\n== net forward + loss gradient check (tiny net) ==\n");
    { Net net(6, /*ch=*/4, /*blocks=*/1, /*vhid=*/8);
      auto x = Tensor::randn({2, 3, 6, 6}, 1, false);
      auto pr = net.forward(x);
      std::printf("  forward: policy (%d,%d) value (%d,%d), v0=%.3f in (-1,1)\n",
                  pr.first.shape()[0], pr.first.shape()[1], pr.second.shape()[0], pr.second.shape()[1],
                  pr.second.data()[0]);
      int A = net.A;
      auto pi = Tensor::zeros({2, A}, false);
      for (int b = 0; b < 2; ++b) pi.data()[b * A + (b % A)] = 1.f;
      auto z = Tensor::from({0.5f, -0.5f}, {2, 1}, false);
      net.training = true;
      // check grad wrt the value/policy FC biases (cheap, exercises the full chain)
      check("net loss d/pb", gradcheck({net.pb}, [&] { return net.loss(x, pi, z); }, 20, 1e-2f), 3e-2f);
      check("net loss d/vb2", gradcheck({net.vb2}, [&] { return net.loss(x, pi, z); }, 20, 1e-2f), 3e-2f); }

    std::printf("\n== MCTS sanity ==\n");
    { Net net(6); net.training = false;
      MCTS m; m.sims = 80;
      Othello g(6);
      auto v = m.run(g, net, false);
      int total = 0, chosen = 0; for (int a = 0; a < g.A(); ++a) { total += v[a]; if (v[a] > v[chosen]) chosen = a; }
      bool legal = false; for (int a : g.legal_actions()) if (a == chosen) legal = true;
      std::printf("  sims=80 -> sum visits=%d (expect 80), chosen action legal=%d\n", total, legal); }
    return 0;
}

static Config parse(int argc, char** argv, int start) {
    Config c;
    for (int i = start; i + 1 < argc; i += 2) {
        std::string k = argv[i]; std::string v = argv[i + 1];
        if (k == "--size") c.size = std::atoi(v.c_str());
        else if (k == "--iters") c.iters = std::atoi(v.c_str());
        else if (k == "--sims") c.sims = std::atoi(v.c_str());
        else if (k == "--games") c.games_per_iter = std::atoi(v.c_str());
        else if (k == "--train-steps") c.train_steps = std::atoi(v.c_str());
        else if (k == "--batch") c.batch = std::atoi(v.c_str());
        else if (k == "--ch") c.channels = std::atoi(v.c_str());
        else if (k == "--blocks") c.blocks = std::atoi(v.c_str());
        else if (k == "--arena-games") c.arena_games = std::atoi(v.c_str());
        else if (k == "--lr") c.lr = (float)std::atof(v.c_str());
        else if (k == "--seed") c.seed = (unsigned)std::atoi(v.c_str());
        else if (k == "--out") c.out = v;
    }
    return c;
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "selftest") == 0) return selftest();
    if (std::strcmp(argv[1], "train") == 0) return run_train(parse(argc, argv, 2));
    if (std::strcmp(argv[1], "play") == 0) {
        int size = 6, color = BLACK, sims = 200; std::string w = "best.bin";
        for (int i = 2; i + 1 < argc; i += 2) {
            std::string k = argv[i], v = argv[i + 1];
            if (k == "--size") size = std::atoi(v.c_str());
            else if (k == "--weights") w = v;
            else if (k == "--sims") sims = std::atoi(v.c_str());
            else if (k == "--white") color = WHITE;
        }
        return run_play(size, w, color, sims);
    }
    std::printf("usage: %s [selftest | train <opts> | play <opts>]\n", argv[0]);
    return 1;
}
