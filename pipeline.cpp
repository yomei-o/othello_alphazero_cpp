#include "pipeline.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

using namespace ag;

namespace oth {

// Sample an action index from a probability vector using the shared RNG.
static int sample_from(const std::vector<float>& p) {
    float r = randf(), c = 0.f;
    for (int i = 0; i < (int)p.size(); ++i) { c += p[i]; if (r <= c) return i; }
    for (int i = (int)p.size() - 1; i >= 0; --i) if (p[i] > 0) return i;
    return 0;
}

static int argmax_i(const std::vector<int>& v) {
    int best = 0; for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

void self_play_game(Net& net, MCTS& mcts, const Config& cfg, std::vector<Sample>& out) {
    Othello g(cfg.size);
    std::vector<Sample> local;
    int move = 0;
    while (!g.is_terminal()) {
        auto visits = mcts.run(g, net, /*add_noise=*/true);
        int A = g.A();
        std::vector<float> pi(A, 0.f);
        double s = 0;
        for (int a = 0; a < A; ++a) s += visits[a];
        if (s <= 0) { g.apply(g.legal_actions()[0]); ++move; continue; }
        for (int a = 0; a < A; ++a) pi[a] = (float)(visits[a] / s);   // proportional to visit counts

        Sample smp;
        g.encode(smp.planes);
        smp.pi = pi;
        smp.player = g.player;
        local.push_back(std::move(smp));

        int a = (move < cfg.temp_moves) ? sample_from(pi) : argmax_i(visits);  // temp 1 -> greedy
        g.apply(a);
        ++move;
    }
    // outcome from BLACK's perspective, then projected onto each sample's player
    int bd = g.disc_diff(BLACK);
    float zb = bd > 0 ? 1.f : (bd < 0 ? -1.f : 0.f);
    for (auto& smp : local) {
        smp.z = (smp.player == BLACK) ? zb : -zb;
        out.push_back(std::move(smp));
    }
}

// Run this iteration's self-play games across worker threads. The net is shared
// read-only (eval mode: forward only, no BN-stat writes, no backward), so games
// run concurrently. Each game reseeds the (thread-local) RNG with a deterministic
// per-game value, so results are reproducible regardless of thread scheduling.
static void parallel_self_play(Net& net, const Config& cfg, int iter,
                               std::vector<Sample>& out) {
    unsigned hw = std::thread::hardware_concurrency();
    int T = std::max(1, std::min((int)(hw ? hw : 1), cfg.games_per_iter));
    std::vector<std::vector<Sample>> locals(T);
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t)
        ths.emplace_back([&, t] {
            MCTS sp; sp.sims = cfg.sims;
            for (int gi = t; gi < cfg.games_per_iter; gi += T) {
                seed((uint64_t)cfg.seed * 1000003ull + (uint64_t)iter * 10007ull + gi + 1);
                self_play_game(net, sp, cfg, locals[t]);
            }
        });
    for (auto& th : ths) th.join();
    for (auto& l : locals)
        for (auto& s : l) out.push_back(std::move(s));
}

// Play one greedy game between two nets; `a_is_black` sets colors. Returns the
// disc difference from netA's perspective (>0 => A won).
static int play_match(Net& A, Net& B, const Config& cfg, bool a_is_black, int sims) {
    Othello g(cfg.size);
    MCTS m; m.sims = sims;
    while (!g.is_terminal()) {
        Net& mover = ((g.player == BLACK) == a_is_black) ? A : B;
        auto v = m.run(g, mover, /*add_noise=*/false);
        g.apply(argmax_i(v));
    }
    int ad = g.disc_diff(a_is_black ? BLACK : WHITE);
    return ad;
}

float arena(Net& cand, Net& opp, const Config& cfg, int games) {
    cand.training = false; opp.training = false;   // eval mode (eval_state won't set it)
    float wins = 0; int played = 0;
    for (int i = 0; i < games; ++i) {
        int ad = play_match(cand, opp, cfg, /*a_is_black=*/(i % 2 == 0), cfg.sims);
        if (ad > 0) wins += 1; else if (ad == 0) wins += 0.5f;
        ++played;
    }
    return played ? wins / played : 0.f;
}

float eval_vs_random(Net& net, const Config& cfg, int games) {
    net.training = false;                          // eval mode
    unsigned hw = std::thread::hardware_concurrency();
    int T = std::max(1, std::min((int)(hw ? hw : 1), games));
    std::vector<float> wins(T, 0.f);
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t)
        ths.emplace_back([&, t] {
            MCTS m; m.sims = cfg.sims;
            for (int i = t; i < games; i += T) {
                seed((uint64_t)cfg.seed * 7919ull + i + 1);   // reproducible per game
                bool net_black = (i % 2 == 0);
                Othello g(cfg.size);
                while (!g.is_terminal()) {
                    if ((g.player == BLACK) == net_black) {
                        g.apply(argmax_i(m.run(g, net, false)));
                    } else {
                        auto la = g.legal_actions();
                        g.apply(la[(int)(randf() * la.size()) % la.size()]);
                    }
                }
                int d = g.disc_diff(net_black ? BLACK : WHITE);
                if (d > 0) wins[t] += 1; else if (d == 0) wins[t] += 0.5f;
            }
        });
    for (auto& th : ths) th.join();
    float total = 0; for (float w : wins) total += w;
    return total / games;
}

int run_train(const Config& cfg) {
    seed(cfg.seed);
    Net net(cfg.size, cfg.channels, cfg.blocks, cfg.vhid);
    Net best(cfg.size, cfg.channels, cfg.blocks, cfg.vhid);  // best-by-winrate checkpoint
    Net init(cfg.size, cfg.channels, cfg.blocks, cfg.vhid);  // frozen starting net (baseline)
    best.copy_from(net);
    init.copy_from(net);

    auto params = net.params();
    std::vector<std::vector<float>> vel(params.size());
    for (size_t i = 0; i < params.size(); ++i) vel[i].assign(params[i].numel(), 0.f);

    std::deque<Sample> buffer;
    size_t buf_cap = (size_t)cfg.buffer_games * cfg.size * cfg.size;  // ~ samples/game * games

    std::printf("== AlphaZero Othello %dx%d (from-scratch autograd, CPU) ==\n", cfg.size, cfg.size);
    std::printf("   ch=%d blocks=%d sims=%d | iters=%d games/iter=%d train/iter=%d batch=%d\n",
                cfg.channels, cfg.blocks, cfg.sims, cfg.iters, cfg.games_per_iter,
                cfg.train_steps, cfg.batch);

    float best_wr = eval_vs_random(net, cfg, 30);
    best.save(cfg.out);
    std::printf("   [init] win vs random = %.0f%%\n", 100 * best_wr);

    for (int it = 1; it <= cfg.iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        // --- self-play with the LATEST net (AlphaZero uses the newest weights,
        //     not a gated "best" -- that gate is AlphaGo Zero's; it can deadlock) ---
        net.training = false;
        std::vector<Sample> fresh;
        parallel_self_play(net, cfg, it, fresh);   // games run across all cores
        for (auto& s : fresh) buffer.push_back(std::move(s));
        while (buffer.size() > buf_cap) buffer.pop_front();

        // --- train ---
        net.training = true;
        int A = cfg.size * cfg.size + 1, hw = cfg.size * cfg.size;
        double loss_acc = 0; int nb = 0;
        for (int step = 0; step < cfg.train_steps && (int)buffer.size() >= cfg.batch; ++step) {
            int B = cfg.batch;
            auto x = Tensor::zeros({B, 3, cfg.size, cfg.size}, false);
            auto tpi = Tensor::zeros({B, A}, false);
            auto tz = Tensor::zeros({B, 1}, false);
            for (int b = 0; b < B; ++b) {
                const Sample& s = buffer[(int)(randf() * buffer.size()) % buffer.size()];
                std::copy(s.planes.begin(), s.planes.end(), x.data().begin() + (size_t)b * 3 * hw);
                std::copy(s.pi.begin(), s.pi.end(), tpi.data().begin() + (size_t)b * A);
                tz.data()[b] = s.z;
            }
            for (auto& p : params) p.zero_grad();
            auto loss = net.loss(x, tpi, tz);
            loss.backward();
            for (size_t i = 0; i < params.size(); ++i) {
                auto& pd = params[i].data(); auto& pg = params[i].grad();
                for (size_t j = 0; j < pd.size(); ++j) {
                    float g = pg[j] + cfg.weight_decay * pd[j];
                    vel[i][j] = cfg.momentum * vel[i][j] - cfg.lr * g;
                    pd[j] += vel[i][j];
                }
            }
            loss_acc += loss.item(); ++nb;
        }

        // --- evaluate: vs a random player (learning signal) and vs the frozen
        //     starting net (shows improvement over where we began) ---
        net.training = false;
        float vr = eval_vs_random(net, cfg, 30);
        float wi = arena(net, init, cfg, cfg.arena_games);
        const char* tag = "";
        if (vr >= best_wr) { best_wr = vr; best.copy_from(net); best.save(cfg.out); tag = " *saved"; }

        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("iter %2d | loss %.3f | vs random %.0f%% (best %.0f%%) | vs init-net %.0f%% | buf %zu | %.1fs%s\n",
                    it, nb ? loss_acc / nb : 0.0, 100 * vr, 100 * best_wr, 100 * wi,
                    buffer.size(), secs, tag);
        std::fflush(stdout);
    }
    std::printf("best (vs-random %.0f%%) saved -> %s\n", 100 * best_wr, cfg.out.c_str());
    return 0;
}

}  // namespace oth
