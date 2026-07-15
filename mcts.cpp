#include "mcts.h"
#include <cmath>
#include <algorithm>

using namespace ag;

namespace oth {

// Softmax the net's logits over the legal actions only, writing priors into nd->P.
// Returns the net's value estimate for the side to move at nd.
float MCTS::expand(Node* nd, Net& net) {
    int A = nd->state.A();
    nd->P.assign(A, 0.f);
    nd->N.assign(A, 0);
    nd->W.assign(A, 0.f);
    nd->child.resize(A);
    nd->legal = nd->state.legal_actions();

    std::vector<float> logits; float value;
    net.eval_state(nd->state, logits, value);

    float m = -1e30f;
    for (int a : nd->legal) m = std::max(m, logits[a]);
    float s = 0.f;
    for (int a : nd->legal) { float e = std::exp(logits[a] - m); nd->P[a] = e; s += e; }
    for (int a : nd->legal) nd->P[a] /= (s + 1e-12f);

    nd->expanded = true;
    return value;
}

int MCTS::select(Node* nd) const {
    float sq = std::sqrt((float)(1 + nd->sumN));
    int best = nd->legal[0]; float best_score = -1e30f;
    for (int a : nd->legal) {
        float q = nd->N[a] > 0 ? nd->W[a] / nd->N[a] : 0.f;
        float u = c_puct * nd->P[a] * sq / (1 + nd->N[a]);
        float score = q + u;
        if (score > best_score) { best_score = score; best = a; }
    }
    return best;
}

float MCTS::simulate(Node* nd, Net& net) {
    if (nd->terminal) return nd->state.terminal_value();
    if (!nd->expanded) return expand(nd, net);

    int a = select(nd);
    if (!nd->child[a]) {
        auto c = std::make_unique<Node>();
        c->state = nd->state;
        c->state.apply(a);
        c->terminal = c->state.is_terminal();
        nd->child[a] = std::move(c);
    }
    float v = -simulate(nd->child[a].get(), net);   // flip to this node's view
    nd->N[a] += 1;
    nd->W[a] += v;
    nd->sumN += 1;
    return v;
}

std::vector<int> MCTS::run(const Othello& root_state, Net& net, bool add_noise) {
    auto root = std::make_unique<Node>();
    root->state = root_state;
    root->terminal = root_state.is_terminal();
    if (root->terminal) return std::vector<int>(root_state.A(), 0);

    expand(root.get(), net);

    if (add_noise && !root->legal.empty()) {
        // Dirichlet(alpha) over legal actions via normalized Gamma(alpha,1) draws
        std::vector<float> g(root->legal.size());
        float gs = 0.f;
        for (size_t i = 0; i < g.size(); ++i) {
            // Gamma sampling: Marsaglia-Tsang (alpha>0). Uses ag::randf() RNG.
            float a = dir_alpha, d = a - 1.0f / 3.0f, cc = 1.0f / std::sqrt(9 * d);
            if (a < 1) d = a + 1 - 1.0f / 3.0f, cc = 1.0f / std::sqrt(9 * d);
            float x, v;
            for (;;) {
                float u1 = randf(), u2 = randf();
                float nrm = std::sqrt(-2.f * std::log(u1 + 1e-12f)) * std::cos(6.2831853f * u2);
                v = 1 + cc * nrm;
                if (v <= 0) continue;
                v = v * v * v;
                float u = randf();
                if (u < 1 - 0.0331f * nrm * nrm * nrm * nrm) { x = d * v; break; }
                if (std::log(u + 1e-12f) < 0.5f * nrm * nrm + d * (1 - v + std::log(v + 1e-12f))) { x = d * v; break; }
            }
            if (dir_alpha < 1) x *= std::pow(randf() + 1e-12f, 1.0f / dir_alpha);
            g[i] = x; gs += x;
        }
        for (size_t i = 0; i < g.size(); ++i) {
            int a = root->legal[i];
            root->P[a] = (1 - dir_eps) * root->P[a] + dir_eps * (g[i] / (gs + 1e-12f));
        }
    }

    for (int i = 0; i < sims; ++i) simulate(root.get(), net);
    return root->N;
}

}  // namespace oth
