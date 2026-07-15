#pragma once
// Monte-Carlo Tree Search with the PUCT rule, guided by the policy/value net
// (AlphaZero-style). Each simulation descends by maximizing Q + U, expands one
// leaf via a single network evaluation, and backs the value up the path with a
// sign flip per ply (players alternate). Root priors get Dirichlet noise during
// self-play so exploration doesn't collapse onto the net's current favorite.
#include "game.h"
#include "net.h"
#include <vector>
#include <memory>

namespace oth {

struct Node {
    Othello state;
    bool terminal = false;
    bool expanded = false;
    std::vector<int> legal;                    // legal actions at this node
    std::vector<float> P;                       // priors over A (0 for illegal)
    std::vector<int> N;                         // visit counts over A
    std::vector<float> W;                       // value sums over A
    std::vector<std::unique_ptr<Node>> child;   // size A, lazily created
    int sumN = 0;
};

class MCTS {
public:
    float c_puct = 1.5f;
    int sims = 100;
    float dir_alpha = 0.6f;    // Dirichlet concentration (smaller board -> larger)
    float dir_eps = 0.25f;     // mixing weight for root noise

    // Run `sims` simulations from `root_state` and return per-action visit counts
    // (size A). add_noise=true injects root Dirichlet noise (self-play only).
    std::vector<int> run(const Othello& root_state, Net& net, bool add_noise);

private:
    float simulate(Node* nd, Net& net);
    float expand(Node* nd, Net& net);
    int select(Node* nd) const;
};

}  // namespace oth
