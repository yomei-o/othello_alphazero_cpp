#pragma once
// The AlphaZero policy/value network, built on the self-made autograd.
// A small conv ResNet with two heads:
//   policy head -> logits over A = n*n + 1 actions (last = pass)
//   value  head -> a scalar in (-1,1) via tanh (expected outcome for side-to-move)
// Same Conv->BN->ReLU building block as the from-scratch YOLO, reused here.
#include "autograd.h"
#include "game.h"
#include <vector>
#include <utility>
#include <string>

namespace oth {

struct ConvBN {
    ag::Tensor w, bias;             // conv weight; bias is a fixed zero (BN shifts)
    ag::Tensor gamma, beta;         // BN affine (learnable)
    ag::Tensor run_mean, run_var;   // BN running stats (no grad)
    int stride, pad;
};

struct ResBlock { ConvBN a, b; };   // conv-bn-relu -> conv-bn -> (+input) -> relu

class Net {
public:
    int n, ch, nblocks, A, hw, vhid;
    bool training = true;

    ConvBN stem;
    std::vector<ResBlock> blocks;
    ConvBN phead;                   // policy conv (ch->2)
    ag::Tensor pw, pb;              // policy FC (2*hw -> A)
    ConvBN vhead;                   // value conv (ch->1)
    ag::Tensor vw1, vb1, vw2, vb2;  // value FC (hw -> vhid -> 1)

    Net(int boardn, int channels = 32, int nb = 3, int valhid = 32);

    // x:(N,3,n,n) -> {policy_logits:(N,A), value:(N,1)}
    std::pair<ag::Tensor, ag::Tensor> forward(const ag::Tensor& x);
    std::vector<ag::Tensor> params();

    // AlphaZero loss: policy cross-entropy(target_pi, logits) + value MSE(v, z).
    ag::Tensor loss(const ag::Tensor& x, const ag::Tensor& target_pi,
                    const ag::Tensor& target_z);

    // Single-state inference for MCTS (eval mode, batch 1): fills raw policy
    // logits over A and the scalar value in (-1,1).
    void eval_state(const Othello& s, std::vector<float>& logits, float& value);

    // BN running stats (not trained but needed for correct inference/serialize).
    std::vector<ag::Tensor> buffers();
    void copy_from(Net& other);           // clone weights+buffers (same arch)
    bool save(const std::string& path);
    bool load(const std::string& path);
};

}  // namespace oth
