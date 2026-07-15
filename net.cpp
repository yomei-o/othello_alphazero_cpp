#include "net.h"
#include <cmath>
#include <cstdio>

using namespace ag;

namespace oth {

static Tensor ones(std::vector<int> shape, bool rg) {
    auto t = Tensor::zeros(std::move(shape), rg);
    for (auto& v : t.data()) v = 1.f;
    return t;
}

static ConvBN make_convbn(int cin, int cout, int k, int stride, int pad) {
    ConvBN c;
    c.w = Tensor::randn({cout, cin, k, k}, std::sqrt(2.0f / (cin * k * k)), true);
    c.bias = Tensor::zeros({cout}, false);
    c.gamma = ones({cout}, true);
    c.beta = Tensor::zeros({cout}, true);
    c.run_mean = Tensor::zeros({cout}, false);
    c.run_var = ones({cout}, false);
    c.stride = stride; c.pad = pad;
    return c;
}

// Conv -> BatchNorm -> [ReLU]
static Tensor cbn(Net& net, ConvBN& c, const Tensor& x, bool act) {
    auto h = conv2d(x, c.w, c.bias, c.stride, c.pad);
    h = batchnorm2d(h, c.gamma, c.beta, c.run_mean, c.run_var, net.training);
    return act ? relu(h) : h;
}

// FC: (N,in) @ W(in,out) + b(out)
static Tensor fc(const Tensor& x, const Tensor& w, const Tensor& b) {
    return add_bias_2d(matmul(x, w), b);
}

static Tensor fc_weight(int fin, int fout) {
    return Tensor::randn({fin, fout}, std::sqrt(2.0f / fin), true);
}

Net::Net(int boardn, int channels, int nb, int valhid)
    : n(boardn), ch(channels), nblocks(nb), vhid(valhid) {
    hw = n * n;
    A = hw + 1;
    stem = make_convbn(3, ch, 3, 1, 1);
    for (int i = 0; i < nblocks; ++i)
        blocks.push_back({make_convbn(ch, ch, 3, 1, 1), make_convbn(ch, ch, 3, 1, 1)});
    phead = make_convbn(ch, 2, 1, 1, 0);
    pw = fc_weight(2 * hw, A);
    pb = Tensor::zeros({A}, true);
    vhead = make_convbn(ch, 1, 1, 1, 0);
    vw1 = fc_weight(hw, vhid);
    vb1 = Tensor::zeros({vhid}, true);
    vw2 = fc_weight(vhid, 1);
    vb2 = Tensor::zeros({1}, true);
}

std::pair<Tensor, Tensor> Net::forward(const Tensor& x) {
    int N = x.shape()[0];
    auto h = cbn(*this, stem, x, true);
    for (auto& blk : blocks) {
        auto y = cbn(*this, blk.a, h, true);
        y = cbn(*this, blk.b, y, false);
        h = relu(add(h, y));                       // residual
    }
    // policy head
    auto p = cbn(*this, phead, h, true);           // (N,2,n,n)
    p = reshape(p, {N, 2 * hw});
    auto logits = fc(p, pw, pb);                    // (N,A)
    // value head
    auto v = cbn(*this, vhead, h, true);           // (N,1,n,n)
    v = reshape(v, {N, hw});
    v = relu(fc(v, vw1, vb1));                       // (N,vhid)
    v = tanh_(fc(v, vw2, vb2));                      // (N,1)
    return {logits, v};
}

std::vector<Tensor> Net::params() {
    std::vector<Tensor> p;
    auto add_cbn = [&](ConvBN& c) { p.push_back(c.w); p.push_back(c.gamma); p.push_back(c.beta); };
    add_cbn(stem);
    for (auto& blk : blocks) { add_cbn(blk.a); add_cbn(blk.b); }
    add_cbn(phead); p.push_back(pw); p.push_back(pb);
    add_cbn(vhead); p.push_back(vw1); p.push_back(vb1); p.push_back(vw2); p.push_back(vb2);
    return p;
}

Tensor Net::loss(const Tensor& x, const Tensor& target_pi, const Tensor& target_z) {
    int N = x.shape()[0];
    auto pr = forward(x);
    auto logp = log_softmax_rows(pr.first);
    // policy cross-entropy: -(1/N) sum_n sum_a pi * log_softmax
    auto pol = mul_scalar(sum(mul(target_pi, logp)), -1.0f / N);
    // value MSE
    auto diff = sub(pr.second, target_z);
    auto val = mean(mul(diff, diff));
    return add(pol, val);
}

std::vector<Tensor> Net::buffers() {
    std::vector<Tensor> b;
    auto add_cbn = [&](ConvBN& c) { b.push_back(c.run_mean); b.push_back(c.run_var); };
    add_cbn(stem);
    for (auto& blk : blocks) { add_cbn(blk.a); add_cbn(blk.b); }
    add_cbn(phead); add_cbn(vhead);
    return b;
}

void Net::copy_from(Net& other) {
    auto ps = params(), qs = other.params();
    for (size_t i = 0; i < ps.size(); ++i) ps[i].data() = qs[i].data();
    auto bs = buffers(), cs = other.buffers();
    for (size_t i = 0; i < bs.size(); ++i) bs[i].data() = cs[i].data();
}

bool Net::save(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto write = [&](std::vector<Tensor> ts) {
        for (auto& t : ts) {
            int nsz = t.numel();
            std::fwrite(&nsz, sizeof(int), 1, f);
            std::fwrite(t.data().data(), sizeof(float), nsz, f);
        }
    };
    write(params()); write(buffers());
    std::fclose(f);
    return true;
}

bool Net::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto read = [&](std::vector<Tensor> ts) {
        for (auto& t : ts) {
            int nsz = 0;
            if (std::fread(&nsz, sizeof(int), 1, f) != 1 || nsz != t.numel()) return false;
            std::fread(t.data().data(), sizeof(float), nsz, f);
        }
        return true;
    };
    bool ok = read(params()) && read(buffers());
    std::fclose(f);
    return ok;
}

void Net::eval_state(const Othello& s, std::vector<float>& logits, float& value) {
    // NOTE: does not touch `training` (that would be a data race under parallel
    // self-play). Callers must put the net in eval mode first (training=false),
    // which self-play / arena / eval_vs_random all do.
    std::vector<float> planes;
    s.encode(planes);
    auto x = Tensor::from(std::move(planes), {1, 3, n, n}, false);
    auto pr = forward(x);
    logits.assign(pr.first.data().begin(), pr.first.data().end());   // A values
    value = pr.second.data()[0];
}

}  // namespace oth
