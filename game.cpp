#include "game.h"

namespace oth {

// 8 directions
static const int DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int DC[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

void Othello::reset() {
    std::fill(b.begin(), b.end(), (int8_t)EMPTY);
    int c = n / 2;
    b[(c - 1) * n + (c - 1)] = WHITE;
    b[(c - 1) * n + c] = BLACK;
    b[c * n + (c - 1)] = BLACK;
    b[c * n + c] = WHITE;
    player = BLACK;
}

// Does placing `who` at cell idx bracket a line of opponent discs in (dr,dc)?
bool Othello::flips_in_dir(int idx, int who, int dr, int dc) const {
    int r = idx / n, c = idx % n;
    r += dr; c += dc;
    int seen = 0;
    while (r >= 0 && r < n && c >= 0 && c < n) {
        int8_t v = b[r * n + c];
        if (v == -who) { seen++; r += dr; c += dc; }
        else if (v == who) return seen > 0;   // own disc closes the bracket
        else break;                            // empty -> no bracket
    }
    return false;
}

bool Othello::has_move(int who) const {
    for (int idx = 0; idx < n * n; ++idx) {
        if (b[idx] != EMPTY) continue;
        for (int d = 0; d < 8; ++d)
            if (flips_in_dir(idx, who, DR[d], DC[d])) return true;
    }
    return false;
}

bool Othello::is_terminal() const {
    return !has_move(BLACK) && !has_move(WHITE);
}

std::vector<int> Othello::legal_actions() const {
    std::vector<int> out;
    for (int idx = 0; idx < n * n; ++idx) {
        if (b[idx] != EMPTY) continue;
        for (int d = 0; d < 8; ++d)
            if (flips_in_dir(idx, player, DR[d], DC[d])) { out.push_back(idx); break; }
    }
    if (out.empty() && has_move(-player)) out.push_back(PASS());  // must pass
    return out;                                                   // empty => terminal
}

void Othello::apply(int action) {
    if (action == PASS()) { player = -player; return; }
    int who = player;
    b[action] = (int8_t)who;
    int r0 = action / n, c0 = action % n;
    for (int d = 0; d < 8; ++d) {
        if (!flips_in_dir(action, who, DR[d], DC[d])) continue;
        int r = r0 + DR[d], c = c0 + DC[d];
        while (b[r * n + c] == -who) {           // flip the bracketed opponents
            b[r * n + c] = (int8_t)who;
            r += DR[d]; c += DC[d];
        }
    }
    player = -player;
}

int Othello::disc_diff(int who) const {
    int s = 0;
    for (int8_t v : b) s += (v == who) - (v == -who);
    return s;
}

float Othello::terminal_value() const {
    int d = disc_diff(player);
    return d > 0 ? 1.f : (d < 0 ? -1.f : 0.f);
}

void Othello::encode(std::vector<float>& planes) const {
    planes.assign((size_t)3 * n * n, 0.f);
    int hw = n * n;
    for (int i = 0; i < hw; ++i) {
        if (b[i] == player) planes[i] = 1.f;          // plane 0: my stones
        else if (b[i] == -player) planes[hw + i] = 1.f;  // plane 1: opponent
        planes[2 * hw + i] = 1.f;                       // plane 2: all-ones bias
    }
}

}  // namespace oth
