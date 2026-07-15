#pragma once
// Othello / Reversi rules, dependency-free. Board size is configurable at
// runtime (6 by default for fast CPU self-play, 8 for real Othello). The board
// is always seen from the side-to-move's perspective when encoded for the net,
// so the policy/value network only ever learns "my stones vs your stones".
#include <vector>
#include <cstdint>

namespace oth {

constexpr int BLACK = +1;   // moves first
constexpr int WHITE = -1;
constexpr int EMPTY = 0;

class Othello {
public:
    int n;                       // board side length (6 or 8)
    std::vector<int8_t> b;       // n*n, values in {EMPTY, BLACK, WHITE}
    int player;                  // side to move (BLACK/WHITE)

    explicit Othello(int size = 6) : n(size), b(size * size, EMPTY), player(BLACK) {
        reset();
    }

    int A() const { return n * n + 1; }   // action space: cells + one "pass"
    int PASS() const { return n * n; }

    void reset();

    // Legal placement actions for the side to move. If there are none but the
    // game is not over, returns {PASS}. Empty only when the game is terminal.
    std::vector<int> legal_actions() const;
    bool has_move(int who) const;         // does `who` have any legal placement?
    bool is_terminal() const;             // neither side can place

    void apply(int action);               // place+flip, or pass; flips `player`

    // Game outcome in [-1,1] from the perspective of the side to move
    // (used as the MCTS leaf value at terminal nodes). +1 win / 0 draw / -1 loss.
    float terminal_value() const;
    int disc_diff(int who) const;         // #who - #opponent

    // Encode into 3 planes (my stones, opponent stones, all-ones bias) laid out
    // as (3, n, n) row-major, from the side-to-move's perspective.
    void encode(std::vector<float>& planes) const;

private:
    bool flips_in_dir(int idx, int who, int dr, int dc) const;
};

}  // namespace oth
