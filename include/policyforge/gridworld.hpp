#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace policyforge {

enum class Action : std::size_t { Up = 0, Right = 1, Down = 2, Left = 3 };

constexpr std::array<Action, 4> kActions{Action::Up, Action::Right, Action::Down, Action::Left};

struct Position {
    int x{};
    int y{};
    auto operator<=>(const Position&) const = default;
};

struct Outcome {
    double probability{};
    std::size_t next_state{};
    double reward{};
    bool terminal{};
};

struct StepResult {
    std::size_t next_state{};
    double reward{};
    bool terminal{};
};

class GridWorld {
public:
    static GridWorld load(const std::filesystem::path& path);

    GridWorld(int width,
              int height,
              Position start,
              std::vector<Position> walls,
              std::unordered_map<std::size_t, double> terminal_rewards,
              double step_reward,
              double slip_probability);

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] std::size_t state_count() const noexcept;
    [[nodiscard]] std::size_t start_state() const noexcept { return index(start_); }
    [[nodiscard]] double step_reward() const noexcept { return step_reward_; }
    [[nodiscard]] double slip_probability() const noexcept { return slip_probability_; }
    [[nodiscard]] std::size_t index(Position position) const noexcept;
    [[nodiscard]] Position position(std::size_t state) const noexcept;
    [[nodiscard]] bool is_wall(std::size_t state) const noexcept;
    [[nodiscard]] bool is_terminal(std::size_t state) const noexcept;
    [[nodiscard]] std::optional<double> terminal_reward(std::size_t state) const noexcept;
    [[nodiscard]] std::vector<std::size_t> active_states() const;
    [[nodiscard]] std::vector<Outcome> outcomes(std::size_t state, Action action) const;
    [[nodiscard]] StepResult sample(std::size_t state, Action action, std::mt19937& random) const;
    [[nodiscard]] std::string render_policy(const std::vector<Action>& policy) const;

private:
    int width_;
    int height_;
    Position start_;
    std::vector<bool> walls_;
    std::unordered_map<std::size_t, double> terminal_rewards_;
    double step_reward_;
    double slip_probability_;

    [[nodiscard]] bool in_bounds(Position position) const noexcept;
    [[nodiscard]] std::size_t move(std::size_t state, Action action) const noexcept;
};

[[nodiscard]] const char* action_name(Action action) noexcept;
[[nodiscard]] const char* action_symbol(Action action) noexcept;

}  // namespace policyforge
