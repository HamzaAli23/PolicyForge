#include "policyforge/gridworld.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace policyforge {
namespace {

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<double> numbers(const std::string& value) {
    std::vector<double> result;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        result.push_back(std::stod(trim(part)));
    }
    return result;
}

Action left_of(Action action) {
    return static_cast<Action>((static_cast<std::size_t>(action) + 3) % 4);
}

Action right_of(Action action) {
    return static_cast<Action>((static_cast<std::size_t>(action) + 1) % 4);
}

}  // namespace

GridWorld GridWorld::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open environment: " + path.string());

    int width = 0;
    int height = 0;
    Position start{-1, -1};
    std::vector<Position> walls;
    std::vector<std::array<double, 3>> terminals;
    double step_reward = -0.04;
    double slip_probability = 0.10;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("Expected key=value at line " + std::to_string(line_number));
        }
        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (key == "width") width = std::stoi(value);
        else if (key == "height") height = std::stoi(value);
        else if (key == "start") {
            const auto parsed = numbers(value);
            if (parsed.size() != 2) throw std::runtime_error("start requires x,y");
            start = {static_cast<int>(parsed[0]), static_cast<int>(parsed[1])};
        } else if (key == "wall") {
            const auto parsed = numbers(value);
            if (parsed.size() != 2) throw std::runtime_error("wall requires x,y");
            walls.push_back({static_cast<int>(parsed[0]), static_cast<int>(parsed[1])});
        } else if (key == "terminal") {
            const auto parsed = numbers(value);
            if (parsed.size() != 3) throw std::runtime_error("terminal requires x,y,reward");
            terminals.push_back({parsed[0], parsed[1], parsed[2]});
        } else if (key == "step_reward") step_reward = std::stod(value);
        else if (key == "slip_probability") slip_probability = std::stod(value);
        else throw std::runtime_error("Unknown environment key: " + key);
    }

    if (width <= 0 || height <= 0 || start.x < 0 || start.y < 0) {
        throw std::runtime_error("Environment requires positive width/height and a start position");
    }
    std::unordered_map<std::size_t, double> terminal_rewards;
    for (const auto& terminal : terminals) {
        const int x = static_cast<int>(terminal[0]);
        const int y = static_cast<int>(terminal[1]);
        if (x < 0 || x >= width || y < 0 || y >= height) {
            throw std::runtime_error("Terminal position is outside the grid");
        }
        terminal_rewards[static_cast<std::size_t>(y * width + x)] = terminal[2];
    }
    return GridWorld(width, height, start, walls, terminal_rewards, step_reward, slip_probability);
}

GridWorld::GridWorld(int width,
                     int height,
                     Position start,
                     std::vector<Position> walls,
                     std::unordered_map<std::size_t, double> terminal_rewards,
                     double step_reward,
                     double slip_probability)
    : width_(width),
      height_(height),
      start_(start),
      walls_(static_cast<std::size_t>(width * height), false),
      terminal_rewards_(std::move(terminal_rewards)),
      step_reward_(step_reward),
      slip_probability_(slip_probability) {
    if (width <= 0 || height <= 0 || !in_bounds(start)) throw std::invalid_argument("Invalid grid dimensions or start");
    if (slip_probability < 0 || slip_probability > 0.5) throw std::invalid_argument("Slip probability must be in [0, 0.5]");
    for (Position wall : walls) {
        if (!in_bounds(wall)) throw std::invalid_argument("Wall outside the grid");
        walls_[index(wall)] = true;
    }
    if (is_wall(index(start_)) || is_terminal(index(start_))) throw std::invalid_argument("Start must be an active state");
    for (const auto& [state, reward] : terminal_rewards_) {
        (void) reward;
        if (state >= state_count() || is_wall(state)) throw std::invalid_argument("Invalid terminal state");
    }
}

std::size_t GridWorld::state_count() const noexcept {
    return static_cast<std::size_t>(width_ * height_);
}

std::size_t GridWorld::index(Position point) const noexcept {
    return static_cast<std::size_t>(point.y * width_ + point.x);
}

Position GridWorld::position(std::size_t state) const noexcept {
    return {static_cast<int>(state % static_cast<std::size_t>(width_)),
            static_cast<int>(state / static_cast<std::size_t>(width_))};
}

bool GridWorld::is_wall(std::size_t state) const noexcept {
    return state >= walls_.size() || walls_[state];
}

bool GridWorld::is_terminal(std::size_t state) const noexcept {
    return terminal_rewards_.contains(state);
}

std::optional<double> GridWorld::terminal_reward(std::size_t state) const noexcept {
    const auto found = terminal_rewards_.find(state);
    return found == terminal_rewards_.end() ? std::nullopt : std::optional<double>(found->second);
}

std::vector<std::size_t> GridWorld::active_states() const {
    std::vector<std::size_t> states;
    for (std::size_t state = 0; state < state_count(); ++state) {
        if (!is_wall(state)) states.push_back(state);
    }
    return states;
}

std::vector<Outcome> GridWorld::outcomes(std::size_t state, Action action) const {
    if (state >= state_count() || is_wall(state)) throw std::out_of_range("Invalid state");
    if (is_terminal(state)) return {{1.0, state, 0.0, true}};

    const std::array<std::pair<Action, double>, 3> candidates{{
        {action, 1.0 - 2.0 * slip_probability_},
        {left_of(action), slip_probability_},
        {right_of(action), slip_probability_}
    }};
    std::map<std::size_t, double> combined;
    for (const auto& [candidate, probability] : candidates) {
        if (probability > 0) combined[move(state, candidate)] += probability;
    }

    std::vector<Outcome> result;
    for (const auto& [next, probability] : combined) {
        const auto terminal = terminal_reward(next);
        result.push_back({probability, next, terminal.value_or(step_reward_), terminal.has_value()});
    }
    return result;
}

StepResult GridWorld::sample(std::size_t state, Action action, std::mt19937& random) const {
    const auto possible = outcomes(state, action);
    std::uniform_real_distribution<double> draw(0.0, 1.0);
    const double selected = draw(random);
    double cumulative = 0;
    for (const Outcome& outcome : possible) {
        cumulative += outcome.probability;
        if (selected <= cumulative + 1e-12) return {outcome.next_state, outcome.reward, outcome.terminal};
    }
    const Outcome& fallback = possible.back();
    return {fallback.next_state, fallback.reward, fallback.terminal};
}

std::string GridWorld::render_policy(const std::vector<Action>& policy) const {
    if (policy.size() != state_count()) throw std::invalid_argument("Policy size does not match environment");
    std::ostringstream output;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const std::size_t state = index({x, y});
            if (is_wall(state)) output << "#####";
            else if (const auto reward = terminal_reward(state)) {
                output << (reward.value() >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << reward.value();
            } else output << "  " << action_symbol(policy[state]) << "  ";
            if (x + 1 < width_) output << " | ";
        }
        output << '\n';
    }
    return output.str();
}

bool GridWorld::in_bounds(Position point) const noexcept {
    return point.x >= 0 && point.x < width_ && point.y >= 0 && point.y < height_;
}

std::size_t GridWorld::move(std::size_t state, Action action) const noexcept {
    Position next = position(state);
    switch (action) {
        case Action::Up: --next.y; break;
        case Action::Right: ++next.x; break;
        case Action::Down: ++next.y; break;
        case Action::Left: --next.x; break;
    }
    if (!in_bounds(next) || is_wall(index(next))) return state;
    return index(next);
}

const char* action_name(Action action) noexcept {
    switch (action) {
        case Action::Up: return "UP";
        case Action::Right: return "RIGHT";
        case Action::Down: return "DOWN";
        case Action::Left: return "LEFT";
    }
    return "UNKNOWN";
}

const char* action_symbol(Action action) noexcept {
    switch (action) {
        case Action::Up: return "^";
        case Action::Right: return ">";
        case Action::Down: return "v";
        case Action::Left: return "<";
    }
    return "?";
}

}  // namespace policyforge
