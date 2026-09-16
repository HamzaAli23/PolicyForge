#pragma once

#include "policyforge/gridworld.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace policyforge {

struct ValueIterationConfig {
    double gamma{0.95};
    double tolerance{1e-9};
    int max_iterations{10'000};
};

struct ValueIterationResult {
    std::vector<double> values;
    std::vector<Action> policy;
    std::vector<double> residuals;
    int iterations{};
};

struct QLearningConfig {
    int episodes{5'000};
    int max_steps{200};
    double alpha{0.16};
    double gamma{0.95};
    double epsilon_start{1.0};
    double epsilon_end{0.04};
    std::uint32_t seed{2026};
};

struct QLearningResult {
    std::vector<std::array<double, 4>> q_values;
    std::vector<Action> policy;
    std::vector<double> episode_returns;
    std::vector<int> episode_steps;
    int successful_episodes{};
};

struct EvaluationResult {
    int episodes{};
    int successes{};
    int failures{};
    double success_rate{};
    double average_return{};
    double average_steps{};
};

[[nodiscard]] ValueIterationResult value_iteration(const GridWorld& environment,
                                                    const ValueIterationConfig& config = {});

[[nodiscard]] QLearningResult q_learning(const GridWorld& environment,
                                         const QLearningConfig& config = {});

[[nodiscard]] EvaluationResult evaluate_policy(const GridWorld& environment,
                                               const std::vector<Action>& policy,
                                               int episodes,
                                               int max_steps,
                                               std::uint32_t seed,
                                               int workers = 1);

[[nodiscard]] double policy_agreement(const GridWorld& environment,
                                      const std::vector<Action>& first,
                                      const std::vector<Action>& second);

}  // namespace policyforge
