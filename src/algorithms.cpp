#include "policyforge/algorithms.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace policyforge {
namespace {

double action_value(const GridWorld& environment,
                    std::size_t state,
                    Action action,
                    const std::vector<double>& values,
                    double gamma) {
    double expected = 0;
    for (const Outcome& outcome : environment.outcomes(state, action)) {
        const double continuation = outcome.terminal ? 0.0 : gamma * values[outcome.next_state];
        expected += outcome.probability * (outcome.reward + continuation);
    }
    return expected;
}

Action best_action(const std::array<double, 4>& values) {
    return static_cast<Action>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

double epsilon_for(int episode, const QLearningConfig& config) {
    if (config.episodes <= 1) return config.epsilon_end;
    const double progress = static_cast<double>(episode) / static_cast<double>(config.episodes - 1);
    return config.epsilon_start * std::pow(config.epsilon_end / config.epsilon_start, progress);
}

struct Aggregate {
    int successes{};
    int failures{};
    double total_return{};
    long long total_steps{};
};

}  // namespace

ValueIterationResult value_iteration(const GridWorld& environment, const ValueIterationConfig& config) {
    if (config.gamma < 0 || config.gamma >= 1 || config.tolerance <= 0 || config.max_iterations < 1) {
        throw std::invalid_argument("Invalid value-iteration configuration");
    }
    std::vector<double> values(environment.state_count(), 0.0);
    std::vector<double> next(values.size(), 0.0);
    std::vector<double> residuals;
    int iteration = 0;
    for (; iteration < config.max_iterations; ++iteration) {
        double residual = 0;
        for (std::size_t state : environment.active_states()) {
            if (environment.is_terminal(state)) {
                next[state] = 0;
                continue;
            }
            double best = -std::numeric_limits<double>::infinity();
            for (Action action : kActions) {
                best = std::max(best, action_value(environment, state, action, values, config.gamma));
            }
            next[state] = best;
            residual = std::max(residual, std::abs(next[state] - values[state]));
        }
        values.swap(next);
        residuals.push_back(residual);
        if (residual < config.tolerance) {
            ++iteration;
            break;
        }
    }

    std::vector<Action> policy(environment.state_count(), Action::Up);
    for (std::size_t state : environment.active_states()) {
        if (environment.is_terminal(state)) continue;
        std::array<double, 4> action_values{};
        for (Action action : kActions) {
            action_values[static_cast<std::size_t>(action)] = action_value(environment, state, action, values, config.gamma);
        }
        policy[state] = best_action(action_values);
    }
    return {std::move(values), std::move(policy), std::move(residuals), iteration};
}

QLearningResult q_learning(const GridWorld& environment, const QLearningConfig& config) {
    if (config.episodes < 1 || config.max_steps < 1 || config.alpha <= 0 || config.alpha > 1 ||
        config.gamma < 0 || config.gamma >= 1 || config.epsilon_start <= 0 || config.epsilon_end <= 0 ||
        config.epsilon_end > config.epsilon_start) {
        throw std::invalid_argument("Invalid Q-learning configuration");
    }
    std::vector<std::array<double, 4>> q_values(environment.state_count());
    std::vector<double> returns;
    std::vector<int> steps;
    returns.reserve(static_cast<std::size_t>(config.episodes));
    steps.reserve(static_cast<std::size_t>(config.episodes));
    std::mt19937 random(config.seed);
    std::uniform_real_distribution<double> explore(0.0, 1.0);
    std::uniform_int_distribution<int> random_action(0, 3);
    int successes = 0;

    for (int episode = 0; episode < config.episodes; ++episode) {
        std::size_t state = environment.start_state();
        double episode_return = 0;
        double discount = 1;
        int completed_steps = 0;
        const double epsilon = epsilon_for(episode, config);
        for (; completed_steps < config.max_steps; ++completed_steps) {
            const Action action = explore(random) < epsilon
                ? static_cast<Action>(random_action(random))
                : best_action(q_values[state]);
            const StepResult transition = environment.sample(state, action, random);
            const double future = transition.terminal
                ? 0.0
                : *std::max_element(q_values[transition.next_state].begin(), q_values[transition.next_state].end());
            double& current = q_values[state][static_cast<std::size_t>(action)];
            current += config.alpha * (transition.reward + config.gamma * future - current);
            episode_return += discount * transition.reward;
            discount *= config.gamma;
            state = transition.next_state;
            if (transition.terminal) {
                if (transition.reward > 0) ++successes;
                ++completed_steps;
                break;
            }
        }
        returns.push_back(episode_return);
        steps.push_back(completed_steps);
    }

    std::vector<Action> policy(environment.state_count(), Action::Up);
    for (std::size_t state : environment.active_states()) {
        if (!environment.is_terminal(state)) policy[state] = best_action(q_values[state]);
    }
    return {std::move(q_values), std::move(policy), std::move(returns), std::move(steps), successes};
}

EvaluationResult evaluate_policy(const GridWorld& environment,
                                 const std::vector<Action>& policy,
                                 int episodes,
                                 int max_steps,
                                 std::uint32_t seed,
                                 int workers) {
    if (policy.size() != environment.state_count() || episodes < 1 || max_steps < 1 || workers < 1) {
        throw std::invalid_argument("Invalid policy-evaluation configuration");
    }
    workers = std::min(workers, episodes);
    std::vector<Aggregate> aggregates(static_cast<std::size_t>(workers));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            std::mt19937 random(seed + static_cast<std::uint32_t>(worker * 7'919));
            Aggregate local;
            for (int episode = worker; episode < episodes; episode += workers) {
                std::size_t state = environment.start_state();
                double episode_return = 0;
                double discount = 1;
                int executed = 0;
                bool reached_positive_terminal = false;
                for (; executed < max_steps; ++executed) {
                    const StepResult transition = environment.sample(state, policy[state], random);
                    episode_return += discount * transition.reward;
                    discount *= 0.95;
                    state = transition.next_state;
                    if (transition.terminal) {
                        reached_positive_terminal = transition.reward > 0;
                        ++executed;
                        break;
                    }
                }
                if (reached_positive_terminal) ++local.successes;
                else ++local.failures;
                local.total_return += episode_return;
                local.total_steps += executed;
            }
            aggregates[static_cast<std::size_t>(worker)] = local;
        });
    }
    for (std::thread& thread : threads) thread.join();

    Aggregate total;
    for (const Aggregate& aggregate : aggregates) {
        total.successes += aggregate.successes;
        total.failures += aggregate.failures;
        total.total_return += aggregate.total_return;
        total.total_steps += aggregate.total_steps;
    }
    return {episodes,
            total.successes,
            total.failures,
            100.0 * total.successes / episodes,
            total.total_return / episodes,
            static_cast<double>(total.total_steps) / episodes};
}

double policy_agreement(const GridWorld& environment,
                        const std::vector<Action>& first,
                        const std::vector<Action>& second) {
    if (first.size() != environment.state_count() || second.size() != environment.state_count()) {
        throw std::invalid_argument("Policy size does not match environment");
    }
    int comparable = 0;
    int matches = 0;
    for (std::size_t state : environment.active_states()) {
        if (environment.is_terminal(state)) continue;
        ++comparable;
        if (first[state] == second[state]) ++matches;
    }
    return comparable == 0 ? 100.0 : 100.0 * matches / comparable;
}

}  // namespace policyforge
