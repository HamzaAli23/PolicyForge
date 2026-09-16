#include "policyforge/algorithms.hpp"
#include "policyforge/gridworld.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using policyforge::Action;
using policyforge::GridWorld;
using policyforge::Position;

struct TestCase {
    std::string name;
    std::function<void()> run;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                                 ", received " + std::to_string(actual));
    }
}

GridWorld small_world(double slip = 0.0) {
    return GridWorld(3, 2, {0, 1}, {{1, 0}}, {{2, 10.0}, {5, -5.0}}, -0.1, slip);
}

}  // namespace

int main() {
    const std::vector<TestCase> tests{
        {"loads the bundled environment", [] {
            const auto world = GridWorld::load("examples/campus-navigation.env");
            require(world.width() == 7 && world.height() == 6, "unexpected dimensions");
            require(world.position(world.start_state()) == Position{0, 5}, "unexpected start position");
            require(world.active_states().size() == 37, "unexpected active-state count");
        }},
        {"transition probabilities sum to one", [] {
            const auto world = small_world(0.1);
            double total = 0;
            for (const auto& outcome : world.outcomes(world.start_state(), Action::Right)) {
                total += outcome.probability;
            }
            require_close(total, 1.0, 1e-12, "invalid probability mass");
        }},
        {"walls and boundaries keep the agent in place", [] {
            const auto world = small_world();
            const auto boundary = world.outcomes(world.start_state(), Action::Left);
            require(boundary.size() == 1 && boundary.front().next_state == world.start_state(),
                    "boundary did not block movement");
            const auto wall = world.outcomes(world.index({0, 0}), Action::Right);
            require(wall.size() == 1 && wall.front().next_state == world.index({0, 0}),
                    "wall did not block movement");
        }},
        {"terminal rewards are emitted on entry", [] {
            const auto world = small_world();
            const auto outcome = world.outcomes(world.index({1, 1}), Action::Right).front();
            require(outcome.terminal, "terminal state was not marked");
            require_close(outcome.reward, -5.0, 1e-12, "wrong terminal reward");
        }},
        {"value iteration converges", [] {
            const auto world = GridWorld::load("examples/campus-navigation.env");
            const auto result = policyforge::value_iteration(world);
            require(result.iterations > 1 && result.iterations < 10'000, "solver did not converge");
            require(result.residuals.back() < 1e-9, "final Bellman residual is too large");
            require(result.values[world.start_state()] > 0, "start-state value should be positive");
        }},
        {"Q-learning is reproducible for a fixed seed", [] {
            const auto world = GridWorld::load("examples/campus-navigation.env");
            policyforge::QLearningConfig config;
            config.episodes = 1'200;
            config.seed = 77;
            const auto first = policyforge::q_learning(world, config);
            const auto second = policyforge::q_learning(world, config);
            require(first.policy == second.policy, "policies differ for the same seed");
            require(first.episode_returns == second.episode_returns, "returns differ for the same seed");
        }},
        {"Q-learning approaches the reference policy", [] {
            const auto world = GridWorld::load("examples/campus-navigation.env");
            const auto reference = policyforge::value_iteration(world);
            policyforge::QLearningConfig config;
            config.episodes = 6'000;
            config.seed = 2026;
            const auto learned = policyforge::q_learning(world, config);
            const double agreement = policyforge::policy_agreement(world, reference.policy, learned.policy);
            require(agreement >= 80.0, "policy agreement fell below 80 percent");
        }},
        {"parallel evaluation is reproducible", [] {
            const auto world = GridWorld::load("examples/campus-navigation.env");
            const auto reference = policyforge::value_iteration(world);
            const auto first = policyforge::evaluate_policy(world, reference.policy, 400, 180, 991, 4);
            const auto second = policyforge::evaluate_policy(world, reference.policy, 400, 180, 991, 4);
            require(first.successes == second.successes, "success totals differ");
            require_close(first.average_return, second.average_return, 1e-12, "returns differ");
        }},
        {"invalid policy sizes are rejected", [] {
            const auto world = small_world();
            bool rejected = false;
            try {
                static_cast<void>(policyforge::evaluate_policy(world, {Action::Up}, 10, 10, 1));
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "invalid policy was accepted");
        }},
        {"invalid environment configuration is rejected", [] {
            bool rejected = false;
            try {
                static_cast<void>(GridWorld(2, 2, {0, 0}, {}, {}, -0.1, 0.6));
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "invalid slip probability was accepted");
        }}
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& failure) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << failure.what() << '\n';
        }
    }
    std::cout << '\n' << (tests.size() - static_cast<std::size_t>(failures)) << '/' << tests.size()
              << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
