#include "policyforge/algorithms.hpp"
#include "policyforge/gridworld.hpp"
#include "policyforge/report.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

struct Options {
    std::filesystem::path environment{"examples/campus-navigation.env"};
    std::filesystem::path output{"runs/demo"};
    int episodes{6'000};
    int evaluation_episodes{2'000};
    int max_steps{200};
    int workers{static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))};
    std::uint32_t seed{2026};
};

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
    if (++index >= argc) throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--environment") options.environment = require_value(argc, argv, index, argument);
        else if (argument == "--output") options.output = require_value(argc, argv, index, argument);
        else if (argument == "--episodes") options.episodes = std::stoi(require_value(argc, argv, index, argument));
        else if (argument == "--evaluation-episodes") options.evaluation_episodes = std::stoi(require_value(argc, argv, index, argument));
        else if (argument == "--max-steps") options.max_steps = std::stoi(require_value(argc, argv, index, argument));
        else if (argument == "--workers") options.workers = std::stoi(require_value(argc, argv, index, argument));
        else if (argument == "--seed") options.seed = static_cast<std::uint32_t>(std::stoul(require_value(argc, argv, index, argument)));
        else if (argument == "--help") {
            std::cout << "PolicyForge - reproducible reinforcement-learning experiments\n\n"
                      << "Options:\n"
                      << "  --environment PATH          GridWorld environment file\n"
                      << "  --output PATH               Experiment output directory\n"
                      << "  --episodes N                Q-learning training episodes\n"
                      << "  --evaluation-episodes N     Held-out policy rollouts\n"
                      << "  --max-steps N               Maximum steps per episode\n"
                      << "  --workers N                 Parallel evaluation workers\n"
                      << "  --seed N                    Reproducible random seed\n";
            std::exit(0);
        } else throw std::invalid_argument("Unknown option: " + argument);
    }
    if (options.episodes < 1 || options.evaluation_episodes < 1 || options.max_steps < 1 || options.workers < 1) {
        throw std::invalid_argument("Episode, step and worker counts must be positive");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const policyforge::GridWorld environment = policyforge::GridWorld::load(options.environment);
        policyforge::ValueIterationConfig value_config;
        policyforge::QLearningConfig q_config;
        q_config.episodes = options.episodes;
        q_config.max_steps = options.max_steps;
        q_config.seed = options.seed;

        std::cout << "Computing dynamic-programming reference policy...\n";
        auto optimal = policyforge::value_iteration(environment, value_config);
        std::cout << "Training Q-learning agent for " << options.episodes << " episodes...\n";
        auto learned = policyforge::q_learning(environment, q_config);
        std::cout << "Evaluating policies across " << options.evaluation_episodes << " parallel rollouts...\n";
        auto optimal_evaluation = policyforge::evaluate_policy(environment, optimal.policy,
                options.evaluation_episodes, options.max_steps, options.seed + 10'000, options.workers);
        auto learned_evaluation = policyforge::evaluate_policy(environment, learned.policy,
                options.evaluation_episodes, options.max_steps, options.seed + 10'000, options.workers);
        const double agreement = policyforge::policy_agreement(environment, optimal.policy, learned.policy);

        policyforge::ExperimentReport report{
            value_config, q_config, std::move(optimal), std::move(learned),
            optimal_evaluation, learned_evaluation, agreement, options.workers};
        policyforge::write_report(environment, report, options.output, options.environment);

        std::cout << std::fixed << std::setprecision(2)
                  << "Policy agreement: " << agreement << "%\n"
                  << "Learned policy success rate: " << learned_evaluation.success_rate << "%\n"
                  << "Report: " << (options.output / "report.html").string() << '\n';
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "PolicyForge error: " << failure.what() << '\n';
        return 1;
    }
}
