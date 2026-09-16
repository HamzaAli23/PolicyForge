#pragma once

#include "policyforge/algorithms.hpp"

#include <filesystem>

namespace policyforge {

struct ExperimentReport {
    ValueIterationConfig value_config;
    QLearningConfig q_config;
    ValueIterationResult optimal;
    QLearningResult learned;
    EvaluationResult optimal_evaluation;
    EvaluationResult learned_evaluation;
    double agreement_percent{};
    int evaluation_workers{};
};

void write_report(const GridWorld& environment,
                  const ExperimentReport& report,
                  const std::filesystem::path& output_directory,
                  const std::filesystem::path& environment_file);

}  // namespace policyforge
