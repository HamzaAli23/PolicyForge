# Architecture

PolicyForge separates the environment, algorithms, evaluation, and presentation layers so each can be tested independently.

## Components

| Component | Responsibility |
|---|---|
| `GridWorld` | Parses environment files, validates state geometry, provides transition distributions, and samples stochastic steps. |
| Value iteration | Uses the known transition model to solve the Bellman optimality equation. |
| Q-learning | Learns state-action values from sampled transitions without reading transition probabilities directly. |
| Policy evaluation | Runs held-out episodes across worker threads and merges independent aggregates. |
| Report writer | Produces HTML, JSON, CSV, and plain-text policy artifacts. |
| CLI | Validates experiment options and coordinates one complete run. |

## State and transition model

Each grid coordinate maps to a row-major state index. Walls are excluded from decision states, while terminal states are absorbing during planning and stop an episode during sampling.

For an intended action `a`, the transition distribution is:

```text
P(a)       = 1 - 2s
P(left(a)) = s
P(right(a))= s
```

where `s` is the configured slip probability. Outcomes that lead to the same state are combined, which is important beside walls and grid boundaries.

## Reproducibility

Training uses a Mersenne Twister seeded from the CLI. Each evaluation worker receives a deterministic seed derived from the experiment seed and worker index. Repeating an experiment with identical inputs and worker count therefore reproduces the same artifacts and aggregate metrics.

## Concurrency

Evaluation episodes are distributed by worker index. Workers write only to their own aggregate, so the simulation loop requires no lock. The main thread joins all workers and performs the final reduction.

## Dependency boundary

The core uses only C++20 and the standard library. There is no network access, database, package manager, or runtime service. The Make and CMake builds compile the same source files, while the PowerShell scripts provide a direct path for Windows users with GCC.
