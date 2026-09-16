# Test report

## Result

`10/10 tests passed` with GCC and the C++20 build flags defined in the Makefile.

## Coverage

The test executable verifies:

1. bundled environment parsing;
2. transition probability normalization;
3. wall and boundary behaviour;
4. rewards emitted on terminal entry;
5. value-iteration convergence and residual tolerance;
6. Q-learning reproducibility under a fixed seed;
7. learned-policy agreement with the reference policy;
8. deterministic aggregation for parallel evaluation;
9. rejection of a policy with the wrong state count;
10. rejection of an invalid slip probability.

## Run locally

```bash
make test
```

The GitHub Actions workflow also compiles the application, executes a shorter reproducible experiment, and verifies that all report artifacts are non-empty.
