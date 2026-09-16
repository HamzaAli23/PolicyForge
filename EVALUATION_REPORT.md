# Evaluation report

## Configuration

| Parameter | Value |
|---|---:|
| Environment | `examples/campus-navigation.env` |
| Random seed | 2026 |
| Q-learning episodes | 6,000 |
| Evaluation episodes per policy | 2,000 |
| Maximum steps per episode | 200 |
| Evaluation workers | 4 |
| Discount factor | 0.95 |
| Learning rate | 0.16 |
| Epsilon schedule | 1.00 to 0.04, exponential |
| Slip probability | 0.10 |

## Results

| Measurement | Value iteration | Q-learning |
|---|---:|---:|
| Positive-terminal episodes | 1,766 | 1,954 |
| Positive-terminal rate | 88.30% | 97.70% |
| Mean discounted return | 3.6930 | 2.9429 |
| Mean steps | 13.9800 | 23.0890 |

Value iteration converged after 61 iterations. The learned policy agreed with the reference policy on 91.1765% of non-wall, non-terminal states.

## Interpretation

The reference policy optimizes expected discounted return, not positive-terminal rate in isolation. It reaches the positive terminal more quickly but accepts more risk near a negative terminal. The learned policy follows a longer route in several states; that route succeeds more often in this fixed rollout set, but its extra step penalties and discounting reduce its mean return.

Policy agreement is also a strict action-by-action comparison. Different actions can have similar expected values, so disagreement does not automatically imply a proportionate performance loss.

## Reproduction

```bash
make all
./build/policyforge \
  --environment examples/campus-navigation.env \
  --output runs/demo \
  --episodes 6000 \
  --evaluation-episodes 2000 \
  --max-steps 200 \
  --workers 4 \
  --seed 2026
```

The generated `summary.json`, policy files, training log, and HTML report provide the raw artifacts for review.

## Limitations

This is one controlled environment and one fixed seed. A comparative study should run multiple seeds, report dispersion, and evaluate additional layouts before drawing broader conclusions about either algorithm.
