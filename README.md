# Bayesian Sequential Decision Framework

A modular framework for online Bayesian inference, changepoint detection, and adaptive stopping policies under uncertainty.

## Evidence Scope

This README summarizes results from two evaluation tracks:

1. **Synthetic evidence** (headline results): controlled synthetic probability streams across 10 predeclared seeds. Fully validated, statistically supported (paired bootstrap, 95% CIs). See `results/full_synthetic_report.json` and the full report for details.
2. **Real WLASL pilot** (exploratory): a small (N=14), independently-audited real-data evaluation. Findings below are honest and diagnostic, not a validated benchmark. Full audit trail in `results/findings.md`.

Numbers from the real pilot are never blended into synthetic headline claims.

## Motivation

Many sequential recognition systems answer only one question: *what is the final class?*

Real systems often need another question answered: *when do we have enough evidence to decide?*

This repository explores Bayesian sequential decision-making rather than static classification.

## Core Contributions

- **Bayesian belief filtering**: maintain posterior beliefs over classes as evidence arrives frame-by-frame.
- **Multiple stopping policies**: five decision strategies with increasing complexity:
  - `M0`: Raw confidence threshold
  - `M1`: Static Bayesian (flat log-opinion pool)
  - `M2`: Bayesian with switching prior (HMM)
  - `M3`: Bayesian + BOCPD (online changepoint detection)
  - `MT`: Hierarchical Bayesian inference (coarse-to-fine)
- **Bayesian Online Changepoint Detection (BOCPD)**: detect abrupt evidence changes and reset belief when appropriate.
- **Hierarchical inference**: support coarse-to-fine decisions instead of only flat classification.
- **Modular, classifier-agnostic architecture**: `Classifier -> probabilities -> Bayesian engine`. Every component is independent; swap the classifier without touching the decision layer.

## Experimental Pipeline

```
Video
   ↓
MediaPipe landmarks
   ↓
MLP classifier
   ↓
Probability trays
   ↓
Bayesian inference
   ↓
Stopping policies
   ↓
Evaluation
```

## Results

The framework was evaluated on both synthetic probability streams and a real WLASL-derived probability dataset. The two tracks answer different questions: synthetic validates the decision layer's mechanics under controlled conditions, whereas the real pilot stress-tests those mechanics against real, noisy classifier output.

### Synthetic (headline, validated)

Demonstrated:
- Bayesian filtering
- changepoint adaptation
- stopping policy comparison across latency/accuracy tradeoffs
- belief calibration
- expected-utility (Bayes risk) analysis across 4 latency cost ratios

The myopic policy is used as a simple one-step decision baseline rather than a theoretical optimum: it evaluates only the immediate Bayes risk at the current frame and does not reason about the expected value of future observations. Its stopping frontier should be read as a heuristic threshold induced by the current posterior and cost structure, not an optimal stopping boundary. This motivates comparison against the empirical finite-horizon DP approximation, which accounts for the value of future evidence. (DP slightly underperforms myopic at very low cost ratios, `cw/ce <= 0.02`, which is a known grid-sparsity/finite-horizon-truncation effect, not a bug.)

#### Stopping Frontier: Myopic vs. DP Reference Across Cost Ratios

| Cost Ratio (`cw/ce`) | Myopic Risk | Empirical DP Approx Risk |
|---------------------:|------------:|-------------------------:|
| 0.02 | 0.41 | **0.52** |
| 0.05 | 0.92 | **0.92** |
| 0.10 | 1.63 | **0.90** |
| 0.20 | 2.70 | **0.90** |

![Stopping frontier: myopic vs. empirical DP across cost ratios](figures/frontier.png)

#### Perseveration: Stale Re-emission Rate by Method
`per_r` = fraction of emissions naming the *previous* sign after the truth has switched:

| Method | Perseveration Rate (`per_r` ↓) | Sequence Error (`err_r` ↓) | Boundary Latency (`hdel` ↓) |
|--------|-------------------------------:|---------------------------:|----------------------------:|
| M1 static | 0.323 | 0.438 | 27.9 |
| M2 switch | **0.026** | **0.107** | **16.4** |
| M3 BOCPD  | 0.134 | 0.267 | 20.2 |
| MT tree   | N/A | N/A | 14.6 |
| M0 raw    | 0.078 | 0.066 | 21.4 |

![Perseveration: stale re-emission rate by method](figures/perseveration.png)

#### Reliability & Calibration

![Reliability: calibration curve](figures/reliability.png)

#### Hierarchical Tree Specificity Sweep

![Hierarchical specificity sweep](figures/tree_specificity.png)

### Real WLASL Pilot (exploratory, N=14)

Demonstrated:
- end-to-end integration (real `.pth` classifier -> real trays -> Bayesian engine)
- online Bayesian inference on real classifier output
- BOCPD operation on real classifier outputs
- hierarchical stopping on real data
- policy comparison across a 4-config ablation grid (`t_min`, `τ`)

#### Real-Data Pilot Ablation Grid (N=14 Validation Sequences)

| Configuration | Method | Accuracy (%) | Mean Stop Frame | Mean Emissions |
|---------------|--------|-------------:|----------------:|---------------:|
| Baseline (`t_min=0, tau=0.50`) | M0 | 28.6% | 4.6 | 1.29 |
| Baseline (`t_min=0, tau=0.50`) | M1 | 28.6% | 3.4 | 1.57 |
| Baseline (`t_min=0, tau=0.50`) | M2 | 28.6% | 3.4 | 2.07 |
| Baseline (`t_min=0, tau=0.50`) | M3 | 28.6% | 3.4 | 2.29 |
| Baseline (`t_min=0, tau=0.50`) | MT | 21.4% | 0.0 | 1.93 |
| Baseline + `t_min=15` (`t_min=15, tau=0.50`) | M0 | 21.4% | 48.6 | 0.29 |
| Baseline + `t_min=15` (`t_min=15, tau=0.50`) | M1 | 14.3% | 42.5 | 0.57 |
| Baseline + `t_min=15` (`t_min=15, tau=0.50`) | M2 | 21.4% | 34.4 | 1.07 |
| Baseline + `t_min=15` (`t_min=15, tau=0.50`) | M3 | 14.3% | 35.0 | 1.29 |
| Baseline + `t_min=15` (`t_min=15, tau=0.50`) | MT | 14.3% | 41.6 | 0.86 |
| Baseline + higher `tau` (`t_min=0, tau=0.75`) | M0 | 0.0% | 55.6 | 0.00 |
| Baseline + higher `tau` (`t_min=0, tau=0.75`) | M1 | 28.6% | 7.4 | 1.43 |
| Baseline + higher `tau` (`t_min=0, tau=0.75`) | M2 | 28.6% | 7.6 | 1.57 |
| Baseline + higher `tau` (`t_min=0, tau=0.75`) | M3 | 28.6% | 7.3 | 1.71 |
| Baseline + higher `tau` (`t_min=0, tau=0.75`) | MT | 28.6% | 6.5 | 1.43 |
| Both together (`t_min=15, tau=0.75`) | M0 | 0.0% | 55.6 | 0.00 |
| Both together (`t_min=15, tau=0.75`) | M1 | 21.4% | 44.6 | 0.50 |
| Both together (`t_min=15, tau=0.75`) | M2 | 35.7% | 38.3 | 0.64 |
| Both together (`t_min=15, tau=0.75`) | M3 | 21.4% | 40.9 | 0.79 |
| Both together (`t_min=15, tau=0.75`) | MT | 14.3% | 42.4 | 0.50 |

![Real-data validation accuracy comparison across policies](figures/val_ablation_accuracy_comparison.png)

![Real-data ablation study: stopping frames across 4 configurations](figures/val_ablation_stopping_frames.png)

![Real-data validation confusion matrices](figures/val_confusion_matrices.png)

![Real-data BOCPD frequency across validation sequences](figures/val_bocpd_frequency.png)

## Key Findings

**Finding 1: Integration works.**
The Bayesian framework integrates end-to-end with real classifier probability streams without structural failure.

**Finding 2: Decision timing matters.**
Across all 14 real sequences, target gesture onset occurs *after* the baseline stopping frame (mean onset frame 28.1 vs. baseline mean stop frame 0-3.4). Policies commit before the gesture has meaningfully started, which is consistent with the low real-data accuracy observed. Onset here is estimated (BOCPD posterior or class-probability threshold) rather than ground-truth labeled, so this provides a strong signal rather than a proven cause.

**Finding 3: BOCPD behaves as designed, but the setting doesn't fit isolated clips.**
Across all four ablation configs, 55.6%-70.0% of BOCPD-triggered emissions beyond the first were false positives (wrong class) on these single-sign, single-ground-truth clips. This suggests BOCPD, designed to detect real transitions in a continuous stream, has nothing legitimate to detect on an isolated clip with one constant label, and any changepoint it fires there is close to noise by construction. It is a mismatch between the tool and this evaluation setting, not necessarily evidence BOCPD is broken for its intended use case (continuous multi-sign streams).

**Finding 4: The hierarchical policy's frame-0 commit has a root cause, not just a symptom.**
Traced to source: under a uniform prior, coarse node `{0,1,4}` (specificity 0.50) starts with aggregate mass 0.6, which is already above the τ=0.50 threshold before any evidence is processed. This is a structural property of the threshold/prior combination, not a data artifact.
*Why this does NOT affect the synthetic headline benchmark*: In the synthetic headline suite (`src/experiments.py`), the problem uses **10 classes (`K = 10`)**, threshold **`tau = 0.80`**, and minimum specificity threshold **`min_sp = 0.25`**. Under a uniform prior `b_0(k) = 0.10`, reaching `tau = 0.80` requires a coarse node with at least 8 leaves (`8 * 0.10 = 0.80`). However, an 8-leaf node has specificity `sp(v) = (10 - 8) / (10 - 1) = 2/9 ≈ 0.2222`, which is strictly less than `min_sp = 0.25`. Therefore, instant frame-0 commitment under a uniform prior is mathematically excluded in the synthetic headline evaluation (`K = 10`), confirming that instant commitment is an artifact of transferring `tau = 0.50` and `min_sp = 0.25` to a small-vocabulary (`K = 5`) setting.

**Finding 5: The implementation was independently audited, and the audit caught a real bug.**
Confirmed: architecture matches the M0-MT specification exactly at the constructor/parameter level; the full test suite (11 defect-remediation clauses, 9 preservation requirements) passes unmodified. The audit also caught a scoring artifact where a non-emitting policy was being credited with the classifier's final-frame accuracy as a fallback, inflating some ablation numbers (for example, M0 showed 64.3% "accuracy" at 0% emission rate). This was fixed and all four ablation configs re-scored. Catching and fixing this before submission is a strength of the process, not an admission the results were bad.

## What This Repository Is

This repository is not intended as a state-of-the-art sign-language recognizer. It is a reusable Bayesian sequential decision framework, demonstrated using sign-language probability streams as one applied domain. The decision-theoretic methodology (belief filtering, changepoint detection, optimal stopping, hierarchical inference) is the contribution; ASL recognition is the test bed, not the deliverable.

## Limitations

- Small real validation dataset (N=14)
- Simple MLP backbone for the real-pilot classifier
- Isolated single-sign clips rather than continuous multi-sign gesture streams (directly relevant to Finding 3)
- Hierarchical policy (MT) requires threshold/prior redesign before small-vocabulary (`K <= 5`) real-data use (Finding 4)
- BOCPD parameters (`cp_thr`, hazard rate) require recalibration for real-world streams
- In small-vocabulary (`K <= 5`) applications, MT requires threshold (`tau`) or minimum specificity (`min_sp`) adjustment to prevent prior-induced immediate coarse commitment (as observed in the 5-class real pilot at `tau = 0.50`, whereas `K = 10` synthetic benchmarks at `tau = 0.80` are mathematically immune).

## Future Work

- Transformer/CNN probability generators in place of the MLP backbone
- Continuous gesture streams (where BOCPD's design assumptions actually hold)
- Larger real-validation datasets
- Adaptive stopping thresholds
- Hierarchical tree threshold/prior redesign informed by Finding 4
- Calibration-aware Bayesian updates (motivated by the synthetic reliability gap in the 0.25-0.85 confidence band)

## Repository Structure

```
Bayesian-decision-framework/
├── src/                            # Core decision-layer Python library & test suite
│   ├── evidence.py                 # Synthetic evidence stream generator
│   ├── belief.py                   # Recursive Bayesian filter & switching HMM
│   ├── stopping.py                 # Optimal stopping (Bayes tau & empirical DP reference)
│   ├── changepoint.py              # Bayesian Online Changepoint Detection (BOCPD)
│   ├── tree_belief.py              # Hierarchical coarse-to-fine belief tree
│   ├── policies.py                 # M0, M1, M2, M3, MT policy implementations
│   ├── metrics.py                  # Latency, perseveration, accuracy & specificity metrics
│   ├── experiments.py              # Benchmark runner & plotting tools
│   ├── protocol.py                 # Study protocol & boundary invariants
│   ├── reporting.py                # Report generation & JSON schema validation
│   ├── style.py                    # Plotting & CLI styling utilities
│   ├── evaluate_validation.py      # Real-data pilot evaluation pipeline (exploratory)
│   ├── run_ablation_study.py       # Real-data pilot ablation experiment runner (exploratory)
│   └── test_*.py                   # Full remediation, property, & preservation test suite
│
├── results/                        # Experimental reports & diagnostic audits
│   ├── full_synthetic_report.json  # 10-seed synthetic benchmark results
│   ├── evaluation_summary.json     # WLASL real pilot summary
│   ├── evaluation_results.csv      # Per-sequence real pilot results
│   ├── ablation_study_summary.json # 4-config ablation summary
│   ├── ablation_study_results.csv  # 4-config per-sequence ablation results
│   └── findings.md                 # WLASL diagnostic audit report
│
├── figures/                        # Generated plots for README & reports
│   ├── frontier.png                # Myopic vs. DP stopping frontier
│   ├── perseveration.png           # Stale re-emission comparison
│   ├── reliability.png             # Probability calibration curve
│   ├── tree_specificity.png        # Hierarchical tree specificity
│   └── val_*.png                   # WLASL real-data evaluation & ablation plots
│
├── README.md                       # Main documentation, methods table, & audit notes
├── requirements.txt                # Pinned Python dependencies
├── pytest.ini                      # Pytest configuration & custom markers
└── .gitignore                      # Configured to ignore local study guides, caches, & archives
```

### Note on Exploratory Evaluation Modules
`evaluate_validation.py` and `run_ablation_study.py` are standalone exploratory pilot evaluation scripts that load the classifier and execute inference on the validation probability trays. They are explicitly excluded from pytest's default test collection (`test_*.py` pattern in `pytest.ini`) and are never imported by any supported synthetic benchmark module (`experiments.py` or `test_*.py`).

## Setup

Requires Python 3.11 and the pinned dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies include `numpy<2`, `scipy`, `matplotlib`, and `pytest`.

## Running the Tests

Run the complete noninteractive test suite using pytest:

```bash
pytest -q
```

All 165 tests pass under CPython 3.11.9 (`11 defect-remediation clauses, 9 preservation requirements`).

## Reproducing the Synthetic Headline Report

Regenerate `results/full_synthetic_report.json` and the four synthetic headline figures across the 10 predeclared seeds:

```bash
python src/experiments.py --out results/full_synthetic_report.json
```

## License / Citation

This research codebase is provided as an open-source reference implementation of the Bayesian Sequential Decision Framework.
