# Bayesian Sequential Decision Framework

A modular framework for online Bayesian inference, changepoint detection, and adaptive stopping policies under uncertainty.

**TL;DR**
- **What it does:** Wraps any frame-level classifier in a sequential Bayesian decision layer that decides *when* enough evidence has accumulated to emit a prediction thus trading off latency against accuracy rather than waiting for a video clip to end.
- **Validated synthetic results ($K = 10$, 10 seeds):** Switching HMM belief filtering (`M2`) cuts post-transition perseveration by **92%** over static Bayesian accumulation (`0.026` vs. `0.323`) and beats raw confidence thresholding on boundary latency (`16.4` vs. `21.4` frames), while empirical Dynamic Programming (`DP`) reduces Bayes risk by up to **55%** over myopic stopping.
- **Exploratory real-data pilot (WLASL, $N = 14$):** End-to-end integration works on real MediaPipe + MLP probability streams, though accuracy is low (`0.0%–35.7%`). Diagnostic auditing traced this to three concrete root causes: pre-onset commitment (stopping at frames `0–3.4` before gestures actually begin around frame `28.1`), BOCPD false positives on single-sign clips that lack real transitions, and coarse-node prior saturation at $K = 5$.

### Links:
[Further explanation of some topics](https://medium.com/@ayushibhattacharya88/devlog-1-bayesian-sequential-decision-framework-pt-1-490c5b16657e): Explaining evidence.py and belief.py

## Evidence Scope

Before you scroll down to the graphs, let us establish one ground rule. This repository evaluates the framework across two completely separate tracks, and I do not mix their numbers:

1. **Synthetic evidence** (headline results): controlled synthetic probability streams across 10 predeclared seeds. This track is fully validated and statistically supported (paired bootstrap, 95% CIs). It exists to verify that the mathematical mechanics actually do what they claim to do under known conditions. See `results/full_synthetic_report.json` for the full breakdown.
2. **Real WLASL pilot** (exploratory): a small ($N = 14$), independently-audited evaluation on real sign-language video data. The findings here are honest and diagnostic. The full audit trail lives in `results/findings.md`.

Numbers from the real pilot are never blended into the synthetic headline claims.

## Motivation & Core Contributions

Imagine you are playing rock, paper, scissors. Your hand does not teleport from rock to paper instantly. There is a physical transition period where your fingers uncurl, and if you pause a video right at the beginning of that movement, even a human will struggle to tell what sign is coming next. Information in the real world does not arrive all at once; it accumulates frame by frame until a concrete story forms.

Yet most sequential recognition systems only bother answering one question: *what is the final class once the clip is over?* Real-time systems have a much harder problem to solve: *when do we have enough evidence to actually commit to a decision?* If you decide too early, you guess on transition noise. If you wait until the very end, you are too slow to be useful. Everything in this repository revolves around that tension: turning noisy, frame-by-frame evidence into an evolving belief, and knowing when to stop waiting.

To do that without welding the decision logic to a specific neural network, the architecture is strictly modular (`Classifier -> probabilities -> Bayesian engine`). You can swap out the upstream classifier tomorrow without touching a single line of the decision layer. Inside that engine, we maintain posterior beliefs over classes as frames arrive, using five stopping policies of increasing sophistication:

- `M0` (**Raw confidence threshold**): The memoryless baseline. If a single frame's raw softmax probability crosses the threshold, it fires. No memory, no patience.
- `M1` (**Static Bayesian**): Accumulates log-evidence across frames using a tempered log-opinion pool. Great when the world is static; terrible when reality changes mid-stream.
- `M2` (**Bayesian with switching prior / HMM**): Adds a small probability of regime change at every step; a tiny drop of mathematical amnesia so the model does not stay stubbornly locked into yesterday's sign.
- `M3` (**Bayesian + BOCPD**): Couples belief filtering with Bayesian Online Changepoint Detection to explicitly monitor whether the underlying sign just switched and reset belief when needed.
- `MT` (**Hierarchical Bayesian inference**): Organizes classes into a coarse-to-fine taxonomy tree. When the evidence is too ambiguous to pick a specific leaf class, it can still commit early to a broader parent category.

## Method at a Glance

The decision layer operates on **probability trays** which are pre-computed per-frame softmax probability matrices ($T \times K$) exported by an upstream classifier. Why call them trays? Because the upstream classifier serves up a flat grid of discriminative probabilities for every sequence, and the Bayesian engine consumes them one row at a time. (Purists will note that a classifier outputs discriminative posteriors rather than generative likelihoods; we treat them as tempered pseudo-likelihoods to update our state, which I unpack further in the Devlog).

Here are the five core formulations driving the engine:

**1. Tempered Recursive Bayes Update (`M1`)**\
If we naively multiply frame probabilities together, we make the very false assumption that consecutive video frames are independent. Because sensor and pose noise are temporally correlated; if a hand is blurry at frame $t$, it is probably still blurry at frame $t+1$—standard Bayes becomes overconfident ridiculously fast by double-counting overlapping information. We fix this by updating posterior belief $b_t(k)$ over $K$ classes with a tempering exponent $\lambda \in (0, 1]$:

$$
b_t(k) \propto b_{t-1}(k) \, p(x_t \mid k)^\lambda
$$

**2. Switching Prior / Sticky HMM (`M2`)**\
Standard Bayes is great until the sign actually changes in real time. Once `M1` builds up hundreds of frames of log-odds for "Rock", incoming evidence for "Paper" cannot easily overturn that mountain of historical bias, a failure mode called *perseveration*. Instead of assuming a sign is static forever, `M2` models reality as a "sticky" Hidden Markov Model. Before fusing frame $t$, we humble the prior by mixing the previous belief with a uniform reset distribution at switching rate $\rho \in (0, 1)$:

$$
\tilde{b}_t(k) = (1 - \rho) \, b_{t-1}(k) + \frac{\rho}{K}
$$

**3. Myopic Bayes-Risk Stopping Rule**\
Forming beliefs is nice, but eventually the agent has to speak up. Given a per-frame waiting cost $c_w$ and a misclassification error cost $c_e$, the myopic rule stops and emits the most likely class when the immediate expected risk of making a mistake falls below the cost of waiting one more frame (which naturally induces a confidence threshold $\tau = 1 - c_w / c_e$):

$$
c_e \left(1 - \max_k b_t(k)\right) \le c_w \iff \max_k b_t(k) \ge 1 - \frac{c_w}{c_e}
$$

**4. BOCPD Run-Length Posterior (`M3`)**\
Rather than leaking belief at a constant rate like `M2`, BOCPD explicitly tracks the "run length" $r_t$; how many frames have elapsed since the last genuine transition under a constant hazard rate $h \in (0, 1)$ and predictive compatibility $\pi_t^{(r)}$, triggering a belief reset when changepoint mass spikes:

$$
P(r_t = r \mid x_{1:t}) \propto P(r_{t-1} = r - 1 \mid x_{1:t-1}) \, (1 - h) \, \pi_t^{(r-1)} \quad (r > 0)
$$

$$
P(r_t = 0 \mid x_{1:t}) \propto \sum_{r'} P(r_{t-1} = r' \mid x_{1:t-1}) \, h \, \pi_t^{(0)}
$$

**5. Hierarchical Specificity (`MT`)**\
When you cannot tell whether a sign is "Apple" or "Onion" yet, you might still be 90% sure it belongs to the "Food" group. `MT` evaluates internal tree nodes $v$ with leaf count $|v|$ across $K$ classes, committing to the most specific node that satisfies both the confidence threshold $B_t(v) = \sum_{k \in v} b_t(k) \ge \tau$ and a minimum specificity floor `min_sp`:

$$
\mathrm{sp}(v) = \frac{K - |v|}{K - 1}
$$

## Experimental Pipeline

![Bayesian Sequential Decision Framework Flowchart](figures/pipeline_flowchart.png)

```text
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

We evaluated the framework across both our controlled synthetic generator (`evidence.py`, complete with co-articulation ramps and AR(1) correlated noise) and a real WLASL-derived probability dataset. Think of the synthetic suite as the wind tunnel where we verify the aerodynamics, and the real pilot as the bumpy road test where we see what rattles loose.

> **Quick Policy Key:** `M0` = Raw confidence threshold | `M1` = Static Bayesian filter | `M2` = Switching HMM filter | `M3` = Bayesian + BOCPD | `MT` = Hierarchical coarse-to-fine tree.

### Synthetic (headline, validated)

In the synthetic benchmark, we test five things: basic belief filtering, adaptation across sign boundaries, the latency-accuracy frontier across stopping policies, probability calibration, and expected-utility (Bayes risk) minimization across four cost ratios.

Look at the stopping frontier below first. The myopic policy is a one-step heuristic baseline, not a theoretical optimum: it only asks whether stopping *right now* beats paying $c_w$ for one step, without reasoning about how much cleaner the signal might get five frames from now. To see what foresight buys us, we compare it against an empirical finite-horizon Dynamic Programming (`DP`) reference that actually accounts for the value of future observations. (You will notice `DP` slightly trails myopic at the lowest cost ratio, `cw/ce <= 0.02`; that is a known finite-horizon grid-discretization effect rather than a bug.)

#### Stopping Frontier: Myopic vs. DP Reference Across Cost Ratios
`cw/ce` = ratio of per-frame wait cost ($c_w$) to misclassification error cost ($c_e$); **Risk** = realized Bayes risk $c_w \cdot \text{Latency} + c_e \cdot (1 - \text{Accuracy})$.

| Cost Ratio (`cw/ce`) | Myopic Accuracy | Myopic Latency | Myopic Risk | DP Accuracy | DP Latency | DP Risk |
|---------------------:|----------------:|---------------:|------------:|------------:|-----------:|--------:|
| 0.02 | **0.943** | 17.7 | **0.42** | 0.890 | **16.5** | 0.44 |
| 0.05 | **0.885** | 16.0 | 0.81 | 0.860 | **10.2** | **0.65** |
| 0.10 | **0.805** | 14.3 | 1.66 | 0.760 | **6.2** | **0.86** |
| 0.20 | **0.668** | 11.8 | 3.12 | 0.550 | **4.8** | **1.41** |

![Stopping frontier: myopic vs. empirical DP across cost ratios](figures/frontier.png)

#### Perseveration: Stale Re-emission Rate by Method
What happens when a person finishes signing "Rock" and transitions to "Paper"?
- `per_r` (Perseveration Rate): the fraction of emissions after a true transition that stubbornly keep naming the *previous* sign.
- `err_r` (Sequence Error): the overall fraction of incorrect emissions across multi-sign sequences.
- `hdel` (Boundary Latency): the average number of frames it takes the policy to wake up and emit the new sign after the ground truth switches.

| Method | Perseveration Rate (`per_r` ↓) | Sequence Error (`err_r` ↓) | Boundary Latency (`hdel` ↓) |
|--------|-------------------------------:|---------------------------:|----------------------------:|
| M1 static | 0.323 | 0.438 | 27.9 |
| M2 switch | **0.026** | 0.107 | 16.4 |
| M3 BOCPD  | 0.134 | 0.267 | 20.2 |
| MT tree   | N/A | 0.220 | **14.6** |
| M0 raw    | 0.078 | **0.066** | 21.4 |

A reader looking at the table above will immediately ask: *Wait, why does the naive memoryless baseline (`M0 raw`) get a lower sequence error (`0.066`) than your fancy switching HMM (`0.107`)?* Good catch. Because `M0` has zero memory, it never carries historical baggage across a boundary, but it pays for that with a 30% slower boundary reaction time (`21.4` frames vs. `16.4` for `M2`) and 3x worse perseveration (`0.078` vs. `0.026`) when transition frames briefly spike the old class. Meanwhile, `M1 static` does exactly what we feared—it locks onto the first sign, resulting in a massive `32.3%` perseveration rate and sluggish `27.9`-frame boundary delay. `M2 switch` cures that lock-in almost completely (`0.026`), and `MT tree` responds fastest of all (`14.6` frames) because it is allowed to hedge with a coarse parent node when the boundary is still blurry.

![Perseveration: stale re-emission rate by method](figures/perseveration.png)

#### Reliability & Calibration

![Reliability: calibration curve](figures/reliability.png)

#### Hierarchical Tree Specificity Sweep

![Hierarchical specificity sweep](figures/tree_specificity.png)

### Real WLASL Pilot (exploratory, N=14)

Now for the reality check. We hooked the exact same Bayesian engine up to real `.pth` MLP classifier outputs across $N = 14$ validation sequences from WLASL and ran a 4-configuration ablation grid sweeping minimum wait time (`t_min`) and confidence threshold (`tau`).

#### Real-Data Pilot Ablation Grid (N=14 Validation Sequences)

If you are looking for high benchmark accuracy, you will not find it here: across all 20 policy-configuration combinations, accuracy sits between `0.0%` and `35.7%`, and at $N = 14$ no single policy is statistically superior. What this pilot actually gives us is much more interesting for system design—it exposes *why* textbook decision rules break when handed real, uncurated video clips, which we dissect in the **Key Findings** right below.

<details>
<summary><strong>Click to expand the 20-row Real-Data Pilot Ablation Table (N=14)</strong></summary>

<br>

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
| Both together (`t_min=15, tau=0.75`) | M2 | **35.7%** | 38.3 | 0.64 |
| Both together (`t_min=15, tau=0.75`) | M3 | 21.4% | 40.9 | 0.79 |
| Both together (`t_min=15, tau=0.75`) | MT | 14.3% | 42.4 | 0.50 |

</details>

![Real-data validation accuracy comparison across policies](figures/val_ablation_accuracy_comparison.png)

![Real-data ablation study: stopping frames across 4 configurations](figures/val_ablation_stopping_frames.png)

![Real-data validation confusion matrices](figures/val_confusion_matrices.png)

![Real-data BOCPD frequency across validation sequences](figures/val_bocpd_frequency.png)

## Key Findings

**1. The plumbing works end-to-end.**
First, the sanity check: the Bayesian engine ingests real `.pth` classifier probability trays, runs recursive filtering, tracks changepoints, and evaluates hierarchical trees without a single structural crash.

**2. The policies are jumping the gun before the signer even raises their hands.**
Why is baseline accuracy stuck around `28.6%`? Look at the clock. Across all 14 real videos, the baseline policies make up their minds and stop between frames `0` and `3.4`. Meanwhile, our diagnostic onset estimates show the actual target gesture does not meaningfully start until around frame `28.1` on average! In real video clips, people spend the first half-second standing still or lifting their arms, during which the MLP classifier confidently outputs biased idle-pose probabilities. Because the Bayesian filter trusts that steady stream of early frames, it crosses the threshold long before the real sign begins. Forcing the model to wait (`t_min=15`, `tau=0.75`) pushes `M2`'s mean stop frame out to `38.3` and bumps accuracy to `35.7%`.

**3. BOCPD is looking for transitions that do not exist in isolated clips.**
Across all four ablation settings, `55.6%–70.0%` of the extra emissions triggered by BOCPD after the first decision were false positives. Why? Because BOCPD is built to detect regime changes in a *continuous, multi-sign stream*, whereas the WLASL pilot clips are isolated single-sign recordings with one constant label. Handing an isolated clip to a changepoint detector is like hiring a fire lookout to stare at a blank wall, there are no legitimate transitions to find, so any jitter in the classifier's output that trips the detector is guaranteed to be a false alarm.

**4. Why `MT` committed at frame 0 (and why the synthetic benchmark is immune).**
In the baseline real pilot (`K = 5`), the hierarchical policy `MT` had a mean stop frame of `0.0`—it was making a decision before seeing a single frame of video. When we traced this to its mathematical root, the culprit was simple arithmetic: under a uniform 5-class prior ($0.20$ per class), the 3-leaf coarse parent node `{0,1,4}` starts out at frame 0 with an aggregate belief mass of $3 \times 0.20 = 0.60$ and a specificity of $(5 - 3)/(5 - 1) = 0.50$. Because the pilot used $\tau = 0.50$ and `min_sp = 0.25`, that parent node already satisfied both stopping criteria at initialization! Fortunately, this is impossible in our 10-class synthetic benchmark (expand the proof below if you want to check the math).

<details>
<summary><strong>Click to view why this does NOT affect the synthetic headline benchmark (K=10 proof)</strong></summary>

<br>

In the synthetic headline suite (`src/experiments.py`), the problem uses **10 classes ($K = 10$)**, threshold **$\tau = 0.80$**, and minimum specificity threshold **`min_sp = 0.25`**. Under a uniform prior $b_0(k) = 0.10$, reaching $\tau = 0.80$ requires a coarse node with at least 8 leaves ($8 \times 0.10 = 0.80$). However, an 8-leaf node has specificity:

$$
\text{sp}(v) = \frac{10 - 8}{10 - 1} = \frac{2}{9} \approx 0.2222
$$

Because $0.2222$ is strictly less than `min_sp = 0.25`, instant frame-0 commitment under a uniform prior is mathematically excluded in the synthetic headline evaluation ($K = 10$), confirming that instant commitment is an artifact of transferring $\tau = 0.50$ and `min_sp = 0.25` to a small-vocabulary ($K = 5$) setting.

</details>

**5. The verification audit caught a sneaky fallback scoring bug.**
Every policy constructor and mathematical invariant was checked against a spec-driven verification test suite (`11 defect-remediation clauses, 9 preservation requirements` in `src/test_*.py`), which also audited the real-pilot evaluation script (`results/findings.md`). That audit caught a subtle bug: whenever a conservative policy refused to emit a prediction before the video ended (for instance, `M0` at `tau=0.75`, which had a `0%` emission rate), the evaluation script was silently falling back to the classifier's final-frame argmax and crediting `M0` with `64.3%` accuracy for doing nothing. We ripped out that fallback and re-scored all four ablation configurations honestly: if a policy fails to emit, it gets a zero.

## What This Repository Is (and Isn't)

Let us be completely clear: this repository is **not** trying to be a state-of-the-art sign-language recognizer. If you want high accuracy on WLASL, you do not use a lightweight MLP on raw landmarks across 14 clips. Sign language recognition here is simply the test bed. The actual deliverable is the reusable **Bayesian sequential decision layer**—the belief filtering, tempering, switching priors, changepoint resets, and optimal stopping mechanics—which you can bolt onto any noisy sequential classifier where decision timing matters.

## Limitations & Future Work

Because I believe in stating what is broken before someone else points it out, the current limitations of the exploratory real-data setup are straightforward:
- **Tiny validation sample ($N = 14$)** and a **simple MLP backbone** that produces poorly calibrated frame probabilities during rest poses.
- **Isolated single-sign clips** rather than continuous multi-sign streams, which deprives BOCPD (`M3`) and switching priors (`M2`) of the actual sign-to-sign transitions they were built to handle (Finding 3).
- **Small-vocabulary tree calibration ($K \le 5$):** Using `MT` on 5 classes requires raising $\tau$ or `min_sp` so coarse parent nodes do not fire on the uniform prior alone (Finding 4), and BOCPD's hazard rate $h$ and threshold `cp_thr` need recalibration for real sensor noise.

Where does this go next? Right now, our stopping rules rely on frozen hyperparameters, which is a bit rigid for the chaotic real world:
- **Dynamic Policy Optimization (MDP):** Instead of relying on static Empirical Dynamic Programming grids, the next iteration frames optimal stopping as a full Markov Decision Process, training Reinforcement Learning agents to dynamically tune wait cost ($c_w$), error cost ($c_e$), and switching rate ($\rho$) on the fly based on stream volatility.
- **Multi-Agent Reinforcement Learning (MARL):** Putting multiple decision agents on the exact same evidence stream in a competitive game-theoretic setup, forcing them to balance threshold confidence against a competitor beating them to the punch (finding a Nash Equilibrium between speed and recklessness).
- **Continuous Gesture Streams & Deep Backbones:** Moving from isolated clips to continuous, multi-sign videos where BOCPD actually has real transitions to catch, and swapping the toy MLP for Transformer or CNN backbones with richer uncertainty profiles.

## Repository Structure

```text
Bayesian-decision-framework/
├── src/                                # Core decision-layer Python library & test suite
│   ├── evidence.py                     # Synthetic evidence stream generator
│   ├── belief.py                       # Recursive Bayesian filter & switching HMM
│   ├── stopping.py                     # Optimal stopping (Bayes tau & empirical DP reference)
│   ├── changepoint.py                  # Bayesian Online Changepoint Detection (BOCPD)
│   ├── tree_belief.py                  # Hierarchical coarse-to-fine belief tree
│   ├── policies.py                     # M0, M1, M2, M3, MT policy implementations
│   ├── metrics.py                      # Latency, perseveration, accuracy & specificity metrics
│   ├── experiments.py                  # Benchmark runner & plotting tools
│   ├── protocol.py                     # Study protocol & boundary invariants
│   ├── reporting.py                    # Report generation & JSON schema validation
│   ├── style.py                        # Plotting & CLI styling utilities
│   ├── evaluate_validation.py          # Real-data pilot evaluation pipeline (exploratory)
│   ├── run_ablation_study.py           # Real-data pilot ablation experiment runner (exploratory)
│   └── test_*.py                       # Full remediation, property, & preservation test suite
│
├── results/                            # Experimental reports & diagnostic audits
│   ├── full_synthetic_report.json      # 10-seed synthetic benchmark results
│   ├── evaluation_summary.json         # WLASL real pilot summary
│   ├── evaluation_results.csv          # Per-sequence real pilot results
│   ├── ablation_study_summary.json     # 4-config ablation summary
│   ├── ablation_study_results.csv      # 4-config per-sequence ablation results
│   └── findings.md                     # WLASL diagnostic audit report
│
├── figures/                            # Generated plots for README & reports
│   ├── pipeline_flowchart.png          # Experimental pipeline diagram
│   ├── frontier.png                    # Myopic vs. DP stopping frontier
│   ├── perseveration.png               # Stale re-emission comparison
│   ├── reliability.png                 # Probability calibration curve
│   ├── tree_specificity.png            # Hierarchical tree specificity
│   └── val_*.png                       # WLASL real-data evaluation & ablation plots
│
├── LICENSE                             # MIT License
├── CITATION.cff                        # Citation metadata
├── README.md                           # Main documentation, methods table, & audit notes
├── requirements.txt                    # Pinned Python dependencies
├── pytest.ini                          # Pytest configuration & custom markers
└── .gitignore                          # Configured to ignore local study guides, caches, & archives
```

### Note on Exploratory Evaluation Modules
`evaluate_validation.py` and `run_ablation_study.py` are standalone exploratory pilot scripts that load the classifier and run inference on the validation probability trays. They are explicitly excluded from `pytest`'s default collection (`test_*.py` pattern in `pytest.ini`) and are never imported by the synthetic benchmark modules (`experiments.py` or `test_*.py`).

## Setup & Reproduction

Requires Python 3.11 and the pinned dependencies in `requirements.txt` (`numpy<2`, `scipy`, `matplotlib`, and `pytest`):

```bash
pip install -r requirements.txt
```

To run the complete noninteractive test suite (all 165 tests pass under CPython 3.11.9, covering all 11 defect-remediation clauses and 9 preservation requirements):

```bash
pytest -q
```

To regenerate `results/full_synthetic_report.json` and the four synthetic headline figures across all 10 predeclared seeds from scratch:

```bash
python src/experiments.py --out results/full_synthetic_report.json
```

## License / Citation

Released under the [MIT License](LICENSE). This research codebase is provided as an open-source reference implementation of the Bayesian Sequential Decision Framework:

```bibtex
@software{bhattacharya2026bayesian,
  author       = {Bhattacharya, Ayushi},
  title        = {Bayesian Sequential Decision Framework},
  year         = {2026},
  url          = {https://github.com/aiyu221b3/Bayesian-Decision-Framework},
  license      = {MIT}
}
```
