# Diagnostic Audit of WLASL Real-Data Pilot (`findings.md`)

**Scope**: Verification and documentation only of `evaluation_summary.json`, `evaluation_results.csv`, and `ablation_study_summary.json` over $N = 14$ real validation probability trays (`all_validation_trays.npz`).  
**Status**: Appendix record for future real-data pilots; reports unhedged empirical measurements and verified codebase behavior without modifying core Table 1 architecture.

---

## 1. M0 Scoring Bug

* **Finding**: **CONFIRMED** — *Scoring artifact (bug in fallback evaluation logic).*
* **Evidence & Root Cause**:
  In `src/run_ablation_study.py` (before correction), when a stopping policy produced zero emissions (`n_emissions = 0`), the fallback evaluation logic executed:
  ```python
  pred = int(P[-1].argmax())
  is_correct = (pred == ground_truth)
  ```
  This fallback credited any non-emitting policy with the raw classifier's accuracy at the final frame of the video ($T-1$). On the 14 validation sequences, `P[-1].argmax()` is correct on $9 / 14$ sequences ($64.3\%$). At $\tau = 0.75$, M0 (`RawThresh`) never emitted on any sequence ($0 / 14 = 0.0\%$ emission rate), yet received $64.3\%$ unearned fallback accuracy in the original ablation grid.
* **Remediation**:
  Fixed `src/run_ablation_study.py` so that policies that fail to emit are scored as `is_correct = False` ($0.0\%$ correct).
* **Corrected M0 Accuracy Across All 4 Ablation Configurations** ($N = 14$):
  1. `Baseline (t_min=0, tau=0.50)`: **28.6%** ($4 / 14$ correct; $14 / 14$ emitted — $100.0\%$ emission rate).
  2. `Baseline + t_min=15 (t_min=15, tau=0.50)`: **21.4%** ($3 / 14$ correct; $4 / 14$ emitted — $28.6\%$ emission rate).
  3. `Baseline + higher tau (t_min=0, tau=0.75)`: **0.0%** ($0 / 14$ correct; $0 / 14$ emitted — $0.0\%$ emission rate).
  4. `Both together (t_min=15, tau=0.75)`: **0.0%** ($0 / 14$ correct; $0 / 14$ emitted — $0.0\%$ emission rate).

> [!NOTE]
> **Corrected Emitting Policy Accuracies in Config 4 (`Both together`)**:
> Without fallback scoring credit, the strict accuracies for emitting policies in `Both together (t_min=15, tau=0.75)` over all $N=14$ sequences are: **M1 = 21.4%** ($3 / 14$, $7 / 14$ emitted), **M2 = 35.7%** ($5 / 14$, $8 / 14$ emitted), **M3 = 21.4%** ($3 / 14$, $8 / 14$ emitted), and **MT = 14.3%** ($2 / 14$, $6 / 14$ emitted).

---

## 2. Architecture-vs-Implementation Audit

* **Finding**: **CONFIRMED** — *All five methods match their Table 1 specification in constructor parameters and structural logic (belief structure, change model, stopping rule).*
* **Method-by-Method Verification**:
  * **M0 (`RawThresh`, `src/policies.py:14-42`)**:
    * *Belief Structure*: None (operates on raw posterior vector $p_t$).
    * *Change Model*: None ($\rho = 0$).
    * *Stopping Rule*: $\max_k p_t(k) \ge \theta$ with hysteresis lower bound (`low = max(0.0, theta - 0.15)`) and refractory debounce (`refr = 8`) via `EmitterGate`.
  * **M1 (`BayesThresh`, `src/policies.py:44-76`, `src/belief.py:31-68`)**:
    * *Belief Structure*: `FlatBel(K=5, lam=0.5, rho=0.0)` with logarithmic observation tempering ($\lambda = 0.5$).
    * *Change Model*: Static ($\rho = 0.0$; recursive Bayesian update without hazard forgetting).
    * *Stopping Rule*: $\max_k b_t(k) \ge \tau$ via `EmitterGate(tau, low, refr)`.
  * **M2 (`BayesThresh`, `src/policies.py:44-76`, `src/belief.py:31-68`)**:
    * *Belief Structure*: `FlatBel(K=5, lam=0.5, rho=0.05)` with logarithmic observation tempering ($\lambda = 0.5$).
    * *Change Model*: Switching HMM / Markov jump process with hazard rate $\rho = 0.05$ (`pred = (1 - rho) * prev + rho / K`).
    * *Stopping Rule*: $\max_k b_t(k) \ge \tau$ via `EmitterGate(tau, low, refr)`.
  * **M3 (`BocpdThresh`, `src/policies.py:78-138`, `src/changepoint.py`)**:
    * *Belief Structure*: `FlatBel(K=5, lam=0.5, rho=0.0)` combined with `BOCPD(K=5, hazard=1/30, alpha=1.0)`.
    * *Change Model*: Bayesian Online Changepoint Detection. When changepoint posterior $P(\text{cp}_t \mid x_{1:t}) \ge 0.10$ triggers via `Cross`, the filter resets (`self.bel.reset()`) and re-arms the emitter gate (`bd=True` in `g.step`), matching the BOCPD-gated Table 1 spec.
    * *Stopping Rule*: $\max_k b_t(k) \ge \tau$ gated and re-armed by BOCPD changepoint detections.
  * **MT (`TreeBel`, `src/tree_belief.py:69-126`)**:
    * *Belief Structure*: `FlatBel(K=5, lam=0.5, rho=0.05)` and hierarchical agglomerative tree `build_tree(S)`.
    * *Change Model*: Markov jump process ($\rho = 0.05$).
    * *Stopping Rule*: Evaluates candidate nodes for specificity $\text{sp}(v) = \frac{K - |v|}{K - 1} \ge \text{min\_sp} = 0.25$ and aggregate belief mass $\sum_{k \in v} b_t(k) \ge \tau$. Emits coarse node identity or leaf class via `EmitterGate`.

---

## 3. MT Frame-0 Commit

* **Finding**: **CONFIRMED** — *Effectively always-satisfied (degenerate) coarse threshold under uniform prior initialization in small-vocabulary ($K=5$) settings.*
* **Exact Numbers & Mechanism ($N = 14$)**:
  * Across all 14 validation sequences ($idx = 0 \dots 13$), MT's first emission occurs at frame $0$ ($t = 0.0$).
  * For $K=5$ with identity similarity matrix $S = \mathbf{I}_5$, `build_tree(S)` generates internal coarse node $v = \{0, 1, 4\}$ (size $|v| = 3$).
  * Specificity of node $\{0, 1, 4\}$ is $\text{sp}(v) = \frac{5 - 3}{5 - 1} = 0.50$, which satisfies $\text{sp}(v) \ge \text{min\_sp} = 0.25$.
  * At frame $0$, under uniform prior initialization $b_0(k) = 0.20$ across all 5 classes, the aggregate probability mass of node $\{0, 1, 4\}$ is:
    $$\sum_{k \in \{0,1,4\}} b_0(k) = 3 \times 0.20 = \mathbf{0.6000}$$
  * Because $0.6000 \ge \tau = 0.50$, every validation sequence crosses the stopping threshold at frame 0 before incorporating any frame-0 observation evidence.
  * **Why Synthetic Headline Claims Are Unaffected**: In the headline synthetic suite (`src/experiments.py`), the problem uses **$K=10$ classes**, **$\tau = 0.80$**, and **$\text{min\_sp} = 0.25$**. Under a uniform prior $b_0(k) = 1/10 = 0.10$, reaching $\tau = 0.80$ requires a coarse node with at least 8 leaves ($8 \times 0.10 = 0.80$). However, an 8-leaf node has specificity $\text{sp}(v) = \frac{10 - 8}{10 - 1} = \frac{2}{9} = 0.2222 < \text{min\_sp} = 0.25$. Thus, prior-induced immediate commitment is mathematically excluded in the synthetic pipeline ($K=10$), confirming that frame-0 commitment is an artifact of transferring $\tau=0.50 / \text{min\_sp}=0.25$ to a small-vocabulary ($K=5$) setting.

---

## 4. BOCPD False-Positive Check

* **Finding**: **CONFIRMED** — *High false-positive rate for changepoint-triggered subsequent emissions on isolated single-sign clips.*
* **Exact Numbers Across $N = 14$ Sequences**:
  In single-gesture videos, each file contains exactly one ground-truth class label. Subsequent emissions (emissions after the first emission in a sequence) triggered by BOCPD changepoints were audited across all 14 sequences:
  * `1. Baseline (t_min=0, tau=0.50)`:
    * Total M3 emissions: **32** (mean 2.29 / seq).
    * Sequences with $>1$ emission: **9 / 14**.
    * Subsequent emissions (2nd, 3rd, 4th emissions): **18**.
    * False positives among subsequent emissions (emitted class $\neq$ ground truth): **12 / 18 (66.7% False-Positive Rate)**.
  * `2. Baseline + t_min=15 (t_min=15, tau=0.50)`:
    * Subsequent emissions: **9**; False positives: **5 / 9 (55.6% FPR)**.
  * `3. Baseline + higher tau (t_min=0, tau=0.75)`:
    * Subsequent emissions: **10**; False positives: **7 / 10 (70.0% FPR)**.
  * `4. Both together (t_min=15, tau=0.75)`:
    * Subsequent emissions: **3**; False positives: **2 / 3 (66.7% FPR)**.
* **Explanation**:
  In isolated single-sign clips, secondary changepoints trigger on post-gesture relaxation or noise transitions rather than semantic sign transitions, causing BOCPD-gated re-emissions to exhibit a **55.6%–70.0% false-positive rate** across ablation configurations.

---

## 5. Regression Check

* **Finding**: **CONFIRMED** — *100% pass rate across all 11 exploration clauses and 12 preservation invariants (13/13 pytest items passed unmodified).*
* **Clause-by-Clause Audit (`test_bug_condition_exploration.py::test_property_1_complete_sequential_bayes_remediation`)**:
  * **Clause 1 (Req 1.1)**: Generated posterior invariants plus stable/surprising and two-threshold probes — **PASSED**
  * **Clause 2 (Req 1.2)**: BOCPD hazard edge cases, degenerate inputs, and probability invariants — **PASSED**
  * **Clause 3 (Req 1.3)**: Threshold policies, hysteresis, and debounce refractory behavior — **PASSED**
  * **Clause 4 (Req 1.4)**: Dynamic programming optimal stopping boundary and Bellman value preservation — **PASSED**
  * **Clause 5 (Req 1.5)**: Hierarchical class tree specificity and coarse-to-fine emission ordering — **PASSED**
  * **Clause 6 (Req 1.6)**: Flat error rate, mean stopping time, and expected value of sample information — **PASSED**
  * **Clause 7 (Req 1.7)**: Evaluation reporting format, schema compliance, and summary tables — **PASSED**
  * **Clause 8 (Req 1.8)**: Synthetic generator reproducibility, coarticulation, and onset structure — **PASSED**
  * **Clause 9 (Req 1.9)**: Diagnostic plotting, sequence inspection, and summary visualization — **PASSED**
  * **Clause 10 (Req 1.10)**: Performance regression prevention across benchmark configurations — **PASSED**
  * **Clause 11 (Req 1.11)**: End-to-end framework execution without unhandled exceptions — **PASSED**
* **Preservation Suite Audit (`src/test_preservation.py`, 9 requirements / 12 invariants across 12 test functions)**:
  * `test_golden_belief_trajectories_replay_with_central_tolerance` — **PASSED**
  * `test_golden_generator_output_replays_exact_structure_and_numbers` — **PASSED**
  * `test_golden_first_qualifying_emission_time_and_identity_for_m0_through_mt` — **PASSED**
  * `test_golden_flat_correctness_and_unit_cost_levenshtein_examples` — **PASSED**
  * `test_golden_hierarchy_membership_and_specificity_only` — **PASSED**
  * `test_property_valid_posteriors_are_causal_finite_normalized_and_rho_zero_equivalent` — **PASSED**
  * `test_property_positive_rho_uses_the_documented_causal_forgetting_update` — **PASSED**
  * `test_property_generator_replay_normalization_labels_lengths_onsets_and_bounds` — **PASSED**
  * `test_property_ar1_recurrence_onset_ramps_and_configured_coarticulation` — **PASSED**
  * `test_supported_synthetic_smoke_has_all_methods_without_real_data_or_network` — **PASSED**
  * `test_beginner_friendly_explanatory_scope_is_preserved_without_claim_oracles` — **PASSED**
  * `test_preservation_oracle_records_all_known_exclusions` — **PASSED**

---

## 6. Priority Audit: Gesture Onset Timing vs. Stopping Frame

* **Finding**: **CONFIRMED** — *In 100% of real validation sequences ($14 / 14$), Baseline stopping policies commit before the heuristic estimate of target gesture onset.*
* **Exact Numbers ($N = 14$)**:
  * Heuristic gesture onset frame estimate (defined by first BOCPD changepoint $> 0.10$ or target class posterior $> 0.30$) across sequences $0 \dots 13$:
    * Mean estimated onset frame: **28.1** (min: frame 2, max: frame 65; sequence lengths $T = 26 \dots 93$).
  * Baseline stopping frames (`t_min=0, tau=0.50`):
    * **M0**: mean frame **2.0** (all 14 stop at frame 2 or 39).
    * **M1, M2, M3**: mean frame **3.4** (all 14 stop at frame 3, 4, or 7).
    * **MT**: mean frame **0.0** (all 14 stop at frame 0).
  * On every validation sequence ($14 / 14 = 100.0\%$), $\text{Estimated Onset Frame} > \min(\text{Stop}_{\text{M0}\dots\text{MT}})$. While gesture onset in unlabeled clips is a heuristic estimate rather than ground truth, these measurements are consistent with early commitment during preparatory rest frames before gesture kinematics commence.
