# Refactor TODO

## Preamble: Is This Worth It?
- [x] Think why trying exchanging predicting and steering wont make sense? Or maybe make sense?
- [x] ~~Replicate to Qwen R1 1.5B~~ - **ABANDONED**
    - **Reason:** MLP SAE only, no residual stream SAEs available
    - Our methodology requires residual stream SAEs (like GemmaScope, LlamaScope)
    - Findings: 28 MLP layers available (layers.0.mlp to layers.27.mlp) but not compatible with current approach
- [x] What many latents did they steer or F1 with? The inspiration paper → **Only 1** (same as us)
- [x] Is refactoring worth it? → **YES, COMMITTED**
    - Worth learning better code architecture and design
    - Follow the phase-based approach outlined below
- [x] Plan complete. Ready to implement.
- [x] I want to write code better and have better design foresight on what I'm about to do. Do this while on learning mode I guess. Also have learning_notes.md while doing this.
- [x] Should I use einops and einsum? → **YES, selectively** (see Step 4.2)
    - Use `einops.rearrange` for complex reshapes (self-documenting shapes)
    - Keep simple ops as-is (`@` matmul, `.squeeze()`, `.T`)
    - Already using `einops.einsum` in weight_utils.py - good practice from ARENA

---

## Step 0: Decisions (Must Decide First)

These decisions affect how you approach everything else.

- [x] Is there a better way to handle my data? → **YES, Parquet + `phase_output.json` pattern**

    **Decision:** Use parquet for all data, JSON only for small metadata files.

    **Rationale:** I don't manually `cat` files to debug. I use `df.head()` in notebooks or let Claude Code inspect via bash. JSON "human readability" doesn't matter for my workflow.

    **Convention (applies to ALL phases):**
    ```
    data/phase{N}/
    ├── phase_output.json         # Small metadata: points to files, records config used
    ├── selected_features.parquet # Actual data (not "top_20" - number may change)
    └── results.parquet           # Main results
    ```

    **`phase_output.json` structure:**
    ```json
    {
      "phase": "2.5",
      "created": "2024-12-10T12:00:00",
      "config": {
        "n_features": 10,
        "layers": [19, 20],
        "pile_threshold": 0.02
      },
      "files": {
        "selected_features": "selected_features.parquet",
        "layer_analysis": "layer_analysis.parquet"
      },
      "selected_correct_feature": {"layer": 19, "feature_idx": 12345, "t_stat": 8.2},
      "selected_incorrect_feature": {"layer": 20, "feature_idx": 67890, "t_stat": -7.5}
    }
    ```

    **Benefits:**
    - Can manually trace which latent is being tested
    - Auto-discovery reads `phase_output.json` instead of globbing filenames
    - Filename doesn't lie (no "top_20" when it's actually top-10)
    - Parquet is smaller and faster than JSON for data

- [x] Do we implement tests? → **YES, but minimal (~5-10 tests)**

    **Decision:** Selective testing only. Focus on critical utility functions, skip GPU-dependent code.

    **Worth testing:**
    - `extract_code()` - Critical, tricky edge cases
    - `calculate_correction_rate()` / `calculate_corruption_rate()`
    - Metric calculations (AUROC, F1)
    - `phase_output.json` parsing (after refactor)
    - `evaluate_code()` sandbox

    **Not worth testing:**
    - Phase runners (too integrated, need GPU)
    - Model loading/generation
    - SAE encoding/decoding
    - Visualization code

    **Rationale:** Research code changes frequently. Real "test" = experiment produces valid results. Time better spent on experiments than comprehensive test coverage.
- [x] Is there other codebase architecture I should consider? → **Current phase-based structure is fine**

    **Compared with:** [sae_entities](https://github.com/javiferran/sae_entities) (Ferrando et al.)
    - They use functional organization (`dataset/`, `mech_interp/`, `utils/`)
    - We use phase-based organization (`phase0_*`, `phase1_*`, ...)

    **Considered but rejected:**
    - Moving all phases to `phases/` subfolder - too many import changes, marginal benefit

    **Keep doing:**
    - Phase-based structure (justified by compute constraints)
    - Centralized `run.py` entry point
    - Auto-discovery between phases

    **Cleanup needed (already in Step 1 & 3):**
    - Merge `common/` and `common_simplified/`
    - Add descriptions to unnamed phase folders

- [x] Why is the inspiration code so few? While mine is so long? → **Compute constraints justify the extra infrastructure**

    sae_entities (~11K lines) vs pva_sae (~30K lines) difference comes from:
    - Checkpointing every 50 records
    - `--start`/`--end` flags for testing
    - Auto-discovery between phases
    - SSH disconnect resilience
    - Activation caching (can't recompute)

    **This is not over-engineering - it's compute-constraint engineering.**

---

## Step 1: Cleanup (No Dependencies, Enables Everything Else)

Quick wins that make the codebase easier to work with.

- [ ] Delete not needed md or files anymore.
- [ ] Add better folder names especially the folders with no names just phase number.
    - [ ] `phase3_6/` → `phase3_6_hyperparameter_tuning/`
    - [ ] `phase3_8/` → `phase3_8_auroc_f1_evaluation/`
    - [ ] `phase7_12/` → `phase7_12_instruct_evaluation/`
    - [ ] Check for other unnamed folders
- [ ] Improve logging by a lot. Right now it's almost useless as you don't know where it is going and it's mixed up and sometimes it's working, sometimes not.
- [ ] Delete unused data files:
    - [ ] `data/phase2_5/layer_*_features.json` (25 files × 5.2MB = 130MB) - never used
    - [ ] `data/phase2_5/sae_analysis_results.json` - legacy, possibly incorrect

---

## Step 2: Core Infrastructure (Foundation for Later Work)

Fix the plumbing before building on top.

- [ ] Fix why there are multiple things needed to add when adding a new phase in run.py multiple times, config.py
- [ ] Make my data be in HuggingFace not in folders! IMPORTANT. Major improvement.
- [ ] **Multi-model prerequisites** (enables LLAMA + HumanEval experiments)
    - [ ] Ensure `phase_output.json` pattern works for LLAMA and HumanEval
    - [ ] Verify model-aware output paths: `data/phase2_5_llama/`, `data/phase2_5_humaneval/`
    - [ ] Test that config.py MODEL_CONFIGS correctly switches SAE repos

---

## Step 3: Common Module Refactor (Sequential Chain)

Do these in order - each step depends on the previous.

- [ ] Merge common and common_simplified
- [ ] Add other common/reused functions in common
    - [ ] Not only the things I already use that is just located in other phase files but also notice the other repeated functions throughout most of the phases. Or is this even a good decision because sometimes it may constrain us. Flexibility is also a trait we want in some instances.
- [ ] Have better categorization for common
- [ ] Make/create a checkpointing function as a wrapper something so I don't need to reimplement it every phase? Is this possible? What is the design?
- [ ] **Model-agnostic abstractions** (supports LLAMA + HumanEval)
    - [ ] Verify `common/sae_loader.py` handles both GemmaScope (JumpReLU) and LlamaScope (TopK)
    - [ ] Ensure `apply_index_range_filter()` works for HumanEval (164 tasks) not just MBPP (974)
    - [ ] Abstract prompt building for different datasets (MBPP vs HumanEval format differences)

---

## Step 4: Code Quality (Depends on Step 3)

Polish the code after the structure is stable.

### 4.1 List Comprehension Opportunities (21 findings)

Convert verbose loops to Pythonic one-liners. **High-impact examples:**

- [ ] **sae_analyzer.py:299-308** - Dict counting loop → `Counter()`
  ```python
  # BEFORE: 8 lines with manual dict tracking
  correct_layer_counts = {}
  for feat in top_correct:
      layer = feat['layer']
      correct_layer_counts[layer] = correct_layer_counts.get(layer, 0) + 1

  # AFTER: 2 lines with Counter
  from collections import Counter
  correct_layer_counts = Counter(feat['layer'] for feat in top_correct)
  ```

- [ ] **sae_analyzer.py:221-238** - List of dicts loop → list comprehension
  ```python
  # BEFORE: Loop with append
  features_correct = []
  for i in range(num_features):
      features_correct.append({'feature_idx': i, 'separation_score': scores['s_correct'][i].item(), ...})

  # AFTER: List comprehension
  features_correct = [{'feature_idx': i, 'separation_score': scores['s_correct'][i].item(), ...} for i in range(num_features)]
  ```

- [ ] **sae_analyzer.py:275-284** - Nested loop with copy → dict unpacking
  ```python
  # BEFORE: 8 lines with .copy() and assignment
  for layer_idx, layer_results in all_results.items():
      for feature in layer_results['features']['correct']:
          feature_with_layer = feature.copy()
          feature_with_layer['layer'] = layer_idx
          all_features_correct.append(feature_with_layer)

  # AFTER: Nested comprehension with dict unpacking
  all_features_correct = [{**feature, 'layer': layer_idx} for layer_idx, layer_results in all_results.items() for feature in layer_results['features']['correct']]
  ```

- [ ] **steering_coefficient_selector.py:562-569** - Conditional append → ternary comprehension
  ```python
  # BEFORE: 6 lines
  length_ratios = []
  for r in results:
      if len(r['baseline_code']) > 0:
          length_ratios.append(len(r['steered_code']) / len(r['baseline_code']))
      else:
          length_ratios.append(1.0)

  # AFTER: 1 line
  length_ratios = [len(r['steered_code']) / len(r['baseline_code']) if len(r['baseline_code']) > 0 else 1.0 for r in results]
  ```

- [ ] **auroc_f1_evaluator.py:97-103** - F1 threshold loop → comprehension
  ```python
  # BEFORE
  f1_scores = []
  for threshold in thresholds:
      y_pred = (scores >= threshold).astype(int)
      f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

  # AFTER
  f1_scores = [f1_score(y_true, (scores >= threshold).astype(int), zero_division=0) for threshold in thresholds]
  ```

- [ ] **steering_coefficient_selector.py:736-746** - Nested ternary → dict lookup
  ```python
  # BEFORE: Multiple ternary operators
  'layer': self.best_correct_feature['layer'] if steering_type == 'correct' else self.best_incorrect_feature['layer'],
  'feature_index': self.best_correct_feature['feature_idx'] if steering_type == 'correct' else self.best_incorrect_feature['feature_idx'],

  # AFTER: Dict lookup pattern
  feature_info = {'correct': self.best_correct_feature, 'incorrect': self.best_incorrect_feature}
  best_feature = feature_info[steering_type]
  # Then use: 'layer': best_feature['layer'], 'feature_index': best_feature['feature_idx']
  ```

- [ ] **auroc_f1_evaluator.py:669-686** - Summary lines → helper function + unpacking
  ```python
  # BEFORE: 16 lines of repeated f-string formatting
  # AFTER: Extract format_feature_summary() helper, use list unpacking (*)
  ```

**Other comprehension opportunities** (lower priority):
- [ ] sae_analyzer.py:378-401 - Nested filtering with pile threshold
- [ ] sae_analyzer.py:490-498 - Dict iteration for JSON writing
- [ ] auroc_f1_evaluator.py:246-253 - Bar labeling with zip + helper

---

### 4.2 Modern Python Practices

#### Type Hints (~50 files need updates)
- [ ] Replace old-style imports: `from typing import Dict, List, Tuple` → use `dict`, `list`, `tuple` (Python 3.9+)
- [ ] Fix partial type hints (some params typed, others not)
- [ ] Keep `Optional`, `Union`, `Callable` from typing (still needed)

#### Magic Numbers → Named Constants (~20-30 instances)
- [ ] **phase1_simplified/runner.py:169** - `if generation_time > 60` → `GENERATION_TIME_WARNING_THRESHOLD = 60`
- [ ] **phase1_simplified/runner.py:176** - `if len(generated_code) > 3000` → `CODE_LENGTH_WARNING_THRESHOLD = 3000`
- [ ] **phase1_simplified/runner.py:40** - `self.memory_warning_threshold = 85` → `MEMORY_WARNING_PERCENT = 85`
- [ ] **phase1_simplified/runner.py:39** - `self.checkpoint_frequency = 50` → `CHECKPOINT_FREQUENCY = 50`
- [ ] **config.py:174** - `sae_latent_threshold: float = 0.02` → add docstring explaining why 0.02

#### einops for Tensor Operations (4-5 instances)

Use `einops.rearrange` for complex reshapes - makes tensor shapes self-documenting.

- [ ] **steering_metrics.py:241** - Double unsqueeze → einops rearrange
  ```python
  # BEFORE: Shape transformation not obvious
  steering = sae_decoder_direction.unsqueeze(0).unsqueeze(0) * coefficient
  # [d_model] → [1, 1, d_model] - have to trace through mentally

  # AFTER: Self-documenting shape
  from einops import rearrange
  steering = rearrange(sae_decoder_direction, 'd -> 1 1 d') * coefficient
  # Shape transformation is explicit in the string
  ```

- [ ] **threshold_optimizer.py:490** - Same pattern
  ```python
  steering = rearrange(decoder_direction, 'd -> 1 1 d') * self.steering_coefficient
  ```

- [ ] **selective_steering_analyzer.py:394** - Same pattern
  ```python
  steering = rearrange(decoder_direction, 'd -> 1 1 d') * self.config.phase4_8_correct_coefficient
  ```

- [ ] Keep existing `einops.einsum` in **weight_utils.py:40** - already good

**When NOT to use einops** (keep simple):
- Simple matmul: `x @ self.W_enc` - `@` operator is clearer
- Basic squeeze: `activation.squeeze(0)` - obvious enough
- Transpose for loading: `weights['encoder.weight'].T` - standard pattern

**Add shape comments** where einops isn't used:
```python
# Shape: [batch, seq_len, d_model]
residual = input[0]
```

---

### 4.3 Variable Naming Consistency

#### Single-Letter Variables (outside comprehensions)
- [ ] **golden_section_refiner.py:772-773** - `a = bounds['lower']` → `lower_bound = bounds['lower']`
- [ ] **temperature_trends_visualizer.py:109** - `x = np.arange(...)` → `temperature_indices = np.arange(...)`

#### Inconsistent Terminology
- [ ] Standardize: `test_passed` vs `baseline_passed` vs `steered_passed` vs `orthogonalized_passed`
  - Document the naming convention in CLAUDE.md or a style guide
  - `test_passed` = original test result
  - `baseline_passed` = generation without steering
  - `steered_passed` = generation with steering
  - `orthogonalized_passed` = generation with weight orthogonalization

---

### 4.4 Function Structure (Karpathy Style)

#### Long Functions to Split (>50 lines, violate single responsibility)
- [ ] **steering_effect_analyzer.py:95-189** `_load_dependencies()` (95 lines) → split into:
  - `_load_phase_features()`
  - `_load_baseline_data()`
  - `_load_sae_models()`

- [ ] **steering_coefficient_selector.py:76-150** `_load_dependencies()` (75 lines) → similar split

- [ ] **helpers.py:85-147** `extract_code()` (63 lines) → split into:
  - `_extract_code_exact_match()`
  - `_extract_code_by_marker()`
  - `_extract_code_by_last_assert()`
  - `_extract_code_fallback()`

#### Add Early Returns (reduce nesting)
- [ ] **steering_metrics.py:19-80** `calculate_correction_rate()` - deeply nested if/elif
  ```python
  # BEFORE: 3-4 levels of nesting
  if isinstance(results, pd.DataFrame):
      if results.empty:
          return 0.0
      if 'steered_passed' in results.columns:
          ...

  # AFTER: Early returns, flat structure
  if isinstance(results, pd.DataFrame):
      return _correction_rate_from_dataframe(results)
  if isinstance(results, list):
      return _correction_rate_from_list(results)
  raise TypeError(...)
  ```

- [ ] **helpers.py:85-147** `extract_code()` - 4 levels deep, use guard clauses

#### Complex One-Liners to Split
- [ ] **instruct_steering_analyzer.py:633-635** - 200+ char logger.info line → split into multiple lines

---

### 4.5 Repeated Patterns to Abstract (feeds into Step 3)

These findings inform what should go in `common/`:

| Pattern | Files Affected | Suggested Abstraction |
|---------|---------------|----------------------|
| Checkpoint/resume logic | 14+ files | `CheckpointManager` class |
| Start/end index filtering | 19+ files | `apply_index_range_filter()` util |
| Memory management | 30+ files | `MemoryManager` class |
| Dataset-aware path construction | 20+ files | `get_dataset_aware_output_dir()` |
| SAE loading duplication | 8 files | Remove duplicate, use `common.sae_loader` |

**Quick win for Step 3:**
```python
# common/utils.py - add this function
def apply_index_range_filter(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply --start/--end filtering. Used by 19+ phases."""
    start_idx = getattr(config, 'dataset_start_idx', None) or 0
    end_idx = getattr(config, 'dataset_end_idx', None)

    if end_idx is None:
        end_idx = len(df)
    else:
        end_idx = min(end_idx + 1, len(df))  # inclusive

    if start_idx > 0 or end_idx < len(df):
        logger.info(f"Filtering: rows {start_idx}-{end_idx-1}")
        return df.iloc[start_idx:end_idx].copy()
    return df
```

---

## Step 5: Polish & Extras (After Core Refactoring)

Nice-to-haves once the foundation is solid.

- [ ] Improve notebooks. Remove unnecessary cells. Also do list comprehensions.
    - [ ] Understand matplotlib and pandas logic or how it works.
- [ ] Fix the figure generation code. Currently it looks soooo messy.
    - [ ] Make all figures correction green, corruption red, and pick a color for preservation.
- [ ] Add here the ICML LaTeX.

### 5.1 ICML Visualizations (moved from ICML tasks)

These visualizations address reviewer feedback. Do after refactoring is stable.

- [ ] **Top-10 features table** (Reviewers 7JAK, jwL5)
    - Create table showing top-10 features per direction
    - Columns: layer, feature_idx, separation score, t-statistic, AUROC

- [ ] **Feature-Selection Landscape scatter plot** (Reviewers 7JAK, jwL5)
    - X-axis: separation score, Y-axis: t-statistic
    - Show all features as dots, highlight chosen features as statistical outliers (>3σ)

- [ ] **Layer-wise visualization** (adapted from Ferrando et al. 2024, Figure 2)

    **Goal:** Show layerwise evolution of top-4 latents for all 4 directions.

    **Plot specification:**
    ```
    2x2 grid of subplots:

    ┌─────────────────────────────┬─────────────────────────────┐
    │  Correct-predicting         │  Incorrect-predicting       │
    │  (t-statistic)              │  (t-statistic)              │
    ├─────────────────────────────┼─────────────────────────────┤
    │  Correct-steering           │  Incorrect-steering         │
    │  (separation score)         │  (separation score)         │
    └─────────────────────────────┴─────────────────────────────┘

    Each subplot:
    - X-axis: Layer index (1-25)
    - Y-axis: t-statistic (predicting) or separation score (steering)
    - 4 lines: Top-4 latents per layer, colored by rank
    - Error bars: Max/min scores across the top-4
    - Red line (MaxMin): Minimum score of the best latent per layer
    ```

    **Data source:** Recompute from source (Phase 2.5 SAE analysis)

    **Reference:** Ferrando et al. 2024 Figure 2

- [ ] **Steering coefficient search plots** (Reviewer 7JAK)
    - Show coefficient search process for appendix
    - X-axis: coefficient value, Y-axis: correction/corruption rate

---

## Independent (Anytime)

Can be done in parallel with any phase.

- [ ] Consider other improvements to Claude Code like skills.md or hooks to improve my workflow.
    - [ ] Install Claude Code marketplace.

---

## ICML Submission Tasks (Based on ICLR Reviewer Feedback)

Address reviewer concerns with minimal compute. **Run these AFTER refactoring phases are complete.**

**Note:** Visualization tasks moved to Step 5.1 for smoother refactoring flow.

- [x] **Selective steering implementation** (Reviewers RXZd, vRko) - DONE
    - Conclusion: Selective steering in current form still not advisable. Better strategy: generate without steering first, only apply steering on retry if code is detected as wrong.

- [ ] **LLAMA + HumanEval experiments** (All reviewers) - RUN AFTER REFACTORING
    - [ ] Run all phases on `meta-llama/Llama-3.1-8B` with `llama_scope_lxr_8x`
    - [ ] Run all phases on `meta-llama/Llama-3.1-8B-Instruct`
    - [ ] Perform Mechanistic Analysis with HumanEval
    - **Prerequisites:** Step 2 multi-model support, Step 3 model-agnostic abstractions
    - **SAE Verified**: `fnlp/Llama-Scope` 32K (8x expansion) matches Neuronpedia's `llamascope-res-32k`
    - Addresses "single model, single benchmark" criticism

- [ ] **Feature threshold sensitivity analysis** (Reviewer RXZd)
    - [ ] Test sensitivity to the >2% activation threshold on pile-10k
    - [ ] Report how many features get filtered out in top 25

- [x] **Visualizations** → **Moved to Step 5.1**
    - Top-10 features table
    - Feature-Selection Landscape scatter plot
    - Layer-wise visualization (Ferrando et al. Figure 2 style)
    - Steering coefficient search plots

- [x] **Language-agnostic directions** (Reviewers vRko, jwL5) - SKIP
    - **Decision:** Not worth the effort. Reviewers said "ideally" not "required"
    - Would require HumanEval-X, uncertain SAE behavior on other languages
    - With MBPP + HumanEval + LLAMA, we have 3 evaluation points already
    - Frame as future work: "Investigating language-agnostic directions across programming languages is an important direction for future work."

- [x] **Difficulty-agnostic directions** (Reviewer 7JAK) - SKIP
    - **Decision:** Current difficulty stratification (cyclomatic complexity bucketing) is flawed
    - All MBPP/HumanEval problems are beginner-level - artificial bucketing not meaningful
    - Model can't solve harder benchmarks (APPS, CodeContests) so can't study correctness on them
    - Frame honestly: "Our study focuses on entry-level programming tasks where the model achieves non-trivial performance (~30% pass rate). Generalization to more complex programming challenges is limited by current model capabilities."

- [ ] **CoT faithfulness experiment** - Statistical testing methodology
    - [ ] Figure out how to do statistical testing for correct/incorrect related directions (both predicting and steering)
    - [ ]  Its not important if initially correct or incorrect. Whats important is if it predict it will generate incorrect code does the model say it?
    - **Clarification:** No need to test swapping predicting and steering latents - current setup makes sense:
        - t-statistic → for predicting directions
        - separation score → for steering directions
