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
    - Merge `common/` and `common/`
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

- [x] Delete not needed md or files anymore.
    - Deleted: `docs/*_abandon.md`, `investigation_results/`, phase design docs
- [x] Add better folder names especially the folders with no names just phase number.
    - [x] `phase3_6/` → `phase3_6_hyperparameter_baseline/`
    - [x] `phase3_8/` → `phase3_8_auroc_f1_evaluation/`
    - [x] `phase7_12/` → `phase7_12_instruct_auroc_f1/`
    - [x] `phase1_simplified/` → `phase1_latent_selection_dataset/`
    - [x] `phase2_5_simplified/` → `phase2_5_separation_score_analysis/`
    - [x] `phase4_5_model_steering/` → `phase4_5_coefficient_grid_search/`
    - [x] `phase4_6_binary_refinement/` → `phase4_6_golden_section_refinement/`
    - [x] All folders now have descriptive names
- [x] Improve logging by a lot. Right now it's almost useless as you don't know where it is going and it's mixed up and sometimes it's working, sometimes not.
    - Created `tqdm_with_logging()` wrapper in `common/logging.py` for milestone logging (25%, 50%, 75%, 100%)
    - Converted 96 print() statements → logger.info() across 7 files
    - Converted all phase tqdm usages (17 files) to use tqdm_with_logging
    - Terminal shows live tqdm bar as usual, log files get milestone updates
- [x] Delete unused data files:
    - [x] `data/phase2_5/layer_*_features.json` (25 files × 5.2MB = 130MB) - never used
    - [x] `data/phase2_5/sae_analysis_results.json` - legacy, possibly incorrect

---

## Step 2: Core Infrastructure (Foundation for Later Work)

Fix the plumbing before building on top.

- [x] Fix why there are multiple things needed to add when adding a new phase in run.py multiple times, config.py
- [x] Is many if good practice or should we use swithc statements instead?
- [x] Can we now delete phase_runners.py?
- [x] Search for other legacy features that should be removed.
- [x] Find other design flaw similar to to above.
- [x] Is there a design improvement where we could make our code much better or shorter or fewer? or should we leave this for step 3? Or is there instance where it would not be scoped in by phase 3 especially its 2nd bullet?
    - [x] Are you sure there is not more phases? I think most phases is consumed by another phase? There are only few exceptions. 
- [x] In the previous commits we did in these refactoring saga is there left backward compatibility that should be removed? Feel free to search or git history.
- [x] Adapting new consistency or standard for visualization. Like make separate the visualization from the main phase? or put it in notebook? For context I hate that we need to rerun the whole long phases just to change the visualization/table/figure. Help me think of a solution. Or should we do this in step 3 instead?
- [x] Make my data be in HuggingFace not in folders! IMPORTANT. Major improvement.
    - Created `scripts/upload_to_hf.py` for one-time uploads
    - Data at: https://huggingface.co/datasets/kriztahimic/pva-sae-data
    - Added `/hf-upload` slash command
    - Local workflow unchanged (fast), HF for backup/sharing
- [ ] ~~Should we consider designing here to run tests in parallel at for each model?~~ → **Deferred to Step 6**
- [x] **Multi-model prerequisites** → **Complete!**
    - [x] `common/sae_loader.py` exists with universal `load_sae()` and `load_sae_for_config()` functions
    - [x] `MODEL_CONFIGS` in config.py with Gemma + LLAMA settings
    - [x] `get_phase_output_dir()` adds `_llama`/`_humaneval` suffixes automatically
    - [x] `phase_output.json` code added to 25 phase runners (creates manifest on next run)
    - [x] **All 16 phase files updated to use `load_sae_for_config(config, layer_idx, device)`**
    - [x] Removed duplicate `load_gemma_scope_sae()` and `JumpReLUSAE` from phase2_5
    - Note: LLAMA only has `phase1_0_llama` data; run phases 2.5→8.3 when ready for LLAMA experiments
---

## Step 3: Common Module Refactor (Sequential Chain)

Do these in order - each step depends on the previous.

- [x] Merge common and common_simplified → **Done!** All 4 modules moved to common/, 30+ imports updated
- [x] Add other common/reused functions in common
    - [x] Not only the things I already use that is just located in other phase files but also notice the other repeated functions throughout most of the phases. Or is this even a good decision because sometimes it may constrain us. Flexibility is also a trait we want in some instances.
- [x] Have better categorization for common or file separation or groupings of the functions.
- [x] Make/create a checkpointing function as a wrapper something so I don't need to reimplement it every phase? Is this possible? What is the design?
    - [x] Also earlier you recommend CheckpointingManager in common to handle save, load etc. Is that still part of the plan? Is it still a good idea?
- [x] **Model-agnostic abstractions** (supports LLAMA + HumanEval)
    - [x] `common/sae_loader.py` handles both GemmaScope (JumpReLU) and LlamaScope (TopK) ✓
    - [x] Ensure `apply_index_range_filter()` works for HumanEval (164 tasks) not just MBPP (974)
    - [x] Abstract prompt building for different datasets (MBPP vs HumanEval format differences)
- [x] Given what we have accomplished so far in the past two days. Feel free to use git history and read refactor.md. Figure out what is outdated in README.md and CLAUDE.md and update them.
    - Updated README.md: Added Gemma-9B support, fixed duplicate common/ entry, updated roadmap
    - Updated CLAUDE.md: Fixed phase_discovery.py references, updated model options, simplified Model/Dataset-Aware Paths section
---

## Step 4: Code Quality (Depends on Step 3)
- [x] Remove hardcoded path discovery patterns (16 instances across 11 files)
    - **Issue Found:** `discover_latest_phase_output()` didn't accept `config` parameter
    - **Fix Applied:** Added `config` parameter to `discover_latest_phase_output()` in `phase_discovery.py`
    - **Files Updated (Phase 2.5 discovery):**
        - `phase4_5_coefficient_grid_search/steering_coefficient_selector.py`
        - `phase5_3_weight_orthogonalization/weight_orthogonalizer.py`
        - `phase6_3_attention_analysis/attention_analyzer.py`
        - `phase2_15_layerwise_visualization/layerwise_visualizer.py`
        - `phase7_6_instruct_steering/instruct_steering_analyzer.py`
        - `phase8_2_threshold_optimizer/threshold_optimizer.py`
        - `phase4_6_golden_section_refinement/golden_section_refiner.py`
        - `phase8_3_selective_steering/selective_steering_analyzer.py`
        - `phase4_8_steering_analysis/steering_effect_analyzer.py`
    - **Files Updated (bandaid phase_dir construction removed):**
        - `phase4_14_statistical_significance/significance_tester.py` (3 fixes: 3.5, 4.8, 4.12)
        - `phase5_9_orthogonalization_significance/orthogonalization_significance_tester.py` (3 fixes: 3.5, 5.3, 5.6)
        - `phase4_12_zero_disc_steering/zero_disc_steering_generator.py` (1 fix: 3.5)
        - `phase5_6_zero_disc_orthogonalization/zero_disc_weight_orthogonalizer.py` (1 fix: 3.5)
        - `phase3_10_temperature_auroc_f1/temperature_evaluator.py` (1 fix: 3.5)
        - `phase7_6_instruct_steering/instruct_steering_analyzer.py` (1 fix: 7.3)
- [x] Examine this kind of code: `Fixed bfloat16→float32 conversion: .cpu().float().numpy() instead of .cpu().numpy()` What should I do? There's often error here. I think there are other forms of this.
    - **COMPLETED**: Migrated all activation storage to safetensors format
    - Created `common/tensor_utils.py` with centralized utilities:
      - `save_activation()` / `load_activation()` - single tensor
      - `save_activations()` / `load_activations()` - multi-layer dict
      - `save_attention()` / `load_attention()` - attention with metadata (.safetensors + .json)
      - `to_numpy()` - explicit float32 conversion only when needed
    - Updated 16 files to use safetensors, preserving bfloat16 throughout
    - Attention uses companion .json file for metadata (strings, dicts)

- [x] How to get steering coefficient? Autodiscovery or config? What is better for the script?
    - **COMPLETED**: Auto-discovery via manifest system
    - Added `discover_steering_coefficients()` to `common/phase_discovery.py`
    - Uses `get_phase_output_file("4.6", "refined_coefficients", config)` - manifest-based
    - Removed hardcoded values from config.py
    - Updated Phase 4.8, 7.6, 8.3 to use auto-discovered coefficients
    - Fixed bugs in Phase 4.5 and 4.6 (undefined instance variables for manifest writing)

Polish the code after the structure is stable.

### 4.1 List Comprehension Opportunities (21 findings)

Convert verbose loops to Pythonic one-liners. **High-impact examples:**

- [x] **sae_analyzer.py** - Dict counting loop → `Counter()`
  - Converted manual dict.get() + 1 to `dict(Counter(feat['layer'] for feat in top_correct))`

- [x] **sae_analyzer.py** - List of dicts loop → list comprehension
  - Converted loop with append to list comprehension for features_correct/features_incorrect

- [x] **sae_analyzer.py** - Nested loop with copy → dict unpacking
  - Converted feature.copy() + assignment to `{**feature, 'layer': layer_idx}`

- [x] **t_statistic_selector.py** - Same 3 patterns as sae_analyzer.py (identical code)
  - Counter, list comprehension, dict unpacking

- [x] **retry_utils.py** - Error counting loop → Counter with helper function
  - Converted manual error type extraction + counting to Counter

- [x] **attention_analyzer.py** - Means/stds loop → list comprehension
  - Converted conditional append loop to ternary list comprehension

- [x] **steering_coefficient_selector.py** - Conditional append → ternary comprehension
  - Converted length_ratios loop to one-liner list comprehension

- [x] **auroc_f1_evaluator.py** - F1 threshold loop → comprehension
  - Converted f1_scores loop to list comprehension

- [x] **steering_coefficient_selector.py** - Nested ternary → dict lookup
  - Added `feature_lookup` and `coeff_lookup` dicts, then `best_feature = feature_lookup[steering_type]`

- [x] **auroc_f1_evaluator.py** - Summary lines → helper function
  - Extracted `log_feature_summary(name, metrics)` helper function

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

- [ ] rename the latents, directions, features etc. Use one name. Find other variables that have called different names.


#### Single-Letter Variables (outside comprehensions)
- [ ] **golden_section_refiner.py:772-773** - `a = bounds['lower']` → `lower_bound = bounds['lower']`
- [ ] **temperature_trends_visualizer.py:109** - `x = np.arange(...)` → `temperature_indices = np.arange(...)`

#### Inconsistent Terminology

**NOTE (from Step 3 refactor):** Result dict naming convention was considered but deferred here.
Two options exist - consult user before deciding:
- Option A (current): Keep `*_passed` suffix (`test_passed`, `baseline_passed`, `steered_passed`, `orthogonalized_passed`)
- Option B: Use `*_correct` suffix (`initial_correct`, `final_correct`) with semantic focus on outcome not method

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

- [ ] Update the docstrings/commetns.
- [ ] Improve notebooks. Remove unnecessary cells. Also do list comprehensions.
    - [ ] Understand matplotlib and pandas logic or how it works.
- [ ] Fix the figure generation code. Currently it looks soooo messy.
    - [ ] Make all figures correction green, corruption red, and pick a color for preservation.
- [ ] Add here the ICML LaTeX.
- [ ] Rename to code-correctness-sae( the folder, github repo, huggingface etc.)

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
    - [ ] Fix this hardcoded  phase4_5_incorrect_coefficients: List[float] = field(default_factory=lambda: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]). Make it do 1-100 in increments of 10. If still on corruption rate then do 200-1000 in increments of 1000


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

---

## Step 6: Multi-GPU Parallel Execution (After Experiments Work)
- [ ] Test all phase one by one first if it is all running.
- [x] **Gemma-2-9B Support Added:**
    - Added `GEMMA_9B_SPARSITY` dict (42 layers) to `common/config.py`
    - Added `google/gemma-2-9b` and `google/gemma-2-9b-it` to `MODEL_CONFIGS`
    - Updated `load_gemma_scope_sae()` to handle 9B via `model_name` parameter
    - Added `gemma9b` suffix in `get_phase_output_dir()` for 9B models
    - **Usage:** Set `model_name: str = "google/gemma-2-9b"` in config.py

Run experiments across 4 GPUs efficiently. Do this after multi-model support is verified.

### Problem
- 4 GPUs available, but experiments run sequentially
- Different workloads: LLAMA takes longer than Gemma, not all phases run HumanEval
- Static GPU assignment wastes resources (fast jobs finish, GPU sits idle)

### Solution: Job Queue Pattern

- [ ] Add `--model` and `--dataset` CLI flags to `run.py`
- [ ] Create `scripts/job_queue.py` with task queue pattern
- [ ] Workers (one per GPU) grab jobs from queue as they finish
- [ ] Support for job dependencies (Phase 2.5 needs Phase 1)

### Design

```python
# scripts/job_queue.py
JOBS = [
    ("1", "google/gemma-2-2b", "mbpp"),
    ("1", "google/gemma-2-2b", "humaneval"),
    ("1", "meta-llama/Llama-3.1-8B", "mbpp"),
    ("2.5", "google/gemma-2-2b", "mbpp"),  # Runs when GPU free
    # ...
]

# 4 worker threads, each assigned a GPU
# Workers grab next job from queue when current job finishes
```

### Alternative: Task Spooler (Zero Code)
```bash
TS_SLOTS=4
GPU=0 ts python3 run.py phase 1 --model gemma --dataset mbpp
GPU=1 ts python3 run.py phase 1 --model llama --dataset mbpp
# Queue auto-distributes to available GPUs
```

### Prerequisites
- Step 2: Multi-model prerequisites complete
- Step 3: Model-agnostic abstractions working
- Verified: All phases work with `--model` and `--dataset` flags

- [ ] Test run all 10 records.