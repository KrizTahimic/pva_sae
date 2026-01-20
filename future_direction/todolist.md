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

    sae_entities (~11K lines) vs sae_code_correctness (~30K lines) difference comes from:
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
    - Data at: https://huggingface.co/datasets/kriztahimic/sae-code-correctness-data
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
- [x] ~~sae_analyzer.py:378-401 - Nested filtering with pile threshold~~ → **Redesigned as Phase 2.3**
    - Created `phase2_3_pile_frequencies/` to compute SAE frequencies once per layer
    - Created `common/pile_filter_utils.py` with `load_pile_frequencies()` and `apply_pile_filter()`
    - Removed ~85 lines of duplicated pile filtering code from Phase 2.5 and 2.10
    - Test run: 0/20 features filtered (expected - only 67 pile samples, need 10,000 for real filtering)
- [x] sae_analyzer.py:490-498 - Dict iteration for JSON writing → **Skipped** (low value)
- [x] auroc_f1_evaluator.py:246-253 - Bar labeling with zip + helper → **Skipped** (low value)

---

### 4.2 Modern Python Practices

#### Type Hints (~50 files need updates)
- [x] Replace old-style imports: `from typing import Dict, List, Tuple` → use `dict`, `list`, `tuple` (Python 3.9+)
- [x] Fix partial type hints (some params typed, others not)
- [x] Keep `Optional`, `Union`, `Callable` from typing (still needed)

#### Magic Numbers → Named Constants ✓ COMPLETED

Added to **common/config.py** (after line 17):
- [x] `CHECKPOINT_FREQUENCY_DEFAULT = 10`
- [x] `MEMORY_WARNING_PERCENT = 85`, `MEMORY_HIGH_PERCENT = 90`, `MEMORY_CRITICAL_PERCENT = 95`
- [x] `GENERATION_TIME_WARNING_SECONDS = 60`, `CODE_LENGTH_WARNING_CHARS = 3000`
- [x] `MIN_CORRECTION_EFFECT_PERCENT = 10`, `MIN_PRESERVATION_EFFECT_PERCENT = 50`
- [x] `STEERING_EFFECT_THRESHOLD_PERCENT = 90`
- [x] `PLOT_DPI = 300`

Updated 15 files to use these constants.

#### einops for Tensor Operations ✓ COMPLETED

Use `einops.rearrange` and `einops.reduce` for self-documenting tensor operations.

**Steering hooks - UPDATED (2025-12-17):**
- [x] **Changed from all-positions to last-position-only steering**
- [x] Removed `rearrange('d -> 1 1 d')` broadcasting pattern
- [x] Now uses: `residual[:, -1, :] = residual[:, -1, :] + steering`
- [x] Files updated: `steering_metrics.py`, `threshold_optimizer.py`, `selective_steering_analyzer.py`

**Continuous vs Prompt-Only Steering (Deviates from Ferrando 2024)**

- [x] MVP test: Continuous vs prompt-only comparison (LLaMA, 40 samples, probe direction)
- [ ] Production test: Run full SAE steering comparison on all models
- [ ] Validate finding holds with SAE latent directions (not just probe)

**MVP Results (2025-12-17):**

| Mode | Coefficient=1 | Correction Rate |
|------|---------------|-----------------|
| **Continuous** | 1 | **7.5%** (3/40) |
| Prompt-only | 1 | 0% (0/40) |

**Hypothesis**: For **longer generation tasks** (code generation ~50-200 tokens), continuous steering outperforms prompt-only. Ferrando et al. used prompt-only for **entity recognition** (short answers), where KV-cache persistence suffices. For code:
1. Correctness is NOT fully determined at prompt encoding time
2. Steering effect through KV-cache decays over longer generations
3. Each generated token benefits from active steering reinforcement

**Current Decision**: Production code updated to continuous last-position-only steering (pending full validation).
**Experiment Code**: `experiments/linear_probe_sanity_check/run_probe_steering.py` (keeps both modes)
**Documentation**: `future_direction/linear_probe_vs_sae_comparison.md`

**Reduction operations - `reduce('n f -> f', 'mean')`:**
- [x] **sae_analyzer.py:102-112** - Per-feature statistics (4 mean operations)
- [x] **pile_frequency_computer.py:95** - Per-feature activation frequency

**SAE encoding shape prep - `rearrange('d -> 1 d')`:**
- [x] **temperature_evaluator.py:182** - Ensure [1, d_model] for SAE
- [x] **dataset_utils.py:363** - Same pattern

**Already good (no change needed):**
- [x] **weight_utils.py:40** - Already uses `einops.einsum`

**When NOT to use einops** (keep simple):
- Simple matmul: `x @ self.W_enc` - `@` operator is clearer
- Basic squeeze: `activation.squeeze(0)` - obvious enough
- Transpose for loading: `weights['encoder.weight'].T` - standard pattern

- [x] einsum - **weight_utils.py:87** `torch.matmul` → `einops.einsum(..., '... d, d -> ...')` (only 1 opportunity; SAE `@` ops kept as-is)
---

### 4.3 Variable Naming Consistency

- [x] **SAE Terminology Standardization** (Completed 2025-12-13)

  Standardized all SAE-related variable names across the codebase:

  | Old Term | New Term | Type |
  |----------|----------|------|
  | `feature_idx` | `latent_idx` | `int` |
  | `decoder_direction` | `latent_direction` | `torch.Tensor [d_model]` |
  | `feature_activation` | `latent_activation` | `float` |
  | `sae_features` | `latent_activations` | `torch.Tensor [batch, n_latents]` |
  | `correct_feature_idx` | `correct_latent_idx` | `int` |
  | `incorrect_pred_feature` | `incorrect_pred_latent` | `int` |
  | `feature_freqs` | `latent_freqs` | `dict` |

  **Files updated:** 40+ files across common/, phase2_5/, phase2_10/, phase2_15/, phase3_x/, phase4_x/, phase5_x/, phase6_3/, phase7_x/, phase8_x/, docs/

  **JSON output files renamed:** `top_20_features.json` → `top_20_latents.json`

  **Documentation:** Added "SAE Terminology Standard" section to CLAUDE.md

  **Commits:** ea3ce134c, 994963ac7, 0069b1904, c533194da, 2f1981f4b

  **Verification:** grep confirms 0 remaining occurrences of old terminology


#### Single-Letter Variables (outside comprehensions)

**COMPLETED:** All single-letter variables outside comprehensions have been renamed.

Changes made (16 instances across 9 files):
- `a`, `b` → `lower_bound`, `upper_bound` (golden_section_refiner.py)
- `z`, `p` → `poly_coefficients`, `trend_line` (attention_analyzer.py)
- `k` → `sae_topk` (sae_loader.py)
- `x` → `bar_positions` (6 visualization files)

**Verification:** grep confirms 0 remaining single-letter variable assignments

#### Inconsistent Terminology

**COMPLETED (Commit 3):** Test outcome terminology has been standardized across the codebase.

Convention adopted:
- `baseline_passed` = Boolean: did baseline (unmodified) test pass? (Initial state, from Phase 1 generation)
- `steered_correct` = Boolean: is steered output correct? (After steering intervention)
- `orthogonalized_correct` = Boolean: is orthogonalized output correct? (After weight orthogonalization)

Terms removed:
- `test_passed` → `baseline_passed`
- `steered_passed` → `steered_correct`
- `orthogonalized_passed` → `orthogonalized_correct`
- `initial_passed`/`final_passed` → use explicit intervention names
- `is_correct` → `baseline_passed`

**Documentation:** Updated "Test Outcome Terminology Standard" section in CLAUDE.md

**Verification:** grep confirms 0 remaining occurrences of old terminology in Python files

---

### 4.4 Function Structure (Karpathy Style) ✅ COMPLETED

#### Long Functions to Split (>50 lines, violate single responsibility)
- [x] **steering_effect_analyzer.py** `_load_dependencies()` (92 lines) → split into:
  - `_load_pva_latents()` - Load Phase 2.5 latents
  - `_load_baseline_data()` - Load Phase 3.5 baseline
  - `_load_sae_models()` - Load SAEs and extract directions
  - `_load_steering_coefficients()` - Load Phase 4.6 coefficients

- [x] **steering_effect_analyzer.py** `_apply_steering()` (252 lines) → split into:
  - `_get_steering_params()` - Get latent direction and target layer
  - `_generate_steered_output()` - Core generation logic (was nested function)
  - `_finalize_steering_results()` - Aggregate results and merge DataFrames
  - `_apply_steering()` - Now just orchestration (~115 lines)

- [x] **steering_coefficient_selector.py** `_load_dependencies()` (102 lines) → split into:
  - `_load_pva_latents()` - Load Phase 2.5 latents
  - `_load_baseline_data()` - Load Phase 3.6 baseline and split by correctness
  - `_load_sae_models()` - Load SAEs and extract directions

- [x] **dataset_utils.py** `extract_code()` (65 lines) → **SKIPPED** (intentionally)
  - Analysis: This is a sequential fallback pipeline, not multiple responsibilities
  - Splitting would scatter related logic without improving readability
  - Decision: Function length isn't the only measure - cohesion matters more

- [x] Fix:        344 +                      activation_bf16 = activation.to(dtype=self.sae_l19.W_enc.dtype, device=self.device)
    - l_19 rename. Find other instances.
    - **DONE**: Renamed `sae_l19` → `predicting_sae`, `sae_l16` → `steering_sae` in phase8_2 and phase8_3


#### Add Early Returns (reduce nesting)
- [x] **steering_metrics.py:19-80** `calculate_correction_rate()` - deeply nested if/elif
  - **DONE**: Added `_detect_modified_column()` helper, early returns, max 2 levels nesting
  - Also refactored `calculate_corruption_rate()` with same pattern

- [x] **dataset_utils.py:163-225** `extract_code()` - 4 levels deep (was helpers.py)
  - **DONE**: Split into `_extract_raw_code()` + `_trim_to_function()` helpers

- [x] **activation_hooks.py:206-237** `_attention_hook()` - 4 levels deep
  - **DONE**: Guard clauses with early returns

- [x] **pile_filter_utils.py:82-105** `apply_pile_filter()` - 3 levels with if/else
  - **DONE**: Guard clauses with `continue`

- [x] **logging.py:396-416** `tqdm_with_logging()` - 3 levels in generator
  - **DONE**: Extracted `_log_milestone()` helper, guard with `continue`

#### Complex One-Liners to Split
- [x] **instruct_steering_analyzer.py:572-574** - 200+ char logger.info lines
  - **DONE**: Extracted `_format_effect_log()` helper function
- [x] **difficulty_steering_analyzer.py:357,364,371** - 300+ char f-string lines
  - **DONE**: Extracted `_format_counts_row()` helper function
- [x] **universality_analysis.py** - Hardcoded interpretation text (752+ chars)
  - **DONE**: Removed hardcoded interpretations entirely (code should output raw metrics only)

---

### 4.5 Steering Setup Extraction ✅ COMPLETED

Created `common/steering_setup.py` to consolidate duplicated loading code across steering phases.

**New utilities:**
- `load_pva_latents(config)` → Returns `PVALatents` dataclass
- `load_sae_and_directions(config, device, model, correct, incorrect)` → Returns `SAEDirections` dataclass
- `load_baseline_data(config, phase, filename)` → Returns `(DataFrame, phase_dir)`
- `split_by_correctness(data)` → Returns `(correct_df, incorrect_df)`

**Files updated to use shared utilities:**
- [x] `phase4_8_steering_analysis/steering_effect_analyzer.py` (~85 → ~30 lines)
- [x] `phase4_5_coefficient_grid_search/steering_coefficient_selector.py` (~110 → ~35 lines)
- [x] `phase5_3_weight_orthogonalization/weight_orthogonalizer.py` (~80 → ~30 lines)
- [x] `phase7_6_instruct_steering/instruct_steering_analyzer.py` (~90 → ~35 lines)

**Impact:** ~235 lines of duplicated code removed, replaced with ~100 lines of shared utilities.

---

### 4.6 Common Module Abstractions ✅ COMPLETED

All patterns identified during refactoring have been abstracted into `common/`:

| Pattern | Abstraction | Location | Status |
|---------|-------------|----------|--------|
| Checkpoint/resume logic | `CheckpointManager` class | `common/checkpoint_manager.py` | ✅ DONE |
| Start/end index filtering | `filter_by_range()` util | `common/phase_discovery.py` | ✅ DONE |
| Memory management | Utility functions | `common/memory_utils.py` | ✅ DONE |
| Dataset-aware path construction | `get_phase_output_dir()` | `common/phase_discovery.py` | ✅ DONE |
| SAE loading | `load_sae_for_config()` | `common/sae_loader.py` | ✅ DONE |

**Implementation notes:**
- `CheckpointManager`: Task ID-based tracking, version control, auto-cleanup
- `filter_by_range()`: Polymorphic (DataFrame/list), fixed Phase 1 inclusive-end anomaly
- Memory utils: Functions (`check_memory_usage()`, `get_memory_percent()`) rather than class
- `get_phase_output_dir()`: Handles model/dataset suffixes (_llama, _humaneval, etc.)
- `load_sae_for_config()`: Universal loader for GemmaScope (JumpReLU) and LlamaScope (TopK)

---

## Step 5: Polish & Extras (After Core Refactoring)

Nice-to-haves once the foundation is solid.
- [x] Is there bad in my current approach in one source of truth. Context: config.py
    - **Fixed:** Removed 7 defensive `getattr()` calls that created duplicate defaults
    - **Fixed:** Phase 4.10 had swapped default values (bug)
    - **Fixed:** Phase 3.8 manual path construction replaced with `discover_latest_phase_output(config=)`
    - **Created:** `docs/icml_notes.md` for ICML paper insights
- [x] **Delete unused config fields and methods** (18 items total)
    - **Deleted 15 unused fields:**
      - 8 SAE fields (replaced by MODEL_CONFIGS): `sae_repo_id`, `sae_width`, `sae_sparsity`, `sae_hook_component`, `sae_checkpoint_dir`, `sae_save_after_each_layer`, `sae_cleanup_after_layer`, `sae_use_memory_mapping`
      - 2 autosave fields (never implemented): `autosave_frequency`, `autosave_keep_last`
      - 2 memory fields (not used for control): `max_memory_usage_gb`, `max_gpu_memory_usage_gb`
      - 3 other unused: `progress_log_frequency`, `dataset_split`, `dataset_dir`
    - **Deleted 3 unused methods:** `get_phase_output_dir()`, `is_llama_model()`, `is_gemma_model()` (code uses `phase_discovery.py` or inline checks instead)
    - **Updated Phase 2.5 validation:** Now checks `model_name in MODEL_CONFIGS` instead of `sae_repo_id`
    - **Removed CLI mapping:** `'sae_model': 'sae_repo_id'` and `'dataset_dir': 'dataset_dir'`
    - **Impact:** ~40 lines removed, config only contains fields that are actually used
- [x] **Remove dead CLI arg_mapping entries** (12 entries + no_pile_filter handling)
    - **Removed 12 mappings** that referred to CLI arguments that don't exist in `run.py` parser:
      - `model`, `temperature`, `max_new_tokens` (model settings - use config.py instead)
      - `checkpoint_frequency`, `checkpoint_dir` (robustness - use config.py)
      - `latent_threshold`, `pile_threshold`, `pile_samples` (SAE - use config.py)
      - `random_seed`, `n_strata` (split - use config.py)
      - `temperatures`, `steering_coeffs` (fields don't even exist)
    - **Removed dead special handling:** `no_pile_filter` (no `--no-pile-filter` arg in parser)
    - **Kept 4 working mappings:** `start`, `end`, `verbose`, `viz_only`
    - **Impact:** arg_mapping shrunk from 16 to 4 entries, ~15 lines removed
- [x] **Remove legacy `--input` CLI support** (special_args handling)
    - **Removed from config.py:** `special_args` block that stored `_input`, `_test_temps`, `_test_samples_per_temp`
    - **Removed from run.py:** `--input` argument definition and `_input_file` setting
    - **Removed from problem_splitter.py:** `_input_file` fallback (now uses auto-discovery only)
    - **Updated CLAUDE.md:** Removed `--input` from documented options
    - **Impact:** ~20 lines removed across 4 files, auto-discovery is now the only method
- [x] **Remove unused `_load_from_env()` method**
    - Removed `_load_from_env()` method and its call from config.py
    - Feature was never used (no `SAE_CODE_*` environment variables anywhere)
    - **Impact:** ~25 lines removed
- [x] **Fix MODEL_CONFIGS silent fallback bug**
    - **Removed dead `get_model_config()` method** - never called anywhere
    - **Fixed `__post_init__()`** - now raises ValueError for unknown models instead of silently using Gemma-2B
    - **Fixed `sae_loader.py`** - removed if/else inference chain, now requires exact model name
    - **Impact:** Invalid model names now fail fast with clear error message
- [x] **Remove centralized `validate(phase)` method** (poor design)
    - Moved essential check (`dataset_end_idx >= start`) to `__post_init__`
    - Deleted `validate()` method (~90 lines) - phase runners already validate their own requirements
    - Removed `config.validate()` call from run.py
    - **Design improvement:** Config.py no longer knows about phases (Single Responsibility)
- [x] **Auto-discover `phase8_3_percentile` from Phase 8.2** (was hardcoded 70.0)
    - Changed `phase8_3_percentile: float = 70.0` to `Optional[float] = None`
    - Added `discover_optimal_percentile()` function to phase_discovery.py
    - Phase 8.3 now auto-discovers optimal percentile from Phase 8.2
    - Config override still works: setting explicit value bypasses auto-discovery
    - **Behavior:** If Phase 8.2 not run and no override set, raises FileNotFoundError with clear message

- [x] **Search tolerance configuration** (test vs production)
    - **Problem:** Golden section search tolerances were hardcoded or unused (dead code)
    - **Fixed Phase 4.6:** `phase4_6_tolerance` config now actually controls stopping condition
    - **Fixed Phase 4.6 extension:** Now adaptive based on coefficient magnitude (not steering type)
      - If `optimal_coeff >= 100` → use ±100 extension
      - If `optimal_coeff < 100` → use ±10 extension
    - **Fixed Phase 8.2:** Added `phase8_2_tolerance` and `phase8_2_refinement_radius` configs
    - **Config pattern in `common/config.py`:**
      ```python
      # Quick test: stop early (range < 10)
      phase4_6_tolerance: float = 10.0
      # Production: search to convergence (range < 1 = consecutive integers)
      # phase4_6_tolerance: float = 1.0
      ```
    - **For production runs:** Uncomment the `= 1` lines for Phase 4.6 and 8.2 tolerances
    - **Files updated:**
      - `common/config.py` - added configs with test/production comments
      - `phase4_6_golden_section_refinement/golden_section_refiner.py` - uses `config.phase4_6_tolerance`, adaptive extension
      - `common/search_optimization.py` - added `tolerance` parameter to TwoStageOptimizer
      - `phase8_2_threshold_optimizer/threshold_optimizer.py` - passes config values to optimizer

- [x] **Visualization color scheme standardized** (correction=green, corruption=red, preservation=gold)
    - Added color constants to `common/config.py`: `COLOR_CORRECTION`, `COLOR_CORRUPTION`, `COLOR_PRESERVATION`, etc.
    - Added "Visualization Color Scheme" documentation section to `CLAUDE.md`
    - Updated 11 visualization files:
        - Correct-predicting latent: blue → green (5 files: 3.8, 3.10, 3.11, 3.12, 4.7)
        - Preservation rate: blue/orange/purple → gold (6 files: 4.8, 4.14, 4.16, 5.6, 7.6, 7.9)
        - Correction rate: blue → green (4.16)
    - Semantic color mapping: green=good (correction, correct-predicting), red=bad (corruption, incorrect-predicting), gold=maintained (preservation)
- [x] **Terminology standardization complete** (feature→latent, preferring→predicting, feature_type→latent_type, latent_index→latent_idx)
    - Added terminology standards to `CLAUDE.md` SAE Terminology Standard section
    - Updated `common/phase_registry.py`: `top_20_features.json` → `top_20_latents.json` in patterns
    - Updated `docs/test_mbpp_with_imports.py`: renamed function and file reference
    - Updated `phase3_8_auroc_f1_evaluation/auroc_f1_evaluator.py`: `feature_type` → `latent_type` (3 functions)
    - Updated `phase7_12_instruct_auroc_f1/instruct_auroc_f1_evaluator.py`: `feature_type` → `latent_type`, `preferring` → `predicting`, `correct_preferring_feature` → `correct_predicting_latent`
    - Updated `phase3_10_temperature_auroc_f1/temperature_evaluator.py`: `preferring` → `predicting` in comments and titles
    - Updated `phase2_15_layerwise_visualization/layerwise_visualizer.py`: `preferring` → `predicting` in labels
    - Updated `phase4_6_golden_section_refinement/golden_section_refiner.py`: `latent_index` → `latent_idx` in dict keys
- [x] Read the new code. Use explore agents to find if there is still outdated docstrings/comments. If so, let's update them.
- [x] Refactor README.md and CLAUDE.md.
    - CLAUDE.md: 995 → 312 lines (69% redu" * 80` separators, emoji decorations
    - Kept all pandas unlimited display settings (`max_rows`, `max_columns`, `max_colwidth`, `width` = None)
    - List comprehension check: most for-loops in notebooks are OUTPUT cells (generated code), not actionable
- [x] Rename to sae-code-correctness (the folder, github repo, huggingface etc.)
    - Renamed codebase references from `pva_sae` to `sae_code_correctness`
    - External services (GitHub, HuggingFace, conda) to be renamed manually after commit
- [x] Rename the dataset split.
    - Changed `sae` → `selection` (direction selection)
    - Changed `hyperparams` → `tuning` (hyperparameter tuning)
    - Changed `validation` → `analysis` (mechanistic analysis)
    - Updated 16+ files with references to split names 
- [x] Fix MBPP import performance issue.
    - Imports integrated in `common/dataset_utils.py:268-290` via `data/phase0_4_mbpp_imports/required_imports.json`
    - **Failure mode analysis (Dec 2024):** Investigated false negative rate
    - Results: ~1% false negative rate (1 clear case in 100 tasks)
    - 55-62% of failures are WRONG_LOGIC, 10-15% TYPE_MISMATCH, rest runtime/syntax errors
    - Conclusion: MBPP evaluation is sound, no need to switch to MBPP+
    - Full analysis in `future_direction/dataset_and_evaluation_strategy.md`
- [x] Make other generating phases have raw_outputs also
- [x] Fix the figure generation code. Currently it looks soooo messy. (Comment: I forgot what this means. I'm not even sure if this is already been fixed.)
    - [x] Understand matplotlib and pandas logic or how it works. So I can help instruct my preference and good practice. (Comment. I pressed for time. Maybe not do this yet. Skip for now.)


### 5.1 ICML Visualizations (moved from ICML tasks)ction)
    - README.md: 362 → 188 lines (48% reduction)
    - Critical checklist at top with backward compatibility rule
    - Removed redundant sections, code style, screen examples
- [x] Improve notebooks. Remove unnecessary cells. Also do list comprehensions. Also make sure it works again after the new refactored code.
    - Deleted 2 broken notebooks: `notebooks/dataset_building_debug.ipynb`, `notebooks/sae_analysis.ipynb`
    - Cleaned 7 file_check notebooks: removed verbose `"=

These visualizations address reviewer feedback. Do after refactoring is stable.

- [x] **Feature-Selection Landscape scatter plot** (Reviewers 7JAK, jwL5) (Comment: How can we check if this is already wokring? Do we need to try to mock data first or run a subsample of the of the phases? I prefer the subsample. But let me know what you think. I just worried using a mock data will be messy since it will come from different source unless there is a cleaner way to test this? Like importing some library? like pytest in SWE but for data?)
    - X-axis: activation frequency of incorrect code, Y-axis: activation frequency of correct code.
    - Show all features as dots, highlight maybe the top 5 of each direction.
      - maybe also highlight the filtered out directions?

- [x] **Top-10 features table** (Reviewers 7JAK, jwL5) 
    - Create table showing top-10 features per direction
    - Columns: layer, feature_idx, separation score, t-statistic, AUROC

- [x] **Layer-wise visualization** (adapted from Ferrando et al. 2024, Figure 2)

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

- [x] **Steering coefficient search plots** (Reviewer 7JAK)
    - Show coefficient search process for appendix
    - X-axis: coefficient value, Y-axis: correction/corruption rate
    - [x] Fix this hardcoded  phase4_5_incorrect_coefficients: Updated commented grid to `[10, 20, 30, ..., 100, 200, ..., 1000]` (1-100 in increments of 10, then 200-1000 in increments of 100)
    - [x] Added Phase 8.7 for threshold search visualization (similar to Phase 4.7)

- [ ] **Cleanup mock test data** (created 2026-01-18)
    - Delete `data/phase8_2_llama/` mock data after real Phase 8.2 run
    - Delete `data/phase8_7_llama/` mock outputs after real Phase 8.2 run
    - Mock data used to test Phase 8.7 visualization code


## ICML Submission Tasks (Based on ICLR Reviewer Feedback)

Address reviewer concerns with minimal compute. **Run these AFTER refactoring phases are complete.**

**Note:** Visualization tasks moved to Step 5.1 for smoother refactoring flow.

### Paper Narrative Framing (UPDATED 2026-01-20)

- [x] **DECIDED: Linear Representation Hypothesis framing**

  **Core claim:** "Code correctness is linearly represented in LLMs"

  **Key experimental finding (2026-01-20):**
  - Phase 2.7 direction similarity: **LOW cosine similarity** (directions are NOT similar)
  - Probe steering: **13.3% correction rate** (4/30 problems, L11, coeff=34)
  - SAE steering: **0% correction rate** (0/30 problems, L16, coeff=30)
  - Probes OUTPERFORM SAE for steering

  **Updated framing (after results):**
  - Original plan: "Similar directions → converging evidence" ❌
  - Actual finding: Different directions but **same middle-layer region** → converging evidence ✓
  - Linear representation exists in middle layers; SAE and probes find different projections of it

  **Why Linear Representation > SAE framing:**
  - Probes beating SAE is awkward if paper is "about SAE"
  - Probes beating SAE is FINE if paper is "about linear representations"
  - Both methods validate the representation exists; they have different strengths

  **Complementary strengths (key narrative):**
  | Method | Strength | Weakness | Best For |
  |--------|----------|----------|----------|
  | SAE | Unsupervised, interpretable | Lower steering effect | Detection (F1=0.82) |
  | Probe | Higher steering effect | Supervised, less interpretable | Intervention |

  **Narrative arc (revised):**
  1. Intro: LLMs generate buggy code. Can we understand how they represent correctness?
  2. Core claim: Correctness is encoded as a linear direction in activation space
  3. SAE evidence: Unsupervised discovery → F1=0.82 detection, middle layers (L16-17)
  4. Probe evidence: Supervised validation → 13.3% correction, middle layers (L11-18)
  5. Key insight: Both find middle layers despite different directions → robust evidence for linear representation

  **One-liner:** "We show code correctness is linearly represented in LLMs, with SAE providing unsupervised detection (F1=0.82) and probes enabling causal intervention (13.3% correction)"

  **ICML rewriting note:** Focus on the LINEAR REPRESENTATION HYPOTHESIS, not the specific method. SAE is one tool, probes are another. The discovery is the representation, not the tool.

### Linear Probe Baseline Comparison (FULLY SYMMETRIC - Option A)

**Decision (2026-01-20):** Run ALL phases with BOTH SAE and Probe methods.
**Purpose:** Converging evidence for the Linear Representation Hypothesis.
**Reference:** `future_direction/linear_probe_vs_sae_comparison.md`

#### Key Results (Test Run 2026-01-20)

| Task | Method | Best Layer | Key Metric |
|------|--------|------------|------------|
| Detection | LogReg | L18 | AUROC 0.89 |
| | SAE | L17 | F1 0.82 |
| Steering | Mass-mean | L11 | Corr **13.3%** |
| | SAE | L16 | Corr 0% |

**Direction Similarity (Phase 2.7):** LOW cosine similarity → directions are different
**Layer Convergence:** Both methods find middle layers (L11-L18) → validates linear representation

#### Theoretical Pairing

| Probe Type | SAE Equivalent | Use Case |
|------------|----------------|----------|
| LogReg | t-statistic latent (Phase 2.10) | Detection/Prediction |
| Mass-mean | separation score latent (Phase 2.5) | Steering/Intervention |

#### Implementation Status

**Infrastructure (DONE):**
- [x] Phase 2.6: Probe Direction Computation (mass-mean + logreg per layer)
- [x] Phase 2.7: Direction Similarity Analysis
- [x] `--direction-source` CLI flag in run.py
- [x] `config.direction_source` field

**Phase Support Status:**

| Phase | Description | SAE | `probe_logreg` | `probe_mass_mean` | Status |
|-------|-------------|:---:|:--------------:|:-----------------:|--------|
| **Detection Phases** |||||
| 3.8 | AUROC/F1 Evaluation | ✓ | ✓ | - | ✅ DONE |
| 3.10 | Temperature Robustness | ✓ | ✓ | - | ✅ DONE |
| 7.3 | Instruct Detection | ✓ | - | - | ⏭️ N/A (activation extraction only) |
| 7.12 | Instruct AUROC/F1 | ✓ | ✓ | - | ✅ DONE |
| **Steering Phases** |||||
| 4.5 | Coefficient Grid Search | ✓ | - | ✓ | ✅ DONE |
| 4.6 | Golden Section Refinement | ✓ | - | ✓ | ✅ DONE |
| 4.7 | Coefficient Visualization | ✓ | - | ✓ | ✅ DONE |
| 4.8 | Steering Analysis | ✓ | - | ✓ | ✅ DONE |
| 7.6 | Instruct Steering | ✓ | - | ✓ | ✅ DONE |
| **Other Phases** |||||
| 5.3 | Weight Orthogonalization | ✓ | - | ✓ | ✅ DONE |
| 5.6 | Zero-Disc Orthogonalization | ✓ | - | - | ⏭️ N/A (SAE control experiment) |
| 6.3 | Attention Analysis | ✓ | - | - | ⏭️ N/A (uses existing Phase 4.8 data) |
| 8.2 | Threshold Optimizer | ✓ | [ ] | [ ] | 🔧 FUTURE (dual-direction architecture) |
| 8.3 | Selective Steering | ✓ | [ ] | [ ] | 🔧 FUTURE (dual-direction architecture) |

Legend: ✓ = implemented, [ ] = needs implementation, - = not applicable, ⏭️ = not applicable for this phase

#### Implementation Tasks

**Detection phases (add `--direction-source probe_logreg`):**
- [x] **Phase 3.10**: Temperature robustness with probe ✅ DONE (2026-01-20)
  - Load logreg direction from Phase 2.6 via `load_probe_directions_for_predicting()`
  - Compute detection scores across temperatures
- [x] **Phase 7.12**: Instruct AUROC/F1 with probe ✅ DONE (2026-01-20)
  - Added `load_instruct_activations_probe()` function
  - Supports `--direction-source probe_logreg`
- [ ] **Phase 7.3**: Instruct model detection with probe - N/A (activation extraction only, no direction usage)
- [ ] **Phase 8.2**: Threshold optimizer with probe detection - FUTURE (dual-direction architecture)

**Steering phases (add `--direction-source probe_mass_mean`):**
- [x] **Phase 5.3**: Weight orthogonalization with probe direction ✅ DONE (2026-01-20)
  - Orthogonalize mass-mean direction from weights
  - Supports `--direction-source probe_mass_mean`
- [x] **Phase 7.6**: Instruct steering with probe ✅ DONE (2026-01-20)
  - Added probe coefficient discovery from Phase 4.6 probe output
  - Supports `--direction-source probe_mass_mean`
- [ ] **Phase 5.6**: Zero-disc orthogonalization with probe - N/A (SAE control experiment)
- [ ] **Phase 6.3**: Attention analysis with probe direction - N/A (uses existing Phase 4.8 data)
- [ ] **Phase 8.3**: Selective steering with probe - FUTURE (dual-direction architecture)

#### Implementation Pattern

Each phase needs this pattern:

```python
# In phase runner __init__ or _load_dependencies:
if config.direction_source == "probe_logreg":
    probe_data = load_probe_directions(config)  # From Phase 2.6
    self.direction = probe_data["logreg"]["direction"]
    self.layer = probe_data["logreg"]["best_layer"]
elif config.direction_source == "probe_mass_mean":
    probe_data = load_probe_directions(config)
    self.direction = probe_data["mass_mean"]["direction"]
    self.layer = probe_data["mass_mean"]["best_layer"]
else:  # SAE (default)
    # existing SAE loading code
```

**Output directories:** When `--direction-source probe_*`, output goes to `phase{N}_probe/`

#### Methodology Note (vs Marks & Tegmark 2023)

**Their approach:** Expensive causal patching → layer selection
**Our approach:** Separation score (cheap) → layer selection, Phase 4 steering = causal validation

This is valid because:
1. Separation score is a cheap proxy for causal relevance
2. If it picks a bad layer, steering shows 0% correction → we'd know
3. Actual steering experiments ARE the ground truth causal validation

---

- [x] **Selective steering implementation** (Reviewers RXZd, vRko) - DONE
    - Conclusion: Selective steering in current form still not advisable. Better strategy: generate without steering first, only apply steering on retry if code is detected as wrong.

- [ ] **Multi-Model Experiments** (All reviewers) - RUN AFTER REFACTORING

    **Decision: Run ALL phases for larger models (not just core)**

    **Hypothesis:** Larger models may have symmetric representations (both correct AND incorrect-predicting directions work), unlike Gemma-2B's asymmetry.

    | Metric | Gemma-2B (current) | Larger models (predicted) |
    |--------|-------------------|---------------------------|
    | Correct-predicting F1 | 0.504 (weak) | Higher (~0.7+) |
    | Incorrect-predicting F1 | 0.821 (strong) | Similar or higher |
    | Correct steering effect | Weak | Stronger |

    **Why full phases (not just core)?**
    - This is hypothesis-driven, not redundant replication
    - If true → major finding (scale affects representation completeness)
    - If false → still valuable (asymmetry is fundamental across scales)
    - Orthogonalization/attention differences would be compelling evidence

    **Paper framing:**
    > "We investigate whether the asymmetry between correct and incorrect-predicting representations persists across model scales."

    #### Gemma-9B
    - [ ] Run all phases on `google/gemma-2-9b`
    - [ ] Run all phases on `google/gemma-2-9b-it` (instruct)
    - [ ] Compare asymmetry metrics to Gemma-2B

    #### LLaMA-8B
    - [ ] Run all phases on `meta-llama/Llama-3.1-8B` with `llama_scope_lxr_8x`
    - [ ] Run all phases on `meta-llama/Llama-3.1-8B-Instruct`
    - [ ] Compare asymmetry metrics to Gemma models

    #### HumanEval (transfer validation)
    **Purpose:** Test if MBPP-discovered directions transfer to different benchmark.
    **Core phases only** (detection + steering) — ortho/attention/temp test the direction, already validated on MBPP.

    - [ ] Gemma-2B + HumanEval: Core phases (detection + steering)
    - [ ] Gemma-9B + HumanEval: Core phases (detection + steering)
    - [ ] LLaMA-8B + HumanEval: Core phases (detection + steering)

    **Prerequisites:** Step 2 multi-model support, Step 3 model-agnostic abstractions
    **SAE Verified**: `fnlp/Llama-Scope` 32K (8x expansion) matches Neuronpedia's `llamascope-res-32k`
    **Note:** Check if LLAMA orthogonalization weights need adjustment: `['embed', 'attn_o', 'mlp_down']`

- [x] **Error type breakdown analysis** (Reviewer RXZd) - DONE
  - [x] Categorize errors: syntax, logic, type, runtime, etc.
  - [x] Detection: Which error types does incorrect-predicting direction catch better?
  - [x] Steering: Which error types does correction work on?
  - Supports MechInterp narrative by showing what the linear direction encodes
  - **Implementation:**
    - Added `EvaluationResult` dataclass and `evaluate_code_with_error_type()` to `common/dataset_utils.py`
    - Error categories: passed, syntax, name, type, logic, runtime, timeout
    - Updated 13 phase modules to capture `baseline_error_type`, `steered_error_type`, or `orthogonalized_error_type`
    - Created Phase 9.1 (`phase9_1_error_type_analysis/`) for analysis and visualization
    - Fixed Phase 4.8 to properly export `steered_error_type` in JSON and parquet files
  - **Commit:** `fb07b771a` (feat: Add error type breakdown analysis for ICLR reviewer question)    

- [x] **Feature threshold sensitivity analysis** (Reviewer RXZd)
    - [x] **Infrastructure ready**: Phase 2.3 extracts pile frequency computation, enabling easy threshold testing
    - [x] Test sensitivity to the >2% activation threshold on pile-10k (vary threshold, rerun Phase 2.5/2.10 only)
    - [x] Report how many features get filtered out in top-20 (needs full 10,000 pile samples)
    - **Reporting plan**:
        - Main paper: "Of top-20 candidates, X filtered at 2% threshold, Y remained. Selected feature ranked #Z."
        - Appendix: Small table showing filtering counts at 1%, 2%, 5% thresholds (shows threshold isn't arbitrary)
    - **Prediction (uncertain)**: Expect few filtered (0-5 out of 20) because top features by separation score should already be code-specific. Adjust interpretation based on actual results:
        - 0-5 filtered → filter validates selection, report as safety net
        - 5-10 filtered → filter is doing meaningful work, emphasize its importance
        - 10+ filtered → may indicate selection method issues, investigate further
    - **Note**: Current 2% threshold is copied from Ferrando et al. 2024 (entities paper). No first-principles justification yet. Results will inform whether to keep, adjust, or provide post-hoc justification.
- [x] In AUROC and F1? How did the inspiration do it? How did they make the thresholds to test? Copy them or improve.
- [ ] In steering if performance is not changing stop the search already and use the lower steering coefficient. In golden section search.
    - Consider applying the same in phase 3.8 also in threshold.

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

---

## Step 6: Multi-GPU Parallel Execution (After Experiments Work)
- [x] **CLI `--model` and `--dataset` arguments** - Already implemented in run.py:84-101, config.py:232-269
- [ ] Consider condensing the code more like some can be just a variation of one code like steering etc. But take this with high risk and put little importance. Leaning to not do this. or maybe atleast have steering function that will always be called. Ask CC if my current architecture/codebase design make sense or if could be better.
- [ ] Test all phase one by one first if it is all running.
    - [ ] Exmaine each of the output file.
    - [ ] Code review manually. With CC help ofcourse but read all code manually. Make sure I understand and it is correct. Make it a rule for me to actually read the code before testing.
    - [ ] Understand the methods especially the linear algebra. Visualize etc. Enter learning mode. Learn to code. Get used to it.
- [x] Consider batching or not since one problem already do 50% GPU usage?
- [x] and running all four gpu at once. 
- [x] Consider learning and implementing other optimization.
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

- [x] Add `--model` and `--dataset` CLI flags to `run.py` - Already done (lines 84-101)
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


## Future Work (added 2026-01-20)

### Phase 8.2/8.3 Probe Support

Phases 8.2 and 8.3 have a dual-direction architecture (one direction for prediction/threshold checking, another for steering). Full probe support requires careful design:

- [ ] **Phase 8.2 Threshold Optimizer:**
  - Load LogReg probe for incorrect-predicting threshold checking
  - Load Mass-mean probe for correct steering direction
  - Coordinate two different layers (prediction vs steering layers may differ)

- [ ] **Phase 8.3 Selective Steering:**
  - Same dual-direction requirement as Phase 8.2
  - Probe for threshold gating + probe for steering intervention

### Phase 9 Redesign

Current Phase 9.1 implementation should be deleted and redesigned. Instead of a separate analysis phase, add error type statistics directly to each phase's output:

- [ ] Delete current Phase 9.1 implementation
- [ ] Add error type statistics to each phase's output:
  - Phase 1: `baseline_error_type_distribution`
  - Phase 3.5, 3.10: `temperature_error_type_distribution`
  - Phase 4.8, 7.6: `steered_error_type_distribution`
  - Phase 5.3, 5.6: `orthogonalized_error_type_distribution`
- [ ] Create summary visualization that reads from all phases' built-in stats

---

FINAL
- [ ] Add here the ICML LaTeX.
- [ ] Improve the aesthetic of my visualization.