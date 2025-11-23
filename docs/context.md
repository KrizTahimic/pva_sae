# PVA-SAE Technical Manual - Context and Mapping

This document serves as working memory for creating the technical manual. It maps the methodology.pdf diagram to actual code implementation.

## Methodology-to-Phase Mapping

### PCDGE Pattern (Central to All Phases)
**Prompt-Capture-Decompose-Generate-Evaluate** is the core pattern used throughout:
- **Prompt**: Build prompt from MBPP problem using `PromptBuilder.build_prompt()`
- **Capture**: Extract activations via `ActivationExtractor` hooks during generation
- **Decompose**: (For steering) Apply SAE decomposition with `JumpReLUSAE.encode()`
- **Generate**: LLM generates code solution
- **Evaluate**: Test execution using `evaluate_code()` → Correct (pass@1) or Incorrect

### 1. Dataset Preparation (Green Section in Methodology)

| Methodology Name | Code Phase | Main File | Description |
|-----------------|------------|-----------|-------------|
| MBPP Complexity Grading | Phase 0 | `phase0_difficulty_analysis/difficulty_analyzer.py` | Calculates cyclomatic complexity for MBPP problems |
| Data Splitting | Phase 0.1 | `phase0_1_problem_splitting/problem_splitter.py` | Stratified split: 50% SAE analysis, 10% hyperparameter, 40% validation |
| PCDGE (Template) | Phase 1 | `phase1_simplified/runner.py` | Generate code at temp=0.0, capture activations, evaluate correctness |

### 2. Direction Selection (Blue Section in Methodology)

| Methodology Name | Code Phase | Main File | Description |
|-----------------|------------|-----------|-------------|
| Filter Out General Directions | Phase 2.2 | `phase2_2_pile_caching/runner.py` | Cache Pile-10k activations to identify general language features |
| Compute Activation Frequency | Phase 2.5 | `phase2_5_simplified/sae_analyzer.py` | Calculate how often features activate (>0) |
| Compute Separation Scores | Phase 2.5 | `phase2_5_simplified/sae_analyzer.py` | Mean activation difference: correct vs incorrect |
| Compute T-Statistic | Phase 2.10 | `phase2_10_t_statistic_latent_selector/t_statistic_selector.py` | Welch's t-test for feature discrimination |
| ArgMax Selection (Predicting) | Phase 2.5/2.10 | Output: `top_20_features.json` | Top 20 features predicting correctness |
| ArgMax Selection (Steering) | Phase 2.5/2.10 | Output: `top_20_features.json` | Top 20 features for steering interventions |

### 3. Mechanistic Analysis (Purple Section in Methodology)

#### Left Side: Predicting Directions

| Methodology Name | Code Phase | Main File | Description |
|-----------------|------------|-----------|-------------|
| AUROC | Phase 3.8 | `phase3_8/auroc_f1_evaluator.py` | Area Under ROC Curve for feature discrimination |
| F1 (threshold tuning) | Phase 3.8 | `phase3_8/auroc_f1_evaluator.py` | F1 score with optimal threshold from hyperparameter set |
| Temperature Variation | Phase 3.5, 3.10 | `phase3_5_temperature_robustness/` | Test at temps: [0.0, 0.3, 0.6, 0.9, 1.2] |
| Difficulty Variation | Phase 3.12 | `phase3_12_difficulty_auroc_f1/` | Analyze AUROC across complexity levels |

#### Right Side: Steering Directions

| Methodology Name | Code Phase | Main File | Description |
|-----------------|------------|-----------|-------------|
| Activation Steering (coefficient search) | Phase 4.5, 4.6 | `phase4_5_model_steering/`, `phase4_6_binary_refinement/` | Adaptive search for optimal steering coefficient |
| Correction Rate | Phase 4.8 | `phase4_8_steering_analysis/steering_effect_analyzer.py` | % incorrect→correct after steering |
| Corruption Rate | Phase 4.8 | `phase4_8_steering_analysis/steering_effect_analyzer.py` | % correct→incorrect after steering |
| Binomial Test | Phase 4.8, 4.14 | Statistical significance testing | Validate steering effects vs random |
| Attention Analysis | Phase 6.3 | `phase6_3_attention_analysis/attention_analyzer.py` | Compare attention patterns: baseline vs steered |
| Weight Orthogonalization | Phase 5.3, 5.6, 5.9 | `phase5_3_weight_orthogonalization/` | Permanent weight modification instead of hooks |

#### Center: Additional Analyses

| Methodology Name | Code Phase | Main File | Description |
|-----------------|------------|-----------|-------------|
| Persistence Testing | Phase 7.3, 7.6, 7.12 | `phase7_3_instruct_baseline/` | Test feature transfer to instruction-tuned model |
| Logit Lens Analysis | Not implemented | N/A | Future work |

## Main Functions by Phase

### Phase 0: Complexity Grading
- **Class**: `MBPPDifficultyAnalyzer`
- **Main Functions**:
  - `analyze_dataset(dataset_manager)` → Returns enriched DataFrame with complexity
  - `get_complexity_distribution(df)` → Returns complexity statistics
- **Helper Functions**:
  - `get_cyclomatic_complexity(code)` → int: Uses radon library

### Phase 0.1: Data Splitting
- **Main Functions**:
  - `split_problems(df, config)` → Dict[split_name, task_ids]: Stratified splitting
  - `create_complexity_strata(task_ids, complexity_scores, n_strata)` → List of arrays
  - `apply_stratified_interleaving(strata, ratios)` → List of splits
- **Helper Functions**:
  - `create_interleaved_pattern(ratios)` → Pattern for round-robin sampling
  - `save_splits(splits, output_dir, df)` → Saves parquet files

### Phase 1: Dataset Generation (PCDGE)
- **Class**: `Phase1Runner`
- **Main Functions**:
  - `generate_and_extract(prompt, task_id)` → (generated_text, activations_dict)
  - `process_task(task)` → Dict with results (code, correctness, activations)
- **Helper Functions**:
  - `PromptBuilder.build_prompt(problem, test_cases)` → Formatted prompt
  - `ActivationExtractor` (class): Captures residual stream at last token
  - `evaluate_code(code, test_list)` → bool: Executes tests with timeout
  - `extract_code(generated_text, prompt)` → Extracts function definition

### Phase 2.2: Pile Caching
- **Main Functions**:
  - `run_phase2_2_caching(config, device)` → Caches activations from Pile-10k
- **Helper Functions**:
  - `PileActivationHook` (class): Captures activations at random word positions
  - `find_word_position(input_ids, tokenizer, word)` → Token position
  - `validate_pile_sample(text, word)` → bool

### Phase 2.5: SAE Analysis
- **Class**: `SimplifiedSAEAnalyzer`
- **Main Functions**:
  - `analyze_layer(layer_idx)` → Dict with separation scores and statistics
  - `load_activations_for_layer(layer_idx, category)` → (task_ids, activations)
- **Helper Functions**:
  - `load_gemma_scope_sae(layer_idx, device)` → JumpReLUSAE model
  - `JumpReLUSAE.encode(x)` → SAE feature activations
  - `compute_separation_metrics(correct_features, incorrect_features)` → Scores

### Phase 2.10: T-Statistic Selection
- **Class**: `TStatisticSelector`
- **Main Functions**:
  - `compute_t_statistics(correct_features, incorrect_features)` → Dict with t-stats
  - `select_top_features(layer_idx)` → Top features by t-statistic
- **Helper Functions**:
  - `scipy.stats.ttest_ind()` with `equal_var=False` (Welch's t-test)

### Phase 3.8: AUROC/F1 Evaluation
- **Main Functions**:
  - `calculate_metrics(y_true, scores, threshold, feature_type)` → Dict with metrics
  - `find_optimal_threshold(y_true, scores)` → (threshold, metrics)
- **Helper Functions**:
  - `sklearn.metrics.roc_auc_score()` → AUROC
  - `sklearn.metrics.f1_score()` → F1
  - Grid search over 100 threshold candidates

### Phase 4.5: Steering Coefficient Selection
- **Class**: `SteeringCoefficientSelector`
- **Main Functions**:
  - `adaptive_coefficient_search(feature_info, dataset)` → Optimal coefficient
  - `test_coefficient(coefficient, feature, problems)` → Correction/corruption rates
- **Helper Functions**:
  - `create_steering_hook(layer, feature_idx, coefficient, sae)` → PyTorch hook
  - `calculate_correction_rate(baseline_results, steered_results)` → float
  - `calculate_corruption_rate(baseline_results, steered_results)` → float

### Phase 4.8: Steering Effect Analysis
- **Class**: `SteeringEffectAnalyzer`
- **Main Functions**:
  - `analyze_steering_effects(direction_type)` → Results with rates and significance
  - `_apply_steering(problem, feature, coefficient)` → Steered generation result
- **Helper Functions**:
  - `scipy.stats.binomtest()` → Statistical significance
  - Captured attention patterns for mechanistic analysis

### Phase 5.3: Weight Orthogonalization
- **Class**: `WeightOrthogonalizer`
- **Main Functions**:
  - `orthogonalize_and_test(feature_info)` → Results with permanent modifications
  - `_orthogonalize_weights(layer, decoder_direction)` → Modified model
- **Helper Functions**:
  - `orthogonalize_gemma_weights()` → Applies Gram-Schmidt orthogonalization

### Phase 6.3: Attention Analysis
- **Class**: `AttentionAnalyzer`
- **Main Functions**:
  - `analyze_patterns(baseline_dir, steered_dir)` → Statistical comparisons
  - `compute_attention_statistics(attention_data)` → Summary stats
- **Helper Functions**:
  - Loads attention patterns from Phase 3.5 (baseline) and Phase 4.8 (steered)
  - Compares attention head behaviors

### Phase 7.3: Instruction-Tuned Baseline
- **Class**: `InstructBaselineRunner`
- **Main Functions**:
  - `run_baseline(validation_split)` → Generate with gemma-2-2b-it
  - `_discover_best_layers()` → Loads layers from Phase 2.5/2.10
- **Helper Functions**:
  - Same PCDGE pattern but with instruction-tuned model

## Running the Code

### Basic Command Structure
```bash
python3 run.py phase X [options]
```

### Common Options
- `--start N --end M`: Process dataset indices N to M (for parallel processing)
- `--input FILE`: Override auto-discovery, use specific input file
- `--correction-only`: Only run correction experiments (Phase 4.5, 4.8)
- `--corruption-only`: Only run corruption experiments
- `--model NAME`: Specify model (default: google/gemma-2-2b)

### Example Commands
```bash
# Phase 0: Difficulty analysis
python3 run.py phase 0

# Phase 0.1: Data splitting with custom seed
python3 run.py phase 0.1 --random-seed 42 --n-strata 10

# Phase 1: Generate first 100 problems
python3 run.py phase 1 --start 0 --end 100

# Phase 2.5: SAE analysis (auto-discovers Phase 1 output)
python3 run.py phase 2.5

# Phase 3.8: Evaluate AUROC/F1
python3 run.py phase 3.8

# Phase 4.5: Test correction only
python3 run.py phase 4.5 --correction-only
```

### Multi-GPU Execution
```bash
# Split work across 3 GPUs for Phase 1 (487 total problems)
CUDA_VISIBLE_DEVICES=0 python3 run.py phase 1 --start 0 --end 162 &
CUDA_VISIBLE_DEVICES=1 python3 run.py phase 1 --start 163 --end 325 &
CUDA_VISIBLE_DEVICES=2 python3 run.py phase 1 --start 326 --end 487 &
wait
```

## Notes for Documentation

### What to Include:
1. **Complete code listings** for all vital functions (not just signatures)
2. **Concrete command-line examples** with actual flags
3. **Hook implementations** with line-by-line explanations
4. **Output examples** showing what phases produce
5. **File structure diagrams** showing data/ directory layout
6. Use methodology names (not phase numbers) in section headers
7. Reference code location: "implemented in `phase2_5_simplified/sae_analyzer.py:91`"
8. Explain PCDGE pattern upfront as it's used everywhere

### What to Skip:
- Visualization-only phases (2.15, 3.11, 4.7)
- Internal helper functions not vital to understanding
- Boilerplate logging code

### Key Concepts to Explain with Code:
1. **Residual Stream**: Activations captured at last prompt token before generation
2. **SAE (Sparse Autoencoder)**: JumpReLU SAE from GemmaScope decomposes activations
3. **Separation Score**: Mean(correct) - Mean(incorrect) for feature activation
4. **Steering Hook**: PyTorch forward hook that adds `coefficient * SAE_decoder_direction`
5. **Correction Rate**: % of incorrect solutions that become correct after steering
6. **Corruption Rate**: % of correct solutions that become incorrect after steering
7. **Binomial Test**: Statistical test to verify steering effects are significant
