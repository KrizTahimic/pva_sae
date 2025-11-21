# HumanEval and LLAMA Implementation Plan

## Overview

This document outlines the implementation strategy for extending PVA-SAE to support:
1. **HumanEval dataset** (in addition to MBPP)
2. **LLAMA-3-8B model** (in addition to Gemma-2-2B)

This addresses feedback from ICLR 2026 reviewers (RXZd, vRko, 7JAK, jwL5) requesting validation across multiple datasets and models.

---

## Background: What Reviewers Want

From `iclr_reviewers_feedback.md`:

> - [ ] **LLAMA and HumanEval extension** (Reviewers RXZd, vRko, 7JAK, jwL5)
>   - [ ] Perform Mechanistic Analysis with `HumanEval`
>   - [ ] Run all tests on `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` with `llama_scope_lxr_8x`
>   - [ ] Perform all Mechanistic Analysis

**Key constraint**: We do NOT need to re-run hyperparameter tuning and direction selection (Phases 2.5, 2.10, 4.5, 4.6). We'll reuse features identified on MBPP+Gemma.

---

## Current State Analysis

### What Works Today (MBPP + Gemma-2-2B)

1. **Phase 0/0.1**: MBPP difficulty analysis and splitting
2. **Phase 1**: Dataset generation with activation capture
3. **Phase 2.5/2.10**: SAE feature selection (t-statistics, separation scores)
4. **Phase 3.x**: Statistical validation (AUROC, F1, temperature robustness)
5. **Phase 4.x**: Causal validation (steering coefficients, effect analysis)
6. **Phase 5.x**: Weight orthogonalization
7. **Phase 6.3**: Attention analysis
8. **Phase 7.x**: Base vs instruct model comparison
9. **Phase 8.3**: Selective steering

### Key Architectural Insights

From codebase exploration:

- **Dataset-agnostic phases**: 2.5, 2.10, 3.8, 4.14 (work with any activation format)
- **Dataset-specific phases**: 0, 0.1, 1, 3.5, 4.8, 6.3, 7.x, 8.3 (reference MBPP structure)
- **Model-specific phases**: 1, 2.2, 2.5, 4.5, 4.8, 5.x, 7.x (load model weights, SAEs)
- **Prompt building**: `common/prompt_utils.py` - MBPP-specific template
- **Auto-discovery**: `common/utils.py` - no dataset/model filtering currently
- **Config system**: `common/config.py` - single model/dataset at a time

---

## Design Decision 1: Phase Adaptation vs Duplication

### Option A: Duplicate Phases (Rejected)
```
phase1_simplified/           # MBPP + Gemma
phase1_humaneval/           # HumanEval + Gemma
phase1_llama/              # MBPP + LLAMA
phase1_humaneval_llama/    # HumanEval + LLAMA
```

**Pros**: Clear separation, no risk of breaking existing pipeline
**Cons**: 4x code duplication, maintenance nightmare, violates DRY

### Option B: Adaptive Phases (RECOMMENDED)
```
phase1_simplified/          # Supports all datasets + models via config
```

**Pros**: Single codebase, easier maintenance, follows best practices
**Cons**: Requires more upfront refactoring, testing needed

**Decision**: Use **Option B** - adapt existing phases with dataset/model parameters.

---

## Design Decision 2: Output Directory Naming

### Option A: Dataset+Model Suffixes (RECOMMENDED)
```
data/phase1_0/                    # MBPP + Gemma (default, no suffix)
data/phase1_0_humaneval/          # HumanEval + Gemma
data/phase1_0_llama/              # MBPP + LLAMA
data/phase1_0_humaneval_llama/    # HumanEval + LLAMA
```

**Naming convention**:
- Default (MBPP + Gemma): No suffix
- Dataset change only: `_{dataset}`
- Model change only: `_{model_short}`
- Both change: `_{dataset}_{model_short}`

**Model short names**: `gemma` (gemma-2-2b), `llama` (llama-3-8b)

**Pros**: Clear, discoverable, backwards compatible
**Cons**: Slightly longer paths

### Option B: Subdirectories
```
data/phase1_0/mbpp/gemma/
data/phase1_0/humaneval/gemma/
data/phase1_0/mbpp/llama/
```

**Pros**: More organized for many variants
**Cons**: Deeper nesting, auto-discovery complexity

**Decision**: Use **Option A** - suffix-based naming for simplicity.

---

## Design Decision 3: Feature Reuse Strategy

**Question**: Should we reuse Gemma-MBPP features for all experiments?

### Experimental Matrix

|  | MBPP | HumanEval |
|---|------|-----------|
| **Gemma** | Original features (Phases 2.5, 2.10, 4.5) | **Reuse Gemma features** |
| **LLAMA** | **??? Reuse or new ???** | **??? Reuse or new ???** |

### Option A: Full Reuse (Cross-Model Transfer)
Use Gemma-MBPP features for all 4 combinations.

**Rationale**: Tests whether features transfer across datasets AND models
**Risk**: LLAMA SAE has different feature space - indices may not align
**Feasibility**: Requires feature mapping or interpretation alignment

### Option B: Per-Model Features
- Gemma experiments: Use Gemma-MBPP features
- LLAMA experiments: Select new features (run Phases 2.5, 2.10, 4.5 on LLAMA+MBPP)

**Rationale**: Each model has its own feature space
**Effort**: Medium (need to run feature selection on LLAMA)
**Result**: Fair comparison within model, unclear cross-model comparison

### Option C: Hybrid (RECOMMENDED)
- **HumanEval + Gemma**: Reuse Gemma-MBPP features (tests cross-dataset transfer)
- **LLAMA + MBPP**: Select new LLAMA-MBPP features (run Phases 2.5, 2.10, 4.5)
- **LLAMA + HumanEval**: Use LLAMA-MBPP features (tests LLAMA cross-dataset)

**Rationale**:
- Maintains within-model comparability
- Tests cross-dataset transfer for each model
- Scientifically sound (compare apples to apples)

**Decision**: Recommend **Option C** unless user has strong preference.

---

## Design Decision 4: HumanEval Prompt Format

### MBPP Format (Current)
```python
"""Write a function to find the similar elements from the given two tuple lists.
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
# Solution:
"""
```

### HumanEval Format
```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**Key differences**:
1. HumanEval provides function signature + docstring
2. MBPP provides problem description + assertions
3. HumanEval tests are in separate `test` field (not shown in prompt)
4. Evaluation differs: HumanEval needs to complete function body

### Implementation Strategy

Create `HumanEvalPromptBuilder` class:
```python
class HumanEvalPromptBuilder(PromptBuilder):
    @classmethod
    def build_prompt(cls, prompt: str, **kwargs) -> str:
        # HumanEval 'prompt' field already contains signature + docstring
        # Just add code initiator to signal completion
        return f"{prompt}\n    # Solution:\n"
```

---

## Design Decision 5: Auto-Discovery Enhancement

### Current Limitation
`discover_latest_phase_output("phase1")` returns most recent file matching pattern, regardless of dataset/model.

### Proposed Solution: Context-Aware Discovery

```python
def discover_latest_phase_output(
    phase: str,
    dataset: Optional[str] = None,
    model: Optional[str] = None
) -> Path:
    """
    Find latest phase output filtered by dataset/model context.

    Args:
        phase: Phase number (e.g., "1", "2.5", "4.8")
        dataset: Dataset name ("mbpp", "humaneval", None for default)
        model: Model short name ("gemma", "llama", None for default)

    Returns:
        Path to latest matching output file
    """
    # Get base directory with suffix
    output_dir = get_phase_output_dir(phase, dataset, model)

    # Use existing discovery logic
    patterns = PHASE_CONFIGS[phase]["patterns"]
    return find_latest_file(output_dir, patterns)

def get_phase_output_dir(
    phase: str,
    dataset: Optional[str] = None,
    model: Optional[str] = None
) -> Path:
    """Generate output directory path with appropriate suffix."""
    base_dir = f"data/phase{phase}"

    # Normalize to config defaults if None
    dataset = dataset or config.dataset_name
    model = model or config.model_variant

    # Build suffix
    suffix = ""
    if dataset != "mbpp" or model != "gemma":
        parts = []
        if dataset != "mbpp":
            parts.append(dataset)
        if model != "gemma":
            parts.append(model)
        suffix = "_" + "_".join(parts)

    return Path(base_dir + suffix)
```

**Usage in phases**:
```python
# Old way
input_path = discover_latest_phase_output("1")

# New way (context-aware)
input_path = discover_latest_phase_output("1", dataset=config.dataset_name, model=config.model_variant)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Goal**: Set up dataset/model abstraction without breaking existing functionality

1. **Config Enhancement** (`common/config.py`)
   - Add `dataset_name` field (default: "mbpp")
   - Add `model_variant` field (default: "gemma")
   - Add model-specific configs (SAE repo, layer count)
   - Add dataset-specific configs (HF name, prompt builder)

2. **Directory Naming** (`common/utils.py`)
   - Implement `get_phase_output_dir(phase, dataset, model)`
   - Update `discover_latest_phase_output()` with context params
   - Add backward compatibility (fallback to old paths)

3. **Dataset Abstraction** (`common/dataset_loader.py` - NEW)
   - Create `DatasetLoader` base class
   - Implement `MBPPDatasetLoader`
   - Implement `HumanEvalDatasetLoader`
   - Factory function: `get_dataset_loader(dataset_name)`

4. **Prompt Builder Factory** (`common/prompt_utils.py`)
   - Keep existing `PromptBuilder` (MBPP)
   - Add `HumanEvalPromptBuilder`
   - Add `get_prompt_builder(dataset_name)` factory

5. **CLI Enhancement** (`run.py`)
   - Add `--dataset` argument (choices: mbpp, humaneval)
   - Add `--model` argument (choices: gemma, llama)
   - Pass to config for all phases

### Phase 2: HumanEval Support (Week 2)

**Goal**: Run full pipeline on HumanEval + Gemma using MBPP features

1. **Phase 0 Adaptation** (if needed)
   - HumanEval doesn't need difficulty analysis (only 164 problems)
   - Option: Skip Phase 0/0.1, use full dataset

2. **Phase 1 Adaptation**
   - Update to use `DatasetLoader` instead of `load_mbpp_from_phase0_1()`
   - Update to use `get_prompt_builder(dataset_name)`
   - Add HumanEval evaluation logic (function completion)
   - Test: `python3 run.py phase 1 --dataset humaneval --start 0 --end 10`

3. **Feature Reuse Configuration**
   - Add feature override mechanism
   - Phase 3.8/4.8/etc load features from Phase 2.5
   - Allow `--feature-source` to specify which Phase 2.5 to use
   - Default: Use Phase 2.5 from matching model

4. **Run HumanEval Pipeline**
   ```bash
   python3 run.py phase 1 --dataset humaneval --model gemma
   python3 run.py phase 3.8 --dataset humaneval --model gemma
   python3 run.py phase 4.8 --dataset humaneval --model gemma
   # (skip 2.5, 2.10, 4.5, 4.6 - reuse from MBPP+Gemma)
   ```

### Phase 3: LLAMA Support (Week 3)

**Goal**: Run full pipeline on MBPP + LLAMA with new feature selection

1. **Model Config** (`common/config.py`)
   - Add Llama-3.1-8B config
   - Base model: `meta-llama/Llama-3.1-8B`
   - Instruct model: `meta-llama/Llama-3.1-8B-Instruct`
   - SAE: `llama_scope_lxr_8x`
   - Layer count: 32, Hidden size: 4096

2. **Run Feature Selection on LLAMA + MBPP**
   ```bash
   python3 run.py phase 1 --model llama      # Generate dataset
   python3 run.py phase 2.2 --model llama    # Pile baseline
   python3 run.py phase 2.5 --model llama    # SAE analysis
   python3 run.py phase 2.10 --model llama   # T-statistic selection
   python3 run.py phase 4.5 --model llama    # Coefficient search
   python3 run.py phase 4.6 --model llama    # Golden section
   ```

3. **Run Mechanistic Analysis on LLAMA + MBPP**
   ```bash
   python3 run.py phase 3.8 --model llama
   python3 run.py phase 4.8 --model llama
   python3 run.py phase 5.3 --model llama    # Weight orthogonalization
   python3 run.py phase 6.3 --model llama    # Attention
   python3 run.py phase 8.3 --model llama    # Selective steering
   ```

### Phase 4: LLAMA + HumanEval (Week 4)

**Goal**: Complete experimental matrix

1. **Run Pipeline**
   ```bash
   python3 run.py phase 1 --dataset humaneval --model llama
   python3 run.py phase 3.8 --dataset humaneval --model llama
   python3 run.py phase 4.8 --dataset humaneval --model llama
   # (reuse features from LLAMA+MBPP)
   ```

### Phase 5: Analysis & Paper Updates (Week 5)

1. **Comparative Analysis**
   - Cross-dataset transfer (MBPP → HumanEval)
   - Cross-model comparison (Gemma vs LLAMA)
   - Feature universality analysis

2. **Paper Updates** (`iclr2026_conference.tex`)
   - Add HumanEval results to tables
   - Add LLAMA results to figures
   - Discuss generalization findings
   - Update abstract/conclusion

---

## Experimental Matrix Summary

| Experiment | Dataset | Model | Features From | Purpose |
|------------|---------|-------|--------------|---------|
| Baseline (current) | MBPP | Gemma | Phase 2.5 (MBPP+Gemma) | Original paper |
| Cross-dataset | HumanEval | Gemma | Phase 2.5 (MBPP+Gemma) | Dataset generalization |
| Cross-model | MBPP | LLAMA | Phase 2.5 (MBPP+LLAMA) | Model generalization |
| Full cross | HumanEval | LLAMA | Phase 2.5 (MBPP+LLAMA) | Combined generalization |

**Total runs needed**:
- Phase 1 (generation): 4 runs (2 datasets × 2 models)
- Phase 2.5 (features): 1 new run (LLAMA+MBPP) - reuse Gemma+MBPP
- Phase 4.5/4.6 (coefficients): 1 new run (LLAMA+MBPP)
- Validation phases (3.8, 4.8, etc.): 4 runs each

**Estimated compute**:
- Phase 1: ~2-6 hours each = 8-24 hours total
- Phase 2.5: ~30 min (LLAMA only)
- Phase 4.5/4.6: ~2-4 hours (LLAMA only)
- Validation: ~1-3 hours each × 4 = 4-12 hours per phase

**Total**: ~40-80 hours GPU time

---

## Testing Strategy

### Unit Tests
1. Test `get_phase_output_dir()` with all combinations
2. Test `DatasetLoader` for both MBPP and HumanEval
3. Test `PromptBuilder` outputs match expected format
4. Test backward compatibility (existing MBPP paths still work)

### Integration Tests
1. Run Phase 1 on small subset (10 problems) for each combination
2. Verify activation shapes match expected dimensions
3. Verify auto-discovery finds correct files
4. Verify feature loading works across phases

### Regression Tests
1. Re-run existing MBPP+Gemma pipeline
2. Compare outputs to previous results
3. Verify no performance degradation

---

## Risk Mitigation

### Risk 1: Breaking Existing Pipeline
**Mitigation**:
- Keep default behavior unchanged (MBPP + Gemma)
- Use feature flags for new functionality
- Maintain backward compatibility in file discovery

### Risk 2: LLAMA SAE Compatibility
**Mitigation**:
- Verify SAE format matches GemmaScope before starting
- Test SAE loading in isolation
- Have fallback to train new SAE if needed

### Risk 3: HumanEval Evaluation Complexity
**Mitigation**:
- Use existing HumanEval evaluation code from OpenAI
- Test on small subset first
- Handle edge cases (syntax errors, infinite loops)

### Risk 4: Feature Space Mismatch (Cross-Model)
**Mitigation**:
- Don't force cross-model feature reuse
- Select new features for LLAMA
- Document limitations in paper

---

## Open Questions

1. **Feature selection for LLAMA**: Should we run full feature selection (Phases 2.5, 2.10, 4.5, 4.6) or try to map Gemma features? **Recommendation: Run full selection**

2. **HumanEval test format**: Should we modify HumanEval to have assertion-style tests like MBPP, or keep function-call format? **Recommendation: Keep original format**

3. **Difficulty analysis for HumanEval**: HumanEval doesn't have complexity metrics. Skip Phase 0/0.1? **Recommendation: Skip or use simple metrics (solution length)**

4. **Instruction-tuned variants**: Should we test LLAMA-3-8B-Instruct vs LLAMA-3-8B-Base like we did with Gemma? **Recommendation: Yes, if time permits**

5. **Cross-model feature interpretation**: Should we attempt to align feature spaces between Gemma and LLAMA? **Recommendation: No, out of scope**

---

## Success Criteria

### Minimum Viable Extension
- [ ] Phase 1 runs on HumanEval + Gemma
- [ ] Phase 3.8 shows AUROC/F1 on HumanEval
- [ ] Phase 4.8 shows steering effects on HumanEval
- [ ] Paper includes HumanEval results table

### Full Extension
- [ ] All 4 combinations in experimental matrix complete
- [ ] LLAMA features selected via Phases 2.5, 2.10, 4.5, 4.6
- [ ] Comparative analysis shows cross-dataset/cross-model trends
- [ ] Paper includes comprehensive results section
- [ ] Addresses all reviewer concerns

---

## Timeline Estimate

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Core infrastructure | Config, auto-discovery, dataset abstraction |
| 2 | HumanEval + Gemma | Phase 1 working, validation results |
| 3 | LLAMA + MBPP | Feature selection, mechanistic analysis |
| 4 | LLAMA + HumanEval | Complete experimental matrix |
| 5 | Analysis & writing | Updated paper with all results |

**Total**: 5 weeks

**Critical path**: Week 1 (infrastructure) must be solid before parallelizing Week 2-4 tasks

---

## Next Steps

1. **Confirm design decisions** (this discussion)
2. **Set up development branch** (`git checkout -b humaneval-llama-support`)
3. **Implement Week 1 infrastructure**
4. **Test on small subset**
5. **Proceed with full runs**

---

## Appendix: File Modification Checklist

### Core Files to Modify
- [x] `common/config.py` - Add dataset/model configs
- [x] `common/utils.py` - Update auto-discovery
- [x] `common/prompt_utils.py` - Add HumanEval builder
- [x] `run.py` - Add CLI arguments
- [ ] `common/dataset_loader.py` - NEW FILE
- [ ] `common/evaluation.py` - NEW FILE (or update existing)

### Phase Files to Modify
- [ ] `phase0_difficulty_analysis/` - Support HumanEval (or skip)
- [ ] `phase0_1_problem_splitting/` - Support HumanEval (or skip)
- [ ] `phase1_simplified/runner.py` - Use DatasetLoader
- [ ] All validation phases (3.x, 4.x, 5.x, 6.3, 8.3) - Context-aware discovery

### Testing Files to Create
- [ ] `tests/test_dataset_loader.py`
- [ ] `tests/test_prompt_builder.py`
- [ ] `tests/test_auto_discovery.py`
- [ ] `tests/test_config.py`

---

## Notes from Discussion

### Confirmed Decisions (2025-11-21)

1. **Feature Reuse Strategy**: ✅ **Option C (Hybrid)**
   - HumanEval + Gemma: Reuse Gemma-MBPP features
   - LLAMA + MBPP: Select NEW LLAMA-MBPP features
   - LLAMA + HumanEval: Use LLAMA-MBPP features

2. **HumanEval Phase 0/0.1**: ✅ **Skip - Use all 164 problems**
   - No difficulty analysis needed
   - Use full dataset for validation
   - **ACTION ITEM**: Create HumanEval prompt transformation function to fit MBPP template

3. **Implementation Priority**: ✅ **HumanEval + Gemma First**
   - Easier, reuses existing infrastructure
   - Lower risk, faster validation

4. **LLAMA Variants**: ✅ **Test both base and instruct**
   - Base: `meta-llama/Llama-3.1-8B`
   - Instruct: `meta-llama/Llama-3.1-8B-Instruct`
   - SAE: `llama_scope_lxr_8x` (confirmed available)

5. **Output Directory Naming**: ✅ **Suffix-based (Option A)**
   - `data/phase1_0/` (MBPP + Gemma, default)
   - `data/phase1_0_humaneval/` (HumanEval + Gemma)
   - `data/phase1_0_llama/` (MBPP + LLAMA)
   - `data/phase1_0_humaneval_llama/` (HumanEval + LLAMA)

### Updated Experimental Matrix

| Experiment | Dataset | Model | Features From | Priority |
|------------|---------|-------|---------------|----------|
| Baseline | MBPP | Gemma-2-2B | Phase 2.5 (MBPP+Gemma) | ✅ Done |
| **Phase 1** | **HumanEval** | **Gemma-2-2B** | **Phase 2.5 (MBPP+Gemma)** | **🎯 First** |
| Phase 2 | MBPP | Llama-3.1-8B | Phase 2.5 (MBPP+LLAMA) - NEW | Second |
| Phase 3 | HumanEval | Llama-3.1-8B | Phase 2.5 (MBPP+LLAMA) | Third |
| Bonus | MBPP | Llama-3.1-8B-Instruct | Phase 7.x comparison | If time permits |

### Key Implementation Notes

1. **HumanEval Prompt Transformation**
   - HumanEval provides: `prompt` field with function signature + docstring
   - Need to transform to MBPP-style: problem description + test cases + code initiator
   - Implementation: Extract docstring as description, format tests as assertions
   - Location: `common/prompt_utils.py` - new `HumanEvalPromptBuilder` class

2. **LLAMA Model Details**
   - Model: `meta-llama/Llama-3.1-8B` (8B parameters)
   - Instruct: `meta-llama/Llama-3.1-8B-Instruct`
   - SAE: `llama_scope_lxr_8x` (LlamaScope SAE collection)
   - Layers: 32 (vs Gemma's 26)
   - Hidden size: 4096 (vs Gemma's 2304)

3. **Documentation Updates Needed**
   - Update `README.md` with new output directory naming convention
   - Update `CLAUDE.md` with multi-dataset/model usage examples
   - Update `iclr_reviewers_feedback.md` to reflect Llama-3.1-8B
