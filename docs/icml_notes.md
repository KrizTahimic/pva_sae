# ICML Paper Notes

Documentation of methodological improvements and insights for writing the ICML paper.

---

## Improvements Over sae_entities (Ferrando et al. 2024)

### 1. Enhanced Control Feature Selection (Phase 4.10)

**sae_entities approach:**
- Filters features where `abs(min_max_scores) > 0.0` (exact zero discrimination)
- Randomly samples from remaining features
- No activation frequency filtering

**Our improvement:**
- Uses configurable `separation_threshold` (default: 0.01) for near-zero discrimination
- Adds `min_activation_freq` filter (default: 0.001) to exclude dead/rare features
- More robust control selection that avoids statistically unreliable features

**Why this matters:**
- Dead features (never activate) are useless as controls
- Rare features (activate once or twice) give unreliable steering baselines
- Our dual-filter approach ensures control features are both non-discriminative AND active

**Code reference:** `phase4_10_zero_discrimination/zero_discrimination_selector.py`

### 2. [Add more improvements as discovered]

---

## Key Differences from Inspiration Paper

| Aspect | sae_entities | sae_code_correctness | Notes |
|--------|-------------|---------|-------|
| Control feature filtering | `> 0.0` hard threshold | `< 0.01` + activation freq | More robust |
| Pile filtering | >2% threshold | >2% threshold | Same |
| SAE architecture | GemmaScope | GemmaScope + LlamaScope | Multi-model |
| Dataset | Entity recognition | Code correctness (MBPP/HumanEval) | Different domain |
| [Add more rows] | | | |

---

## Insights for Paper Writing

[Add insights as they come up during development]

---

## Reviewer Feedback Notes (from ICLR)

Track how we address reviewer concerns:

| Reviewer | Concern | Our Response | Status |
|----------|---------|--------------|--------|
| 7JAK | Top-10 features table | [TODO] | Pending |
| jwL5 | Feature-Selection Landscape | [TODO] | Pending |
| RXZd | Selective steering | Phase 8 implemented | Done |
| vRko | Multi-model validation | LLAMA + HumanEval planned | Pending |
| [Add more] | | | |

---

## Terminology Decisions

Document terminology choices for consistency:

| Term | Definition | Alternative considered |
|------|------------|----------------------|
| Latent | SAE hidden unit | Feature (too generic) |
| Separation score | mean(correct) - mean(incorrect) | Discrimination score |
| Baseline passed | Original generation was correct | test_passed, is_correct |
| [Add more] | | |
