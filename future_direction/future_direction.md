# Developmental Interpretability Experiments for Code Correctness

## Overview

This document outlines additional experiments to extend my thesis "Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders" with a developmental interpretability perspective — understanding **when** and **how** code correctness features emerge during training.

---

## Core Research Question

> How do code correctness features develop over the course of training? Do they emerge suddenly at phase transitions, or gradually? Does the influence between correct and incorrect code samples change over time?

---

## Proposed Experiments

### Experiment 2: Feature Emergence Timeline

**Goal:** Track when code correctness directions become predictive during training

**Method:**
1. Use Pythia checkpoints (e.g., Pythia-160M at steps 1k, 5k, 10k, 30k, 70k, 143k)
2. For each checkpoint:
   - Train an SAE with same architecture
   - Search for code correctness directions using same methodology as main paper
   - Measure prediction accuracy of the direction
3. Plot: Prediction accuracy vs training step

**Expected outcome:** Either gradual emergence or sharp phase transition

**Related paper:** Wang et al. 2025b (rLLC tracking)

---

### Experiment 1: Influence Dynamics Between Code Samples

**Goal:** Test if influence between correct/incorrect code changes non-monotonically over training (sign flips, peaks at transitions)

**Method:**
1. Create dataset:
   - 50 correct code samples
   - 50 incorrect code samples (same problems, buggy solutions)
2. For each Pythia checkpoint:
   - Run RMSProp-SGLD sampling around checkpoint
   - Compute BIF matrix between all sample pairs
   - Aggregate: BIF(correct→correct), BIF(correct→incorrect), BIF(incorrect→incorrect)
3. Plot three curves over training steps

**What to look for:**
- Sign flips (positive → negative influence or vice versa)
- Peaks at specific checkpoints (indicates phase transition)
- Non-monotonic behavior

**Related paper:** Lee et al. 2025 (Influence Dynamics and Stagewise Data Attribution)

---

### Experiment 3: Susceptibility to Code Correctness

**Goal:** Measure which model components respond to shifts toward correct vs incorrect code

**Method:**
1. Define perturbation directions:
   - $p_{correct}$: distribution over correct code
   - $p_{incorrect}$: distribution over incorrect code
2. For each component (attention head, MLP layer, or SAE feature):
   - Compute susceptibility to each perturbation
   - $\chi = -\text{Cov}(\text{component observable}, \text{loss on perturbed data})$
3. Build susceptibility matrix: components × perturbation directions
4. Identify components that respond oppositely to correct vs incorrect

**Can be done:** At single checkpoint (structural) or across checkpoints (developmental)

**Related paper:** Baker et al. 2025 (Susceptibilities)

---

### Experiment 4: Developmental Visualization

**Goal:** Visualize emergence of code-related structure over training

**Method:**
1. Compute susceptibility matrix at each checkpoint
2. Apply UMAP to visualize component organization
3. Create animation or grid showing structural development

**Related paper:** Wang et al. 2025a (Embryology of a Language Model)

---

## Implementation Details

### Pythia Checkpoints

```python
from transformers import AutoModelForCausalLM

checkpoints = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
               1000, 2000, 4000, 8000, 16000, 32000, 
               64000, 100000, 143000]

for step in checkpoints:
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        revision=f"step{step}"
    )
```

### BIF Estimation (from Influence Dynamics paper)

Core equation:
$$\text{BIF}(z_i, \ell_j) = -\text{Cov}_{p(w|D)}(\ell_i(w), \ell_j(w))$$

Estimated via RMSProp-SGLD:
$$w_{s+1} = w_s - \frac{\hat{\epsilon}_s}{2}\left[\sum_{i} \nabla_w \ell_i(w_s) + \gamma(w_s - w^*)\right] + \mathcal{N}(0, \hat{\epsilon}_s)$$

Key hyperparameters (from paper):
- Step size $\epsilon$: 1e-6 to 1e-3
- Inverse temperature $\beta$: 100 to 10000  
- Localization strength $\gamma$: 100 to 1M
- Chain length: 200-1000 steps
- Number of chains: 4-8

### Susceptibility Estimation

$$\chi_{ij} = -\text{Cov}(\phi_i(w), \ell_j(w))$$

Where:
- $\phi_i(w)$ = observable for component $i$ (e.g., output norm of head)
- $\ell_j(w)$ = loss on data distribution $j$ (e.g., correct code)

---

## Dataset Requirements

### Code Samples for Influence/Susceptibility

Need paired correct/incorrect code:

| Type | Description | Source |
|------|-------------|--------|
| Correct | Working solutions to coding problems | HumanEval, MBPP |
| Incorrect | Buggy solutions to same problems | Mutated correct solutions, or LLM failures |

Minimum: 50 pairs
Recommended: 100-200 pairs for stable covariance estimates

---

## Expected Results

### If code correctness emerges at a phase transition:
- Sharp jump in prediction accuracy at specific checkpoint
- Peak in BIF(correct→correct) at same checkpoint
- Non-monotonic influence curves

### If code correctness emerges gradually:
- Smooth increase in prediction accuracy
- Monotonically decreasing influence (as loss decreases)
- No sharp peaks

---

## Priority Order

1. **Experiment 1 (Influence Dynamics)** — Most novel, directly tests developmental hypothesis
2. **Experiment 2 (Feature Emergence)** — Straightforward extension of existing work
3. **Experiment 3 (Susceptibility)** — Can do at single checkpoint first
4. **Experiment 4 (Visualization)** — Nice to have, depends on results from 1-3

---

## High-Relevance Papers (Detailed)

### [Lee et al. 2025 — Influence Dynamics and Stagewise Data Attribution](https://arxiv.org/abs/2510.12071)
- **arXiv:** 2510.12071
- **Key idea:** Influence between samples changes over training, peaks at phase transitions
- **Use for:** Experiment 1

**Detailed Methodology (BIF via RMSProp-SGLD):**
1. **Initialization**: Start from training checkpoint `w*`, prepare multiple MCMC chains (2-8 chains)
2. **SGLD Sampling**: For each step s, update parameters with noise:
   - `w_{s+1} = w_s - (ε̂_s/2) * [β*n/m * Σ∇ℓ_k + γ(w_s - w*)] + √ε̂_s * η_t`
   - RMSProp preconditioning: `ε̂_t = ε / (√V̂_t + α)` where `V_t = b*V_{t-1} + (1-b)*∇ℓ²`
   - Run for 200-1000 steps per chain
3. **Loss Computation**: At each SGLD draw, compute per-sample losses `L_{i,draw} = ℓ_i(w_s)` and store across all chains
4. **BIF Estimation**: Compute negative covariance between losses:
   - `BIF = (1/(CT-1)) * L * (I - 1/(CT)*11^T)^2 * Φ^T`
   - Use Pearson correlation for numerical stability in language models
5. **Key Hyperparameters**: ε ∈ [1e-7, 1e-2], β ∈ [10, 10000], γ ∈ [0.01, 1e6], C=2-8 chains, T=200-1000 steps
   - High γ and low ε → strongest correlation with ground truth (LOO experiments)

### [Wang et al. 2025b — Differentiation and Specialization of Attention Heads](https://arxiv.org/abs/2410.02984)
- **arXiv:** 2410.02984
- **Venue:** [ICLR 2025 Spotlight](https://iclr.cc/virtual/2025/poster/29600)
- **Key idea:** Refined LLC tracks individual component complexity over training
- **Use for:** Tracking which heads specialize to code

**Detailed Methodology (Refined LLC):**
1. **Parameter Decomposition**: Split parameters w* = (u*, v*) where V is component of interest (e.g., attention head)
2. **Gibbs Posterior Sampling**: Sample from tempered distribution `p(v) ∝ exp{-nβℓ'ₙ(u*, v) - (γ/2)||v - v*||²₂}`
   - β = inverse temperature (controls loss weight)
   - γ = localization strength (keeps samples near checkpoint)
3. **SGLD Sampling**: Generate posterior samples via `vₜ₊₁ = vₜ - (lr/2)∇ℓₙ(vₜ) + noise`
   - For wrLLC: Update only component V parameters, fix others
   - For drLLC: Use alternative data distribution q' (e.g., GitHub code vs all data)
4. **LLC Estimation**: Compute `λ̂(w*; V, q') = nβ × mean[ℓₙ(v) - ℓₙ(v*)]` over posterior samples
   - Lower λ → more degenerate geometry → simpler solution
5. **Developmental Analysis**: Track rLLC curves over training checkpoints to identify specialization patterns
   - Cluster heads by time-series similarity of rLLC trajectories
   - Compare drLLC on different data subsets to detect specialization

### [Baker et al. 2025 — Structural Inference via Susceptibilities](https://arxiv.org/abs/2504.18274)
- **arXiv:** 2504.18274
- **Key idea:** Perturb data distribution, measure component responses
- **Use for:** Experiment 3

**Detailed Methodology (Susceptibility Estimation):**
1. **Bayesian Framework**: Treat neural network as statistical mechanical system with posterior distribution over weights
2. **Data Perturbation**: Create shifted data distributions (e.g., Pile → GitHub code, Pile → legal text)
   - Define perturbation direction j (e.g., +GitHub, -general text)
3. **Observable Selection**: Choose component-localized observables φᵢ(w) for each component i
   - Examples: output norm of attention head, activation magnitude, layer output
4. **SGLD Sampling**: Generate posterior samples around checkpoint w*
   - Use local SGLD with 8 chains, 2000 draws per chain
   - Maintain samples near checkpoint with localization penalty
5. **Susceptibility Computation**: Estimate `χᵢⱼ = -Cov(φᵢ(w), ℓⱼ(w))` from SGLD samples
   - φᵢ(w) = observable for component i
   - ℓⱼ(w) = loss on perturbed distribution j
   - Negative covariance = sensitivity to distribution shift
6. **Per-Token Attribution**: Factorize susceptibility into signed per-token contributions
7. **Response Matrix Analysis**: Build matrix [χᵢⱼ] and apply low-rank decomposition (SVD/PCA)
   - Clusters reveal functional modules (e.g., induction heads, multigram circuits)

---

## Medium-Relevance Papers (For Future Exploration)

### [Wang et al. 2025a — Embryology of a Language Model](https://arxiv.org/abs/2508.00331)
- **arXiv:** 2508.00331
- **Key idea:** UMAP on susceptibility matrix reveals "body plan" emergence
- **Discovers:** Novel structures like "spacing fin" for counting spaces
- **Potential use:** Visualize code structure emergence

**Detailed Methodology (Embryology Visualization):**
1. **Susceptibility Matrix Computation**: For each training checkpoint, compute χᵢⱼ = -Cov(φᵢ(w), ℓⱼ(w))
   - i = component index (attention heads, MLP layers, etc.)
   - j = perturbation direction (data distribution shifts)
   - Creates C×P matrix (C components × P perturbations)
2. **Temporal Stacking**: Collect susceptibility matrices across T training checkpoints
   - Build 3D tensor: χ(t) for t ∈ [checkpoint₁, ..., checkpointₜ]
3. **UMAP Dimensionality Reduction**: Apply UMAP to component vectors
   - Each component i represented by its susceptibility profile across perturbations
   - UMAP projects from P-dimensional space → 2D or 3D for visualization
   - Preserves local structure: similar response profiles cluster together
4. **Body Plan Identification**: Visualize component trajectories through UMAP space over training
   - Color components by layer, head index, or function type
   - Identify stable clusters = functional modules (e.g., induction circuit)
5. **Novel Structure Discovery**: Look for unexpected clusters or trajectory patterns
   - Example: "spacing fin" = components specialized to space token counting
   - Validate discoveries via mechanistic probing and ablation studies
6. **Animation/Timeline**: Create developmental visualization showing cluster formation and separation over training

### [Carroll et al. 2025 — Dynamics of Transient Structure](https://arxiv.org/abs/2501.17745)
- **arXiv:** 2501.17745
- **Key idea:** Models can learn general solution first, then specialize (transient ridge phenomenon)
- **Potential use:** Does model learn "general programming" before "correctness"?

**Detailed Methodology (Transient Structure Detection):**
1. **Function Space Encoding**: At each checkpoint t, map model predictions to fixed dataset
   - Compute f(D,wₜ) ∈ ℝᴮᴷ (B sequences × K tokens)
   - Creates trajectory through function space over training
2. **Joint Trajectory PCA**: Aggregate trajectories across multiple task diversities M
   - Stack all checkpoints into matrix F_M, then vertically concatenate across diversities
   - Apply SVD to get principal components: F = UΛVᵀ
   - Project trajectories into v-dimensional subspace (v=2 or 4): γₘ(t) = πᵥ(f(D,wₜᴹ))
3. **Idealized Solution Projection**: Compute and project reference solutions
   - Ridge regression: t̂ₖ∞ = (XᵀX + σ²I)⁻¹XᵀY
   - dMMSE: Discrete minimum mean squared error over finite task set
   - Both appear as fixed points in PC space
4. **LLC Tracking**: Estimate complexity at each checkpoint
   - Sample from localized Gibbs posterior: p(w) ∝ exp{-nβℓₙ(w) - (γ/2)||w - w*||²}
   - Compute LLC: λ̂(w*) = nβ × [E[ℓₙ(w)] - ℓₙ(w*)]
   - Track λ across training to measure solution complexity evolution
5. **Transient Ridge Identification**: Monitor out-of-distribution (OOD) loss over training
   - Evaluate on data from q∞(S) (infinite diversity)
   - Find tᴹ_crit where OOD loss minimizes before increasing again
   - Non-monotonic OOD loss curve indicates transient ridge (general → specialized transition)
6. **Loss/Complexity Tradeoff Analysis**: Compare Δℓₙ·n vs Δλ·log(n)
   - At small n: ridge preferred (lower loss dominates)
   - At large n: dMMSE preferred (lower complexity dominates)
   - Crossover point predicts when model switches solutions

### [Urdshals & Urdshals 2025 — Structure Development in List-Sorting](https://arxiv.org/abs/2501.18666)
- **arXiv:** 2501.18666
- **Key idea:** Vocabulary-splitting and copy-suppression modes emerge in sorting task
- **Potential use:** Methodology template for algorithmic task interpretability

**Detailed Methodology (LLC + Mechanistic Analysis):**
1. **Circuit Decomposition**: Extract attention head circuits from trained model
   - QK circuit: W_QK^h = W_E W_Q^h (W_K^h)^T W_E^T (determines attention patterns)
   - OV circuit: W_OV^h = W_E W_V^h W_O^h W_U (controls value propagation)
   - Visualize as heatmaps to identify diagonal structures
2. **LLC Computation**: Estimate complexity at checkpoints
   - Compute LLC on validation data using parameter volume near loss minima
   - Lower LLC = broader basin = simpler solution
   - Track LLC evolution to identify when simplification occurs
3. **Development Stage Identification**: Monitor transitions through training
   - Initial learning: Steep loss decrease, rising LLC, forming diagonals
   - Head overlapping: Flat loss, constant LLC, overlapping OV regions across heads
   - Specialization: LLC decreases, heads split into non-overlapping regions
4. **Vocabulary-Splitting Detection**: Heads divide number range into non-overlapping regions
   - Each head's OV circuit shows positive diagonal in distinct vocabulary range
   - Quantify via region size and overlap metrics
   - Emerges naturally even without weight decay
5. **Copy-Suppression Detection**: Parallel heads with complementary roles
   - One head copies (positive OV diagonal), another suppresses (negative OV diagonal)
   - Both attend similarly (similar QK circuits)
   - Validate via ablation: measure accuracy and entropy changes
6. **Dataset Feature Analysis**: Relate structure to data properties
   - Compute gap distribution δᵢ = l_{i+1} - l_i in sorted lists
   - Vary mean gap δ̄ across datasets
   - Map δ̄ to emergent specialization modes (vocabulary-splitting vs copy-suppression)
7. **Weight Decay Effect**: Test simplification with/without regularization
   - Compare circuit emergence across WD strengths
   - Confirm vocabulary-splitting persists without WD (natural preference for simplicity)

---

## Low-Relevance Papers (Background Reading)

### [Chen et al. 2023 — Phase Transitions in Toy Model of Superposition](https://arxiv.org/abs/2310.06301)
- **arXiv:** 2310.06301
- **Key idea:** k-gon critical points determine phase transitions in TMS
- **Very theoretical:** Primarily useful for understanding SLT foundations
- **Why low relevance:** Toy model only, no language model experiments

**Detailed Methodology (TMS Phase Transitions via LLC):**
1. **Toy Model Setup**: Define TMS with n features, d hidden dimensions
   - Feature importance weights: wᵢ for i=1,...,n
   - Hidden representation: W ∈ ℝᵈˣⁿ (parameter matrix)
   - Reconstruction loss: minimize ||WᵀW - diag(w)||²
2. **k-gon Critical Points**: Identify symmetric geometric configurations
   - For d=2 (two hidden dimensions): regular k-gons in parameter space
   - k features arranged symmetrically around origin
   - Each k-gon represents a local/global minimum depending on k and feature weights
3. **LLC Computation for Each k-gon**: Calculate geometric complexity
   - Use analytic formula for TMS: λ can be computed in closed form
   - λ(k-gon) is a geometric invariant depending on k and feature configuration
   - Lower k (fewer features represented) → lower λ (simpler)
   - Higher k (more features) → higher λ (more complex)
4. **Phase Transition Identification**: Compare Bayesian posterior probabilities
   - At sample size n, posterior ratio: log(p(k₁-gon)/p(k₂-gon)) ≈ Δℓₙ·n + Δλ·log(n)
   - Transition occurs when ratios flip: smaller n favors lower loss, larger n favors lower complexity
   - Critical sample size: n* where (Δℓ)/(Δλ/n) ≈ log(n)
5. **SGD Trajectory Analysis**: Track optimization path through k-gon landscape
   - Models travel from high-loss, low-complexity (small k) to low-loss, high-complexity (large k)
   - Phase transitions manifest as jumps between k-gon basins during training
6. **Bayesian vs Dynamical Comparison**: Verify SLT predictions match SGD behavior
   - Bayesian posterior (via SLT) predicts which k-gon dominates at each sample size
   - SGD experiments confirm models converge to predicted k-gon configurations
   - Validates that LLC correctly characterizes phase transition boundaries

---

## Tools and Resources

### DevInterp Python Package
```bash
pip install devinterp
```
- LLC estimation
- SGLD sampling
- Documentation: https://devinterp.com

### Timaeus Research
- Website: https://timaeus.co/research
- Papers: https://devinterp.com/research

### Key References for SLT Background
- [Watanabe 2009 — "Algebraic Geometry and Statistical Learning Theory"](https://www.cambridge.org/core/books/algebraic-geometry-and-statistical-learning-theory/9C8FD1BDC817E2FC79117C7F41544A3A) (Cambridge University Press)
- [Watanabe 2018 — "Mathematical Theory of Bayesian Statistics"](https://www.routledge.com/Mathematical-Theory-of-Bayesian-Statistics/Watanabe/p/book/9780367734817) (CRC Press)
- [Lau et al. 2025 — "The Local Learning Coefficient"](https://arxiv.org/abs/2308.12108) (foundational LLC paper, [PMLR](https://proceedings.mlr.press/v258/lau25a.html))

---

## Timeline Estimate

| Experiment | Compute | Analysis | Total |
|------------|---------|----------|-------|
| Exp 1: Influence Dynamics | 3-5 days | 2 days | 5-7 days |
| Exp 2: Feature Emergence | 2-3 days | 1 day | 3-4 days |
| Exp 3: Susceptibility | 1-2 days | 1 day | 2-3 days |
| Exp 4: Visualization | 1 day | 1 day | 2 days |

**Total estimated time:** 2-3 weeks

---

## Notes

- Pythia models are small enough to run SGLD on consumer GPU (RTX 3090 sufficient for 160M)
- BIF estimation is embarrassingly parallel across checkpoints
- Start with fewer checkpoints (5-6) to validate methodology, then densify if interesting patterns emerge