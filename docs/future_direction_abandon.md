# Future Direction Experiments: Extending Code Correctness Interpretability

## Thesis Summary

This document proposes experiments to extend the thesis "Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders" using developmental interpretability methodologies.

**Key Thesis Findings:**
- **Four code correctness directions** identified via t-statistics (predictors) and separation scores (steering)
- **Asymmetric predictability**: Incorrect-predicting directions reliably predict errors (F1=0.821), while correct-predicting fails (F1=0.504)
- **Interpretation**: Models encode incorrect code as predictable anomalies but lack explicit representations for correctness
- **Attention insight**: Test cases drive generation (27.29 pp differential), not problem descriptions
- **Persistence**: Mechanisms learned in pre-training persist through instruction-tuning
- **Model used**: Gemma-2-2b with GemmaScope SAEs, MBPP dataset

---

## Model Constraints

| Model | Checkpoints | SAEs | MBPP | Use Case |
|-------|-------------|------|------|----------|
| **Pythia** | ✅ 154/model | ✅ Multiple | ✅ 17.8% | **Developmental experiments** |
| OLMo | ✅ 500+/model | ❌ None | ✅ 60.2% | Would need to train SAEs |
| Gemma 2 | ❌ Final only | ✅ 400+ SAEs | ✅ Strong | Current thesis model |
| Llama 3.x | ❌ Final only | ✅ Llama Scope | ✅ 87-88% | Cross-model validation |

**Key insight**: Pythia is the ONLY model with checkpoints + SAEs + MBPP, making it essential for developmental experiments.

---

## Proposed Experiments

### Experiment 1: When Do Code Correctness Directions Emerge?
**Priority**: HIGH — Directly extends thesis with developmental perspective

**Research Question**: Do code correctness directions emerge gradually or suddenly during training? At what checkpoint do incorrect-predicting features become reliable (F1>0.75)?

**Method** (Adapted from Wang et al. 2025b — rLLC):
1. **Checkpoint Selection**: Use Pythia-160M or Pythia-410M at checkpoints [1k, 2k, 4k, 8k, 16k, 32k, 64k, 100k, 143k]
2. **SAE Application**: Load EleutherAI SAEs (`sae-pythia-160m-32x`) at each checkpoint
3. **Direction Search**: Apply thesis methodology (t-statistic, separation score) at each checkpoint
4. **Track Emergence**:
   - Plot t-statistic of best incorrect-predicting feature vs training step
   - Plot F1 score vs training step
   - Identify if emergence is gradual (smooth curve) or sudden (sharp transition)
5. **LLC Tracking**: Estimate LLC at each checkpoint for the prediction feature
   - Sharp LLC change → phase transition
   - Gradual LLC change → continuous learning

**Expected Outcomes**:
- Timeline showing when F1>0.75 threshold is reached
- Evidence for gradual emergence vs phase transition
- Correlation between LLC and prediction accuracy

**Compute**: 2-3 days (9 checkpoints × SAE analysis)

**Connection to Thesis**: Extends the finding that incorrect-predicting achieves F1=0.821 by asking *when* during training this capability emerges.

---

### Experiment 2: Influence Dynamics Between Correct/Incorrect Code
**Priority**: HIGH — Most novel, directly tests developmental hypothesis

**Research Question**: Does the influence between correct and incorrect code samples change non-monotonically over training? Do sign flips or peaks occur at phase transitions?

**Method** (Adapted from Lee et al. 2025 — BIF):
1. **Create Dataset**:
   - 50 correct MBPP solutions from Pythia
   - 50 incorrect MBPP solutions (same problems, buggy outputs)
   - Label pairs for computing BIF matrix
2. **For Each Pythia Checkpoint** [1k, 5k, 10k, 30k, 70k, 143k]:
   - Initialize 4-8 MCMC chains around checkpoint w*
   - Run RMSProp-SGLD: `w_{s+1} = w_s - (ε̂_s/2) * [β*n/m * Σ∇ℓ_k + γ(w_s - w*)] + √ε̂_s * η_t`
   - Hyperparameters: ε ∈ [1e-7, 1e-2], β ∈ [10, 10000], γ ∈ [0.01, 1e6], T=200-1000 steps
3. **Compute BIF Matrix**: For all sample pairs
   - `BIF(zᵢ, ℓⱼ) = -Cov(ℓᵢ(w), ℓⱼ(w))` over SGLD samples
4. **Aggregate by Type**:
   - BIF(correct→correct): How correct samples influence each other
   - BIF(correct→incorrect): Cross-influence
   - BIF(incorrect→incorrect): How incorrect samples influence each other
5. **Plot Three Curves** over training steps
   - Look for: Sign flips, peaks at specific checkpoints, non-monotonic behavior

**Expected Outcomes**:
- If phase transition: Sharp peak in BIF at specific checkpoint
- If gradual: Monotonically decreasing influence
- Identifies when model learns to differentiate correct/incorrect

**Compute**: 5-7 days (SGLD sampling is expensive)

**Connection to Thesis**: Tests whether the asymmetry (incorrect-predicting works, correct-predicting fails) develops suddenly at a phase transition or gradually over training.

---

### Experiment 3: Susceptibility of Model Components to Code Correctness
**Priority**: MEDIUM — Structural analysis at single or multiple checkpoints

**Research Question**: Which model components (attention heads, MLP layers) respond most strongly to shifts toward correct vs incorrect code?

**Method** (Adapted from Baker et al. 2025 — Susceptibilities):
1. **Define Perturbation Directions**:
   - p_correct: Distribution over correct MBPP solutions
   - p_incorrect: Distribution over incorrect MBPP solutions
   - p_general: Pile-10k baseline distribution
2. **Choose Observables** for each component i:
   - Attention heads: Output norm, attention entropy
   - MLP layers: Activation magnitude
   - Residual stream: L2 norm at each layer
3. **SGLD Sampling**: Generate posterior samples (8 chains, 2000 draws)
4. **Compute Susceptibility Matrix**:
   - `χᵢⱼ = -Cov(φᵢ(w), ℓⱼ(w))` for component i, perturbation j
5. **Analysis**:
   - Build response matrix [χᵢⱼ]
   - Apply SVD/PCA to identify functional clusters
   - Find components with opposite responses to correct vs incorrect

**Expected Outcomes**:
- Identification of which layers encode code correctness
- Validation of attention analysis finding (test cases > problem descriptions)
- Discovery of "code correctness circuits" via component clustering

**Compute**: 2-3 days at single checkpoint, 1-2 weeks across checkpoints

**Connection to Thesis**:
- Validates that L16, L19, L25 (thesis-identified important layers) show highest susceptibility to correctness shifts
- Provides alternative validation of attention findings (27.29 pp test case differential)

---

### Experiment 4: Cross-Model Code Correctness Direction Transfer
**Priority**: MEDIUM — Addresses thesis limitation (only Gemma-2-2b studied)

**Research Question**: Do code correctness directions transfer across model families? Can directions from Gemma predict errors in Llama?

**Method**:
1. **Source Models**: Gemma-2-2b (thesis), Pythia-410M, Llama-3.1-8B
2. **For Each Model**:
   - Generate MBPP solutions (same prompt template)
   - Capture residual stream activations at final prompt token
   - Apply SAE decomposition (GemmaScope, EleutherAI SAEs, Llama Scope)
   - Identify code correctness directions using t-statistic method
3. **Transfer Test**:
   - Take Gemma's incorrect-predicting direction
   - Project onto Llama's SAE space (via concept matching or linear probing)
   - Measure F1 score of transferred direction
4. **Correlation Analysis**:
   - Compare top-k features across models
   - Measure overlap in logit lens patterns
   - Test if similar semantic concepts activate across architectures

**Expected Outcomes**:
- High transfer → Universal code correctness mechanism
- Low transfer → Model-specific representations
- Partial transfer → Shared concepts with architecture-specific implementations

**Compute**: 3-5 days

**Connection to Thesis**: Directly addresses the limitation "analysis focused exclusively on Gemma 2 2B" from thesis conclusions.

---

### Experiment 5: Visualizing Code Correctness Body Plan Emergence
**Priority**: LOW — Depends on results from Experiments 1-3

**Research Question**: How does the "body plan" of code correctness mechanisms develop during training?

**Method** (Adapted from Wang et al. 2025a — Embryology):
1. **Build Susceptibility Matrices** at 10+ Pythia checkpoints
2. **Temporal Stacking**: Create 3D tensor χ(t) over checkpoints
3. **UMAP Dimensionality Reduction**:
   - Represent each component by its susceptibility profile
   - Project to 2D space
4. **Visualization**:
   - Color components by layer, head index, or function type
   - Animate trajectories through UMAP space over training
5. **Novel Structure Discovery**:
   - Look for unexpected clusters (analogous to "spacing fin" discovery)
   - Identify "code correctness circuit" formation

**Expected Outcomes**:
- Visual timeline of when L16, L19, L25 features emerge and differentiate
- Developmental path from random initialization to specialized code correctness detection
- Potential discovery of novel circuit structures

**Compute**: 1-2 days for visualization (assumes susceptibility matrices computed from Exp 3)

**Connection to Thesis**: Visualizes the emergence of the layers (L16, L19, L25) that thesis identified as most important for code correctness.

---

### Experiment 6: Training SAEs on OLMo for High-Performance Developmental Analysis
**Priority**: LOW (High effort, high payoff) — Community contribution

**Research Question**: Can we create a second model family meeting all three criteria (checkpoints + SAEs + strong MBPP)?

**Method**:
1. **Train SAEs on OLMo-7B** using EleutherAI/sparsify library
   - Multiple checkpoints: [500B, 1T, 1.5T, 2T, 3T tokens]
   - Architecture: Match GemmaScope (JumpReLU, 16k latents)
2. **Validate SAEs**: Reconstruction accuracy, sparsity metrics
3. **Apply Thesis Methodology**: Find code correctness directions
4. **Developmental Analysis**: Compare emergence curves Pythia vs OLMo
5. **Release to Community**: HuggingFace upload

**Why Valuable**:
- OLMo achieves 60.2% MBPP (vs Pythia's 17.8%)
- Better MBPP → more meaningful correct/incorrect distinction
- Enables studying code correctness in higher-performing models
- Community contribution: fills ecosystem gap

**Compute**: 1-2 weeks for SAE training

**Connection to Thesis**: Creates infrastructure for validating thesis findings on a model with stronger code generation capabilities.

---

## Implementation Roadmap

| Phase | Experiments | Duration | Dependencies |
|-------|------------|----------|--------------|
| 1 | Exp 1 (Emergence Timeline) | 3-4 days | Pythia SAEs loaded |
| 2 | Exp 2 (Influence Dynamics) | 5-7 days | DevInterp library installed |
| 3 | Exp 3 (Susceptibility) | 2-3 days | Builds on Phase 1-2 |
| 4 | Exp 4 (Cross-Model Transfer) | 3-5 days | Multiple model SAEs |
| 5 | Exp 5 (Visualization) | 1-2 days | Depends on Phase 3 |
| 6 | Exp 6 (OLMo SAE Training) | 1-2 weeks | Optional, high effort |

**Total estimated time**: 3-5 weeks for Experiments 1-5

---

## Connection to Thesis Findings

| Thesis Finding | Extended By |
|----------------|-------------|
| Incorrect-predicting F1=0.821 | Exp 1: When does it emerge during training? |
| Asymmetry in prediction/steering | Exp 2: How does influence develop over checkpoints? |
| L16, L19, L25 important layers | Exp 3: Susceptibility validates layer importance |
| Mechanisms persist across fine-tuning | Exp 4: Do they persist across model families? |
| Test cases > problem descriptions | Exp 3: Susceptibility of attention heads to test input |
| Only Gemma-2-2b studied | Exp 4, 6: Cross-model validation |

---

## Key Resources Needed

1. **Models**: Pythia-160M/410M (HuggingFace), Llama-3.1-8B (optional)
2. **SAEs**: EleutherAI SAE collection, Llama Scope (optional)
3. **Libraries**:
   - DevInterp: `pip install devinterp` (LLC estimation, SGLD sampling)
   - SAELens: Existing in thesis codebase
4. **Compute**: GPU with 24GB+ VRAM (existing setup should work)

---

## High-Relevance Papers (Detailed Methodologies)

### [Lee et al. 2025 — Influence Dynamics and Stagewise Data Attribution](https://arxiv.org/abs/2510.12071)
- **arXiv:** 2510.12071
- **Key idea:** Influence between samples changes over training, peaks at phase transitions
- **Use for:** Experiment 2

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
- **Use for:** Experiment 1

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
- **Use for:** Experiment 5

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

## Notes

- Pythia models are small enough to run SGLD on consumer GPU (RTX 3090 sufficient for 160M)
- BIF estimation is embarrassingly parallel across checkpoints
- Start with fewer checkpoints (5-6) to validate methodology, then densify if interesting patterns emerge
- Experiments 1-4 can be started immediately with existing resources
- Experiment 6 (OLMo SAE training) is optional but would be a valuable community contribution
