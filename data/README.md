---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - code
  - sparse-autoencoder
  - interpretability
  - mechanistic-interpretability
  - gemma
size_categories:
  - 1GB<n<10GB
---

# PVA-SAE Data

Experiment data for "Program Value Attribution using Sparse Autoencoders"

This dataset contains all intermediate and final results from analyzing how language models internally represent program correctness using Sparse Autoencoders (SAEs).

## Contents

### Data Preparation (Phase 0)
- **phase0/**: MBPP problem complexity analysis (cyclomatic complexity metrics)
- **phase0_1/**: Problem splits (50% SAE analysis, 10% hyperparams, 40% validation)
- **phase0_2_humaneval/**: HumanEval dataset preprocessed to MBPP format
- **phase0_3_humaneval/**: HumanEval import statement mappings

### Feature Discovery (Phases 1-2)
- **phase1_0/**: Generated code solutions + extracted activations (Gemma-2B)
- **phase1_0_llama/**: Alternative model (LLAMA-3.1-8B) activations
- **phase2_2/**: Pile baseline activations (160K samples for filtering general language features)
- **phase2_5/**: SAE analysis results with separation scores
- **phase2_10/**: T-statistic selected features (Welch's t-test)

### Statistical Validation (Phases 3)
- **phase3_5/**: Temperature robustness testing across [0.0, 0.3, 0.6, 0.9, 1.2]
- **phase3_6/**: Hyperparameter tuning set results
- **phase3_8/**: AUROC and F1 evaluation metrics + ROC curves
- **phase3_10/**: Temperature-based AUROC analysis
- **phase3_12/**: Difficulty-stratified AUROC analysis

### Causal Validation (Phases 4)
- **phase4_5/**: Steering coefficient selection (coarse-to-fine search)
- **phase4_6/**: Golden section refinement results
- **phase4_8/**: Steering effect analysis (correction/corruption rates)
- **phase4_10/**: Zero-discrimination feature selection (control)
- **phase4_14/**: Binomial significance testing

### Weight Orthogonalization (Phases 5)
- **phase5_3/**: Permanent weight modification results
- **phase5_6/**: Zero-discrimination control experiment
- **phase5_9/**: Statistical significance validation

### Mechanistic Analysis (Phase 6)
- **phase6_3/**: Attention pattern analysis comparing baseline vs steered

### Model Comparison (Phases 7)
- **phase7_3/**: Instruction-tuned model (gemma-2-2b-it) baseline
- **phase7_6/**: Instruction-tuned steering analysis
- **phase7_9/**: Cross-model universality analysis
- **phase7_12/**: Comparative AUROC/F1 evaluation

### Selective Steering (Phases 8)
- **phase8_1/**: Percentile threshold calculation
- **phase8_2/**: Threshold optimization results
- **phase8_3/**: Selective steering analysis

## File Types

| Extension | Description | Typical Size |
|-----------|-------------|--------------|
| `.npz` | NumPy compressed arrays (activations) | 1-10 KB each |
| `.parquet` | Apache Parquet (structured datasets) | 10-500 KB |
| `.json` | JSON (metrics, configs, results) | 1-200 KB |
| `.png` | PNG images (plots, visualizations) | 20-230 KB |

## Usage

### Download All Data

```bash
huggingface-cli download kriztahimic/pva-sae-data --local-dir ./data --repo-type dataset
```

### Download Specific Phase

```bash
# Only steering results
huggingface-cli download kriztahimic/pva-sae-data \
    --local-dir ./data \
    --include "phase4_8/*" \
    --repo-type dataset
```

### Use with Code

Clone the analysis code from GitHub:
```bash
git clone https://github.com/KrizTahimic/pva_sae
cd pva_sae
huggingface-cli download kriztahimic/pva-sae-data --local-dir ./data --repo-type dataset
python3 run.py phase 3.8 --viz-only  # Regenerate plots
```

## Models Used

- **Gemma-2-2B** (`google/gemma-2-2b`): Primary model
- **Gemma-2-2B-IT** (`google/gemma-2-2b-it`): Instruction-tuned variant
- **GemmaScope 16k**: Sparse Autoencoder from `google/gemma-scope-2b-pt-res`

## Datasets Used

- **MBPP**: Mostly Basic Programming Problems (974 tasks)
- **HumanEval**: OpenAI hand-written Python problems (164 tasks)

## Citation

```bibtex
@misc{pvasae2025,
  title={Program Value Attribution using Sparse Autoencoders},
  author={Tahimic, Kriz},
  year={2025},
  howpublished={\url{https://github.com/KrizTahimic/pva_sae}}
}
```

## License

Apache 2.0
