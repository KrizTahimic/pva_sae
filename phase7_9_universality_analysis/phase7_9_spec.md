# Phase 7.9: Universality Analysis - Cross-Model Comparison

## Executive Summary

1. **Purpose**: Comprehensive analysis of PVA feature universality across base and instruction-tuned models
2. **Method**: Statistical comparison of steering effectiveness between Gemma-2B and Gemma-2B-IT
3. **Data Sources**: Phases 3.5, 4.8, 7.3, and 7.6 results
4. **Deliverables**: Visualizations, metrics, LaTeX tables, and comprehensive report
5. **Key Question**: Do PVA features discovered in base models transfer to instruction-tuned variants?

## Phase Relationship

### Dependencies
- **Phase 3.5**: Base model validation baseline (temperature 0.0)
- **Phase 4.8**: Base model steering effect analysis
- **Phase 7.3**: Instruction-tuned model baseline generation
- **Phase 7.6**: Instruction-tuned model steering analysis

### Outputs Used by
- **Thesis Chapter**: Results and Discussion sections
- **Paper**: Main findings and visualizations

## Analysis Components

### 1. Baseline Performance Comparison
- Base model (Phase 3.5) vs Instruction-tuned (Phase 7.3)
- Pass rates at temperature 0.0
- Statistical significance testing

### 2. Steering Effectiveness Analysis
- Correction rates (incorrect → correct)
- Corruption rates (correct → incorrect)  
- Preservation rates
- Cross-model differences

### 3. Statistical Validation
- Binomial tests for steering effects
- Significance levels (p < 0.05)
- Effect size comparisons

### 4. Universality Assessment
- Feature transfer evaluation
- Effectiveness comparison
- Overall verdict on universality hypothesis

## Key Findings

### Baseline Performance
- **Base Model**: 29.90% pass rate (116/388 correct)
- **Instruction-Tuned**: 38.40% pass rate (149/388 correct)
- **Improvement**: +8.51% absolute, +28.5% relative

### Steering Effectiveness

#### Base Model (Phase 4.8)
- Correction Rate: 4.04%
- Corruption Rate: 64.66%
- Preservation Rate: 0%

#### Instruction-Tuned Model (Phase 7.6)
- Correction Rate: 2.93%
- Corruption Rate: 82.55%
- Preservation Rate: 91.28%

### Universality Verdict
- **Overall Assessment**: LIMITED transferability
- Features do not transfer effectively across architectures
- Instruction-tuning significantly alters internal representations

## Output Structure

```
data/phase7_9/
├── summary_report.md              # Comprehensive findings report
├── universality_comparison.png    # Multi-panel visualization
├── detailed_metrics.json          # All computed metrics
└── latex_tables.txt              # Paper-ready tables
```

## Visualization Components

### Main Figure (universality_comparison.png)
1. Baseline performance comparison bar chart
2. Steering correction rates comparison
3. Steering corruption rates comparison
4. Preservation rates comparison
5. Statistical significance heatmap
6. Steering coefficients comparison
7. Universality assessment summary
8. Key findings text panel

## Implementation

### Analysis Script
- `phase7_9_universality_analysis/universality_analysis.py`
- Loads data from phases 3.5, 4.8, 7.3, 7.6
- Calculates comprehensive metrics
- Generates visualizations and reports

### Usage
```bash
python3 phase7_9_universality_analysis/universality_analysis.py
```

## Research Implications

### If Features Transfer (Not the case)
- PVA features would be universal across model variants
- SAE-based interpretability would transfer to instruction-tuned models
- Steering could be applied to production models

### If Features Don't Transfer (Actual finding)
- Features are model-specific
- Instruction-tuning changes internal representations
- Need separate feature discovery for each model variant

### Mixed Results (Observed)
- Some effects transfer but with different patterns
- Corruption more effective, correction less effective
- Preservation dramatically improved in instruction-tuned model

## Key Insights

1. **Model-Specific Features**: PVA features appear to be architecture-dependent
2. **Instruction-Tuning Impact**: Significantly changes internal representations
3. **Steering Asymmetry**: Instruction-tuned models more resistant to correction but more vulnerable to corruption
4. **Future Work**: Requires separate feature discovery for each model variant

## Statistical Highlights

- All steering effects statistically significant (p < 0.05) except base model preservation
- Instruction-tuned model shows 91.28% preservation rate vs 0% for base model
- Corruption rate increased by 17.9% in instruction-tuned model
- Correction rate slightly decreased (-1.1%) in instruction-tuned model

## Conclusion

Phase 7.9 provides critical evidence that PVA features discovered through SAE analysis exhibit **limited transferability** across model architectures. While steering effects remain statistically significant in both models, the patterns differ substantially, indicating that interpretability insights are model-specific and that instruction-tuning fundamentally alters the internal feature landscape relevant to code generation tasks.

This finding has important implications for the deployment of interpretability-based interventions in production systems, suggesting that feature discovery and validation must be performed separately for each model variant.