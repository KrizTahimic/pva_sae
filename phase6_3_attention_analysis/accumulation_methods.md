# Phase 6.3: Attention Analysis Data Accumulation Methods

This document describes how attention data is accumulated and processed for each visualization in Phase 6.3.

## Core Accumulation Method

The `aggregate_to_3_bins()` method provides three different accumulation strategies:

1. **Raw Sums**: Direct summation of attention weights within each section
2. **Percentages**: Normalized to sum to 100% per attention head
3. **Length-Normalized**: Divided by section length for fair comparison

## Visualization-Specific Accumulation

### 1. Attention Distribution Stacked Bar Chart (`attention_distribution_chart.png`)
- **Accumulation Method**: Percentages
- **Process**:
  1. Extract attention at final prompt token position
  2. Sum attention within each section (problem/tests/solution)
  3. Convert to percentages: `(section_sum / total_attention) × 100`
  4. Average percentages across all heads for each task
  5. Average across all tasks for each condition (baseline/correct/incorrect)
- **Output**: Stacked bars showing percentage distribution across sections

### 2. Head-Level Attention Heatmap (`head_attention_heatmap.png`)
- **Accumulation Method**: Percentages
- **Process**:
  1. Extract attention at final prompt token for each head
  2. Sum attention within sections per head
  3. Convert to percentages per head
  4. Average across all tasks while preserving head identity
  5. Create matrix: [n_heads × 9] (3 sections × 3 conditions)
- **Output**: Heatmap showing each head's attention percentage to each section

### 3. Attention Change Delta Plots (`attention_delta_plots.png`)
- **Accumulation Method**: Percentage differences
- **Process**:
  1. Calculate percentage attention for baseline and steered conditions
  2. Compute change: `steered_percentage - baseline_percentage`
  3. Average changes across all heads for each task
  4. Average across all tasks
  5. Calculate standard deviation for error bars
- **Output**: Bar charts showing mean attention change with error bars

### 4. Statistical Significance Table (`significance_table.png`)
- **Accumulation Method**: Percentage differences (same as delta plots)
- **Process**:
  1. Use same differences as delta plots
  2. Perform paired t-tests on the differences
  3. Calculate mean change and p-values
  4. Determine significance (p < 0.05)
- **Output**: Table with statistical test results

### 5. Attention Transformation Scatter Plots (`attention_transformation_scatter_*.png`)
- **Accumulation Method**: Percentages
- **Process**:
  1. Extract percentage attention for each head in each task
  2. Create pairs: (baseline_percentage, steered_percentage)
  3. Plot all head-task pairs as points
  4. Separate plots for each section
- **Output**: Scatter plots with diagonal reference line (y=x)

### 6. Head-Specific Transformation Plots (`head_transformation_*.png`)
- **Accumulation Method**: Percentages per head
- **Process**:
  1. Extract percentage attention for specific head across tasks
  2. Create pairs per head: (baseline_percentage, steered_percentage)
  3. Plot each head separately with different colors
  4. Calculate trend lines per head
- **Output**: Scatter plots showing per-head transformations

### 7. Head Attention Change Bar Charts (`head_attention_change_bars_*.png`)
- **Accumulation Method**: Percentage differences per head
- **Process**:
  1. Calculate percentage attention for each head
  2. Compute change per head: `steered - baseline`
  3. Average across tasks while preserving head identity
  4. Create bars for each head's change
- **Output**: Bar charts showing change per head for each section

### 8. Comparative Head Changes (`comparative_head_changes.png`)
- **Accumulation Method**: Percentage differences per head
- **Process**:
  1. Calculate changes for both correct and incorrect steering
  2. Preserve head identity across both conditions
  3. Create side-by-side comparison for each section
  4. Use fixed y-axis scale (-15, 15) for comparison
- **Output**: Grid of bar charts (3 sections × 2 steering types)

## Key Notes

- **Final Token Focus**: All analyses focus on attention at the final prompt token position (where the model makes predictions)
- **Section Boundaries**: Determined by parsing problem structure (problem description, test cases, solution marker)
- **Fallback Strategy**: If boundaries are missing, sequence is divided into thirds
- **Head Averaging**: Most visualizations average across heads, except head-specific plots which preserve head identity
- **Task Averaging**: All visualizations average across the validation set tasks (n=60 in current run)