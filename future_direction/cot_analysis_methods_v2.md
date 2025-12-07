# Mechanistic Analysis of Chain-of-Thought in Reasoning Models

## Overview

This document outlines methodological approaches for extending SAE-based code correctness analysis to reasoning models. These methods are grounded in recent research on CoT faithfulness and reasoning model interpretability.

**Core insight:** You analyze **activation trajectories** (time series), not text directly. The CoT text is context; the quantitative object is feature activations over token positions.

---

## Method 1: Trajectory Comparison

**Goal:** Compare how error-predicting features evolve across correct vs incorrect final outputs.

### Inspiration Papers

- **Lanham et al. (2023)** - "Measuring Faithfulness in Chain-of-Thought Reasoning"
  - Paper: https://arxiv.org/abs/2307.13702
  - Anthropic page: https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning
  - *Note: No public code release, but methodology is well-documented in paper*

- **Turpin et al. (2023)** - "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting" (NeurIPS 2023)
  - Paper: https://arxiv.org/abs/2305.04388
  - **GitHub:** https://github.com/milesaturpin/cot-unfaithfulness
  - Key methodology: Intervening on CoT (adding mistakes, paraphrasing, early truncation) to test faithfulness

### Setup
1. Run N problems (e.g., 500 MBPP) through a reasoning model
2. Record error-feature activation at every token position
3. Label trajectories by final outcome (correct/incorrect)

### Analysis

```python
# Normalize to percentage through CoT (handles variable length)
correct_trajectories = []    # shape: (n_correct, 100)  
incorrect_trajectories = []  # shape: (n_incorrect, 100)

for cot in all_cots:
    activations = get_error_feature_at_each_token(cot)
    normalized = interpolate_to_fixed_length(activations, length=100)
    if cot.final_correct:
        correct_trajectories.append(normalized)
    else:
        incorrect_trajectories.append(normalized)

# Plot mean ± std for each group
plt.plot(np.mean(correct_trajectories, axis=0), label='Correct')
plt.plot(np.mean(incorrect_trajectories, axis=0), label='Incorrect')
plt.fill_between(...)  # confidence intervals
```

### What to Look For
- Do trajectories diverge? At what percentage point?
- Is there a threshold that predicts final outcome?
- Do incorrect trajectories show monotonic increase, or sudden spikes?

---

## Method 2: Event-Triggered Analysis

**Goal:** Analyze feature behavior around specific reasoning events (backtracking, verification, etc.)

### Inspiration Papers

- **Venhoff et al. (2025)** - "Understanding Reasoning in Thinking Language Models via Steering Vectors" (ICLR 2025 Workshop)
  - Paper: https://arxiv.org/abs/2506.18167
  - **GitHub:** https://github.com/cvenhoff/steering-thinking-llms
  - Key finding: Identified steering vectors for backtracking, uncertainty estimation, example testing in DeepSeek-R1-Distill models

- **Goodfire (2025)** - "Under the Hood of a Reasoning Model" - SAEs on DeepSeek R1
  - Blog: https://www.goodfire.ai/research/under-the-hood-of-a-reasoning-model
  - **GitHub:** https://github.com/goodfire-ai/r1-interpretability
  - HuggingFace SAEs: https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37
  - Key contribution: First public SAEs trained on a 671B reasoning model, with features for backtracking behavior

### Step 1: Detect Events

Based on Venhoff et al.'s taxonomy of reasoning behaviors:

```python
# Behavioral categories from Venhoff et al. (2025)
BACKTRACK_PHRASES = [
    "wait", "actually", "let me reconsider", 
    "that's wrong", "hmm", "no,", "I made a mistake",
    "let me try again", "that doesn't work"
]
UNCERTAINTY_PHRASES = [
    "I'm not sure", "might be", "possibly", "maybe",
    "I think", "could be", "let me check"
]
EXAMPLE_TESTING_PHRASES = [
    "let me test", "for example", "if we try",
    "let's verify with", "checking with"
]

def find_events(cot_text, cot_tokens):
    events = []
    for i, token in enumerate(cot_tokens):
        window = get_text_window(cot_text, i, window_size=20)
        
        if any(phrase in window.lower() for phrase in BACKTRACK_PHRASES):
            events.append({'type': 'backtrack', 'position': i})
        elif any(phrase in window.lower() for phrase in UNCERTAINTY_PHRASES):
            events.append({'type': 'uncertainty', 'position': i})
        elif any(phrase in window.lower() for phrase in EXAMPLE_TESTING_PHRASES):
            events.append({'type': 'example_testing', 'position': i})
    
    return events
```

### Step 2: Extract Windows Around Events

```python
WINDOW_SIZE = 10  # tokens before and after

def extract_event_windows(activations, events):
    windows = {'backtrack': [], 'uncertainty': [], 'example_testing': []}
    
    for event in events:
        pos = event['position']
        start = max(0, pos - WINDOW_SIZE)
        end = min(len(activations), pos + WINDOW_SIZE)
        
        window = activations[start:end]
        window = pad_to_length(window, WINDOW_SIZE * 2)
        
        windows[event['type']].append(window)
    
    return windows
```

### What to Look For
- Does error feature spike *before* backtracking? (Model detects error internally → verbalizes)
- Does error feature drop *after* backtracking? (Successful correction)
- Do patterns differ for ultimately-correct vs ultimately-incorrect outputs?

---

## Method 3: LLM-Assisted Segmentation

**Goal:** Analyze feature activations per semantic phase of reasoning.

### Inspiration Papers

- **Arcuschin et al. (2025)** - "Base Models Know How to Reason, Thinking Models Learn When"
  - Paper: https://arxiv.org/abs/2510.07364
  - Key methodology: Using Top-K SAEs to cluster sentence-level activations and create reasoning taxonomies
  - Finding: Thinking models primarily learn *when* to deploy reasoning mechanisms, not new mechanisms

- **Paul et al. (2024)** - "Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning" (EMNLP Findings)
  - Paper: https://arxiv.org/abs/2402.13950
  - **GitHub:** https://github.com/debjitpaul/Causal_CoT
  - Key methodology: Causal mediation analysis on CoT reasoning steps

### Step 1: Segment CoTs with LLM

```python
SEGMENTATION_PROMPT = """
Segment this reasoning trace into phases. Label each phase as one of:
- RESTATE: Restating or understanding the problem
- PLAN: Planning the approach or algorithm
- IMPLEMENT: Writing or describing code logic
- VERIFY: Checking, testing, or validating
- BACKTRACK: Reconsidering or correcting previous steps
- OTHER: Anything else

Reasoning trace:
{cot_text}

Return as JSON array:
[{{"phase": "RESTATE", "start_char": 0, "end_char": 145}}, ...]
"""

def segment_cot(cot_text, llm_client):
    response = llm_client.generate(
        SEGMENTATION_PROMPT.format(cot_text=cot_text)
    )
    return json.loads(response)
```

### Step 2: Compute Per-Phase Statistics

```python
def analyze_phases(cot, segments, activations):
    phase_stats = []
    
    for seg in segments:
        start_tok, end_tok = char_to_token_positions(seg, cot.tokens)
        phase_activations = activations[start_tok:end_tok]
        
        phase_stats.append({
            'phase': seg['phase'],
            'mean_activation': np.mean(phase_activations),
            'max_activation': np.max(phase_activations),
            'activation_trend': np.polyfit(range(len(phase_activations)), 
                                           phase_activations, 1)[0],
            'final_correct': cot.final_correct
        })
    
    return phase_stats
```

---

## Method 4: Critical Point Detection

**Goal:** Find the token position where correct/incorrect trajectories diverge—the "point of no return."

### Inspiration Papers

- **Lanham et al. (2023)** - "Early Answering" experiment
  - Truncated CoT at various points to test when model "commits" to an answer
  - Finding: Models often reach correct answer before completing full CoT

### Statistical Approach

```python
from scipy import stats

def find_divergence_point(correct_trajectories, incorrect_trajectories, 
                          alpha=0.01, min_effect_size=0.3):
    """
    Find earliest normalized position where distributions significantly differ.
    """
    n_positions = correct_trajectories.shape[1]
    
    for t in range(n_positions):
        correct_at_t = correct_trajectories[:, t]
        incorrect_at_t = incorrect_trajectories[:, t]
        
        # Mann-Whitney U test (non-parametric)
        statistic, pvalue = stats.mannwhitneyu(
            correct_at_t, incorrect_at_t, alternative='two-sided'
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(correct_at_t) + np.var(incorrect_at_t)) / 2)
        effect_size = (np.mean(incorrect_at_t) - np.mean(correct_at_t)) / pooled_std
        
        if pvalue < alpha and abs(effect_size) > min_effect_size:
            return t, pvalue, effect_size
    
    return None, None, None
```

---

## Method 5: Faithfulness Analysis

**Goal:** Test whether CoT accurately reflects internal model state (error features).

### Inspiration Papers

- **Turpin et al. (2023)** - Core paper on CoT unfaithfulness
  - **GitHub:** https://github.com/milesaturpin/cot-unfaithfulness
  - Key finding: CoT can be heavily biased without mentioning the bias in explanations

- **Lyu et al. (2023)** - "Faithful Chain-of-Thought Reasoning" (IJCNLP-AACL 2023)
  - Paper: https://arxiv.org/abs/2301.13379
  - **GitHub:** https://github.com/veronica320/Faithful-COT
  - Methodology: Two-stage approach (Translation → Problem Solving) for guaranteed faithfulness

### Build Alignment Matrix

```python
def classify_expressed_confidence(text_window):
    """Classify what the CoT expresses about confidence/correctness."""
    text_lower = text_window.lower()
    
    uncertain_phrases = ["not sure", "might be wrong", "maybe", "i think",
                        "let me check", "could be", "possibly"]
    confident_phrases = ["clearly", "obviously", "this works", "correct",
                        "this is right", "definitely", "simple"]
    
    uncertain_count = sum(1 for p in uncertain_phrases if p in text_lower)
    confident_count = sum(1 for p in confident_phrases if p in text_lower)
    
    if confident_count > uncertain_count:
        return "confident"
    elif uncertain_count > confident_count:
        return "uncertain"
    return "neutral"

def analyze_faithfulness(cot, activations, error_threshold=0.5):
    """
    Compare expressed confidence vs internal error feature.
    """
    results = []
    window_size = 50  # characters
    
    for i in range(0, len(cot.text), window_size):
        text_window = cot.text[i:i+window_size]
        token_range = char_to_token_range(i, i+window_size, cot)
        
        expressed = classify_expressed_confidence(text_window)
        internal_error = np.mean(activations[token_range[0]:token_range[1]])
        
        internal_state = "error_high" if internal_error > error_threshold else "error_low"
        
        # Determine alignment
        if expressed == "confident" and internal_state == "error_high":
            alignment = "unfaithful_overconfident"
        elif expressed == "uncertain" and internal_state == "error_low":
            alignment = "unfaithful_overcautious"
        else:
            alignment = "aligned"
        
        results.append({
            'expressed': expressed,
            'internal_error': internal_error,
            'internal_state': internal_state,
            'alignment': alignment
        })
    
    return results
```

### Confusion Matrix

|  | Error Feature LOW | Error Feature HIGH |
|--|-------------------|-------------------|
| **CoT: "confident"** | Aligned ✓ | Unfaithful (overconfident) ⚠️ |
| **CoT: "uncertain"** | Unfaithful (overcautious) | Aligned ✓ |

---

## Method 6: SAE Feature Analysis on Reasoning Models

**Goal:** Use sparse autoencoders to identify interpretable features in reasoning traces.

### Key Resources

- **Goodfire R1 SAEs**
  - **GitHub:** https://github.com/goodfire-ai/r1-interpretability
  - HuggingFace: https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37
  - Includes: General reasoning SAE + Math-specific SAE
  - Colab notebooks for inference and database querying

- **LlamaScope SAEs** (for distilled models)
  - HuggingFace: https://huggingface.co/fnlp/Llama-Scope
  - Can be applied to DeepSeek-R1-Distill-Llama-8B

### Loading Goodfire SAEs

```python
from sae import load_math_sae
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="Goodfire/DeepSeek-R1-SAE-l37",
    filename="math/DeepSeek-R1-SAE-l37.pt",
    repo_type="model"
)

device = "cpu"
math_sae = load_math_sae(file_path, device)
```

---

## Method 7: Steering Vectors for Reasoning Behaviors

**Goal:** Extract and apply steering vectors to control reasoning behaviors.

### Inspiration Papers

- **Venhoff et al. (2025)** - Steering vectors for thinking LLMs
  - **GitHub:** https://github.com/cvenhoff/steering-thinking-llms
  - Key behaviors: Backtracking, uncertainty estimation, example testing

- **Sinii et al. (2025)** - "Steering LLM Reasoning Through Bias-Only Adaptation" & "Small Vectors, Big Effects"
  - **GitHub:** https://github.com/corl-team/steering-reasoning
  - Mechanistic study of RL-induced reasoning via steering vectors

### Difference-of-Means Approach

From Venhoff et al. (2025):

```python
def compute_steering_vector(model, D_positive, D_negative, layer):
    """
    Compute steering vector as difference of mean activations.
    
    D_positive: samples exhibiting target behavior (e.g., backtracking)
    D_negative: samples not exhibiting target behavior
    """
    pos_activations = []
    neg_activations = []
    
    for sample in D_positive:
        act = get_layer_activation(model, sample, layer)
        pos_activations.append(act)
    
    for sample in D_negative:
        act = get_layer_activation(model, sample, layer)
        neg_activations.append(act)
    
    mean_pos = np.mean(pos_activations, axis=0)
    mean_neg = np.mean(neg_activations, axis=0)
    
    steering_vector = mean_pos - mean_neg
    return steering_vector / np.linalg.norm(steering_vector)
```

---

## Implementation Pipeline

### Complete Minimal Pipeline

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup - Use DeepSeek-R1-Distill for tractability
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# For SAE analysis, try LlamaScope SAEs (trained on Llama 3.1 8B)
# They may transfer reasonably well to the distilled model
# See: https://huggingface.co/fnlp/Llama-Scope

ERROR_FEATURE_IDX = 5441  # from your paper - adapt as needed

# 2. Generate data with <think> tags
def generate_with_activations(prompt):
    # Force thinking mode
    formatted_prompt = f"{prompt}\n<think>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=2048,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    
    # Extract activations at each generated token
    activations = []
    for step_hidden_states in outputs.hidden_states:
        # Get layer 19 (or your target layer)
        layer_act = step_hidden_states[19][:, -1, :]
        # Apply SAE if available, otherwise use raw activation
        activations.append(layer_act.cpu().numpy())
    
    generated_text = tokenizer.decode(outputs.sequences[0])
    
    return {
        'text': generated_text,
        'activations': activations,
        'tokens': outputs.sequences[0].tolist()
    }

# 3. Run on MBPP
from datasets import load_dataset
mbpp = load_dataset("mbpp", split="test")

results = []
for problem in mbpp:
    prompt = format_mbpp_prompt(problem)
    output = generate_with_activations(prompt)
    
    code = extract_code_from_think_tags(output['text'])
    correct = run_tests(code, problem['test_list'])
    
    results.append({
        'activations': output['activations'],
        'text': output['text'],
        'correct': correct
    })

# 4. Apply analysis methods
correct_trajs = normalize_trajectories([r['activations'] for r in results if r['correct']])
incorrect_trajs = normalize_trajectories([r['activations'] for r in results if not r['correct']])

# Method 1: Trajectory comparison
plot_trajectory_comparison(correct_trajs, incorrect_trajs)

# Method 4: Divergence point
div_point, p, d = find_divergence_point(correct_trajs, incorrect_trajs)
print(f"Trajectories diverge at {div_point}% (p={p:.4f}, d={d:.2f})")
```

---

## Key GitHub Repositories Summary

| Paper/Resource | GitHub | Key Contribution |
|----------------|--------|------------------|
| Turpin et al. (2023) - CoT Unfaithfulness | https://github.com/milesaturpin/cot-unfaithfulness | Bias experiments, evaluation code |
| Lyu et al. (2023) - Faithful CoT | https://github.com/veronica320/Faithful-COT | Translation → Solver approach |
| Paul et al. (2024) - Causal CoT | https://github.com/debjitpaul/Causal_CoT | Causal mediation analysis |
| Goodfire (2025) - R1 SAEs | https://github.com/goodfire-ai/r1-interpretability | SAEs for 671B R1, precomputed activations |
| Venhoff et al. (2025) - Steering Thinking LLMs | https://github.com/cvenhoff/steering-thinking-llms | Steering vectors for reasoning behaviors |
| Sinii et al. (2025) - Steering Reasoning | https://github.com/corl-team/steering-reasoning | Bias-only adaptation, RL-induced reasoning |
| LlamaScope SAEs | https://huggingface.co/fnlp/Llama-Scope | Pre-trained SAEs for Llama models |
| Awesome LLM Reasoning | https://github.com/atfortes/LLM-Reasoning-Papers | Comprehensive paper list |
| Awesome Efficient Reasoning | https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs | Efficient reasoning papers |
| Awesome Activation Engineering | https://github.com/ZFancy/awesome-activation-engineering | Steering/activation papers |

---

## Suggested Paper Section Structure

### "5.6 Error Features Track Reasoning Trajectories"

**Paragraph 1: Setup**
- Describe reasoning model used (e.g., DeepSeek-R1-Distill-Llama-8B)
- Explain activation extraction at each CoT token
- Reference: Venhoff et al. (2025) for reasoning behavior taxonomy

**Paragraph 2: Trajectory Divergence (Method 1 + 4)**
- Present trajectory comparison figure
- Report divergence point with statistics
- Reference: Lanham et al. (2023) for early answering methodology

**Paragraph 3: Event Analysis (Method 2)**
- Present event-triggered analysis around backtracking
- Reference: Goodfire (2025) for backtracking features in R1

**Paragraph 4: Faithfulness (Method 5)**
- Present faithfulness confusion matrix
- Report unfaithfulness rates
- Reference: Turpin et al. (2023) for faithfulness framework

**Paragraph 5: Practical Applications**
- Real-time monitoring during generation
- Mid-reasoning intervention to reduce corruption rate
- CoT faithfulness detection for code generation

---

## Resource Estimates

| Method | Compute | Implementation Time | Data Needed |
|--------|---------|---------------------|-------------|
| 1. Trajectory Comparison | Low | 1-2 days | 200-500 CoTs |
| 2. Event-Triggered | Low | 2-3 days | 200-500 CoTs |
| 3. LLM Segmentation | Medium (LLM API) | 3-4 days | 200-500 CoTs |
| 4. Critical Point | Low | 1 day | 200-500 CoTs |
| 5. Faithfulness | Low-Medium | 2-3 days | 200-500 CoTs |
| 6. SAE Analysis | Medium-High | 3-5 days | 200-500 CoTs |
| 7. Steering Vectors | Medium | 3-5 days | Contrastive dataset |

**Minimum viable extension:** Methods 1 + 4 (trajectory comparison + divergence point) can be done in ~3 days with 300 samples.

---

## References

### Core CoT Faithfulness Papers
1. Lanham, T., et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702
2. Turpin, M., et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023. arXiv:2305.04388
3. Lyu, Q., et al. (2023). Faithful Chain-of-Thought Reasoning. IJCNLP-AACL 2023. arXiv:2301.13379
4. Paul, D., et al. (2024). Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning. EMNLP Findings. arXiv:2402.13950

### Reasoning Model Interpretability Papers
5. Venhoff, C., et al. (2025). Understanding Reasoning in Thinking Language Models via Steering Vectors. ICLR 2025 Workshop. arXiv:2506.18167
6. Arcuschin, I., et al. (2025). Base Models Know How to Reason, Thinking Models Learn When. arXiv:2510.07364
7. Hazra, D., et al. (2025). Under the Hood of a Reasoning Model. Goodfire Research Blog.
8. Sinii, V., et al. (2025). Steering LLM Reasoning Through Bias-Only Adaptation. arXiv:2505.18706
9. Sinii, V., et al. (2025). Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors. arXiv:2509.06608

### DeepMind Pragmatic Interpretability
10. Nanda, N., et al. (2025). A Pragmatic Vision for Interpretability. Alignment Forum.
11. Nanda, N. (2025). MATS Applications + Research Directions I'm Currently Excited About. Alignment Forum.
