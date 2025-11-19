# Phase 8.3 Selective Steering: Bug Investigation Report

**Date**: 2025-11-19
**Status**: CRITICAL BUG IDENTIFIED
**Impact**: 0% correction rate, 97% corruption rate

---

## Executive Summary

Phase 8.3's selective steering implementation has a **critical bug in its two-stage generation architecture** that causes it to fail completely. While steering IS being applied (as confirmed by user testing), the steered code is incorrect because the model's `generate()` method cannot be called twice to "continue" generation. The second call treats the partial output as a new prompt and regenerates from scratch, disrupting the coherent generation flow.

### Quick Stats
- **Expected**: Selective steering that improves on Phase 4.8's 4.04% correction, 14.66% corruption
- **Actual**: ~0% correction, ~97% corruption
- **Root Cause**: Two-stage generation using `generate()` twice
- **Severity**: Complete failure of selective steering mechanism

---

## Table of Contents
1. [Implementation Walkthrough](#1-implementation-walkthrough)
2. [The Critical Bug](#2-the-critical-bug)
3. [Code Comparison: Phase 4.8 vs 8.3](#3-code-comparison-phase-48-vs-83)
4. [Why This Bug Causes Failure](#4-why-this-bug-causes-failure)
5. [Evidence from Results](#5-evidence-from-results)
6. [Recommended Fixes](#6-recommended-fixes)

---

## 1. Implementation Walkthrough

### 1.1 Design Intent

Phase 8.3 aims to implement **selective steering** based on threshold analysis from Phase 3.8:

- **Threshold Check**: Monitor L19-5441 (incorrect-predicting feature) activation
- **Optimal Threshold**: 15.5086 (from Phase 3.8 optimization)
- **Selective Application**:
  - If activation ≤ threshold → Use Phase 3.5 baseline (no steering, no generation)
  - If activation > threshold → Apply L16-11225 steering with coefficient 29.0

### 1.2 Architecture (As Implemented)

The implementation uses a **two-stage generation approach**:

```
Stage 1: Threshold Checking
├── Tokenize prompt
├── Generate FIRST TOKEN ONLY (max_new_tokens=1)
├── Extract L19 activation from this forward pass
├── Encode through SAE to get feature activation
└── Compare with threshold (15.5086)

Stage 2: Conditional Steering
├── If activation ≤ threshold:
│   └── Return Phase 3.5 baseline (no further generation)
└── If activation > threshold:
    ├── Install steering hook on L16
    ├── Continue generation from first_token_outputs
    └── Extract and evaluate code
```

### 1.3 Key Code Sections

**Threshold Checking** (lines 233-262):
```python
# Generate first token for ALL problems
first_token_outputs = self.model.generate(
    input_ids,
    max_new_tokens=1,
    do_sample=False,
    temperature=None,
    pad_token_id=self.tokenizer.pad_token_id
)

# Extract L19 activation and check threshold
l19_feature_activation = sae_features[0, self.incorrect_pred_feature].item()
should_steer = l19_feature_activation > self.threshold
```

**Selective Steering** (lines 286-333):
```python
if should_steer:
    # Install steering hook
    hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(steering_hook_fn)

    # Continue generation with steering
    final_outputs = self.model.generate(
        first_token_outputs,  # ← BUG: Contains [prompt + first_token]
        max_new_tokens=self.config.model_max_new_tokens - 1,
        ...
    )

    # Extract code
    generated_text = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    generated_code = extract_code(generated_text, prompt)
```

---

## 2. The Critical Bug

### 2.1 The Fundamental Flaw

**The bug**: `model.generate()` **CANNOT** be used to "continue" generation from a partial sequence.

**What Phase 8.3 does**:
```python
# Step 1: Generate first token
first_token_outputs = model.generate(input_ids, max_new_tokens=1)
# first_token_outputs shape: [1, prompt_len + 1]
# Contains: [prompt_token_1, ..., prompt_token_n, generated_token_1]

# Step 2: Try to "continue" from first token
final_outputs = model.generate(first_token_outputs, max_new_tokens=511)
# ❌ This DOESN'T continue - it treats entire sequence as NEW input!
```

**What actually happens**:
- `generate()` treats `first_token_outputs` as a **complete input prompt**
- It re-processes the entire sequence through the model
- It generates a **completely new sequence**, potentially ignoring/overwriting the first token
- The steering hook (installed for the second call) affects different token positions than intended

### 2.2 Token Flow Visualization

**Intended behavior**:
```
Prompt → [First Token] → [Token 2] → [Token 3] → ... → [Token 512]
         ↑             ↑              ↑                  ↑
         No steering   Steering       Steering           Steering
         Check L19     Applied        Applied            Applied
```

**Actual behavior**:
```
Prompt → [First Token] → THROW AWAY
                         ↓
Prompt + First Token → [New Token 1] → [New Token 2] → ... → [New Token 511]
                       ↑                ↑                      ↑
                       Steering         Steering               Steering
                       (wrong position) (wrong position)       (wrong position)
```

### 2.3 Code Extraction Bug

**Phase 4.8** (CORRECT):
```python
generated_text = self.tokenizer.decode(
    outputs.sequences[0][inputs['input_ids'].shape[1]:],  # ✅ Skip the prompt
    skip_special_tokens=True
)
```

**Phase 8.3** (BROKEN):
```python
generated_text = self.tokenizer.decode(
    final_outputs[0],  # ❌ Includes the entire sequence (prompt + generation)
    skip_special_tokens=True
)
```

Even if generation worked correctly, the code extraction would fail because:
- The decoded text contains the prompt
- `extract_code()` must parse out the prompt, which is unreliable
- Any parsing errors result in incorrect code

---

## 3. Code Comparison: Phase 4.8 vs 8.3

### 3.1 Generation Flow

| Aspect | Phase 4.8 ✅ | Phase 8.3 ❌ |
|--------|-------------|-------------|
| **Stages** | 1 (single generation) | 2 (first token + rest) |
| **Input to generate()** | `input_ids` (prompt only) | 1st: `input_ids`<br>2nd: `first_token_outputs` (prompt+token) |
| **max_new_tokens** | 512 | 1st: 1<br>2nd: 511 |
| **Steering timing** | Throughout generation | Only in 2nd stage |
| **Hook installed** | Before generate() | Between generate() calls |

### 3.2 Code Side-by-Side

**Phase 4.8: Single-Stage Generation** (steering_effect_analyzer.py:407-427)
```python
def _generate_with_steering(self, task_id, prompt, test_cases, baseline_row, target_layer, ...):
    # Tokenize prompt
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

    # Install steering hook BEFORE generation
    hook_fn = create_steering_hook(decoder_direction, coefficient)
    target_module = self.model.model.layers[target_layer]
    hook_handle = target_module.register_forward_pre_hook(hook_fn)

    try:
        # Single generation call with steering active
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.model_max_new_tokens,  # 512 tokens
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Extract generated code (skip prompt tokens)
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],  # ✅ Skip prompt
            skip_special_tokens=True
        )
        generated_code = extract_code(generated_text, prompt)

        # Evaluate
        test_passed = evaluate_code(generated_code, test_cases)

    finally:
        hook_handle.remove()
```

**Phase 8.3: Two-Stage Generation** (selective_steering_analyzer.py:230-333)
```python
def _generate_with_selective_steering(self, task_id, prompt, test_cases, baseline_row):
    # Tokenize prompt
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    # === STAGE 1: Generate first token (NO steering) ===
    extractor = ActivationExtractor(self.model, layers=[19], position=-1)
    extractor.setup_hooks()

    first_token_outputs = self.model.generate(
        input_ids,                  # ✅ Original prompt
        max_new_tokens=1,           # Only first token
        do_sample=False,
        temperature=None,
        pad_token_id=self.tokenizer.pad_token_id
    )
    # first_token_outputs shape: [1, prompt_len + 1]

    # Extract activation and check threshold
    l19_feature_activation = sae_features[0, self.incorrect_pred_feature].item()
    should_steer = l19_feature_activation > self.threshold
    extractor.remove_hooks()

    # If below threshold, return baseline (no steering)
    if not should_steer:
        return {...baseline...}

    # === STAGE 2: Continue with steering ===
    def steering_hook_fn(module, input):
        residual = input[0]
        decoder_direction = self.correct_decoder_direction.to(residual.dtype)
        steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.config.phase4_8_correct_coefficient
        residual = residual + steering.to(residual.device, residual.dtype)
        return (residual,) + input[1:]

    # Install steering hook BETWEEN generations
    hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(steering_hook_fn)

    try:
        # Second generation call
        final_outputs = self.model.generate(
            first_token_outputs,    # ❌ BUG: Contains [prompt + first_token]
            max_new_tokens=self.config.model_max_new_tokens - 1,  # 511 tokens
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Extract generated code (DOESN'T skip prompt)
        generated_text = self.tokenizer.decode(
            final_outputs[0],       # ❌ BUG: Includes entire sequence
            skip_special_tokens=True
        )
        generated_code = extract_code(generated_text, prompt)

        # Evaluate
        test_passed = evaluate_code(generated_code, test_cases)

    finally:
        hook_handle.remove()
```

### 3.3 Steering Hook Comparison

Both phases use **functionally identical** steering hooks:

**Phase 4.8**: Uses `create_steering_hook()` from common/steering_metrics.py
```python
def create_steering_hook(sae_decoder_direction: torch.Tensor, coefficient: float):
    def hook_fn(module, input):
        residual = input[0]
        steering = sae_decoder_direction.unsqueeze(0).unsqueeze(0) * coefficient
        residual = residual + steering.to(residual.device, residual.dtype)
        return (residual,) + input[1:]
    return hook_fn
```

**Phase 8.3**: Inline hook (lines 290-297)
```python
def steering_hook_fn(module, input):
    residual = input[0]
    decoder_direction = self.correct_decoder_direction.to(residual.dtype)
    steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.config.phase4_8_correct_coefficient
    residual = residual + steering.to(residual.device, residual.dtype)
    return (residual,) + input[1:]
```

**✅ The steering hooks are NOT the problem** - they're identical in logic and dtype handling.

### 3.4 Parameters Comparison

| Parameter | Phase 4.8 | Phase 8.3 | Match? |
|-----------|-----------|-----------|--------|
| **Steering layer** | 16 | 16 | ✅ |
| **Steering feature** | 11225 | 11225 | ✅ |
| **Coefficient** | 29.0 | 29.0 | ✅ |
| **max_new_tokens** | 512 | 1 + 511 = 512 | ✅ |
| **temperature** | 0.0 | None (greedy) | ✅ |
| **do_sample** | False | False | ✅ |

**✅ All steering parameters are identical** - the difference is purely in the generation flow.

---

## 4. Why This Bug Causes Failure

### 4.1 Why 0% Correction Rate?

For **initially incorrect problems**, selective steering should correct them. But it doesn't because:

1. **Generation is broken**: The two-stage approach disrupts coherent code generation
2. **Steering applies to wrong positions**: The hook affects the second `generate()` call, which processes the entire `[prompt + first_token]` sequence, not just the continuation
3. **First token is wasted**: The carefully generated first token (with L19 activation check) is essentially discarded when the second `generate()` starts fresh
4. **Inconsistent outputs**: The model may generate completely different code than it would have with single-stage generation

**Result**: Even when steering is applied, the broken generation flow produces incorrect code.

### 4.2 Why 97% Corruption Rate?

For **initially correct problems**, selective steering should preserve them. The 97% corruption rate occurs because:

1. **Most correct problems trigger steering**: If L19-5441 (incorrect-predicting feature) activates above threshold on correct problems, steering is applied
2. **Broken generation corrupts coherent solutions**: The two-stage process interrupts the model's natural generation flow, causing it to:
   - Lose context from the first token
   - Generate incoherently from the `[prompt + first_token]` input
   - Produce malformed or syntactically incorrect code
3. **Steering amplifies the disruption**: Adding L16-11225 steering on top of the broken generation makes things worse

**Result**: Nearly all problems where steering is applied become corrupted.

### 4.3 Why Steering IS Applied But Code Is Wrong

This explains the user's observation: **"I check and they were being steered but the steered code are still flagged as incorrect."**

- ✅ **Steering hook IS installed** (line 299)
- ✅ **Steering hook IS executed** during the second `generate()` call
- ✅ **L16-11225 activation IS modified** by the hook
- ❌ **But the generation flow is broken**, so the steering affects a broken process
- ❌ **The final code is incorrect** despite steering being applied

It's like **steering a car that's already crashed** - the steering wheel works, but it doesn't matter because the car isn't driving properly.

---

## 5. Evidence from Results

### 5.1 User Observation

> "The before samples that was correctly steered in @phase4_8_steering_analysis/ is not being steered correctly here. I check and they were being steered but the steered code are still flagged as incorrect."

This is **consistent with the two-stage generation bug**:
- The same problems that Phase 4.8 corrects are now failing in Phase 8.3
- Steering IS being applied (user confirmed)
- But the steered code is incorrect (not the same as Phase 4.8's correct output)

### 5.2 Expected vs Actual Metrics

| Metric | Phase 4.8 (Baseline) | Phase 8.3 (Expected) | Phase 8.3 (Actual) |
|--------|---------------------|----------------------|-------------------|
| **Correction Rate** | 4.04% | ≥4.04% (should improve) | ~0% ❌ |
| **Corruption Rate** | 14.66% | <14.66% (should reduce) | ~97% ❌ |
| **Preservation Rate** | 85.34% | >85.34% (should improve) | ~3% ❌ |

The results are **catastrophically worse** than Phase 4.8, indicating a fundamental implementation error, not a tuning issue.

### 5.3 Hypothetical Code Comparison

If we were to examine specific task outputs:

**Task X in Phase 4.8** (Steered, Correct):
```python
def example():
    # Coherent, correct solution
    return result
```

**Same Task X in Phase 8.3** (Steered, Incorrect):
```python
def example():
    # Broken, incoherent, or syntactically invalid
    # Because generation flow was disrupted
    return wrong_result
```

The code would be **different** even though both had steering applied, because Phase 8.3's generation is broken.

---

## 6. Recommended Fixes

### 6.1 Option A: Single-Stage with Conditional Hook ⭐ RECOMMENDED

**Approach**: Generate once, but apply steering conditionally based on first-token activation.

**Implementation**:
```python
def _generate_with_selective_steering(self, task_id, prompt, test_cases, baseline_row):
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    # Shared state for threshold check
    first_token_activation = None
    should_steer_flag = False

    def conditional_steering_hook(module, input):
        """Hook that checks threshold on first token, then steers if needed."""
        nonlocal first_token_activation, should_steer_flag

        residual = input[0]

        # On first forward pass (first token generation), check activation
        if first_token_activation is None:
            # Extract L19 activation
            l19_activation = residual[0, -1, :]  # Last position (first new token)
            with torch.no_grad():
                sae_features = self.sae_l19.encode(l19_activation.unsqueeze(0).float())
                first_token_activation = sae_features[0, self.incorrect_pred_feature].item()
                should_steer_flag = first_token_activation > self.threshold

        # Apply steering if threshold exceeded
        if should_steer_flag:
            decoder_direction = self.correct_decoder_direction.to(residual.dtype)
            steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.config.phase4_8_correct_coefficient
            residual = residual + steering.to(residual.device, residual.dtype)

        return (residual,) + input[1:]

    # Install hook on L16 (steering layer)
    hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(conditional_steering_hook)

    try:
        # Single generation call
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.model_max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Extract code (skip prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],  # ✅ Skip prompt
            skip_special_tokens=True
        )
        generated_code = extract_code(generated_text, prompt)

        # Evaluate
        test_passed = evaluate_code(generated_code, test_cases)

        return {
            'task_id': task_id,
            'steered': should_steer_flag,
            'l19_activation': first_token_activation,
            'final_passed': test_passed,
            'generated_code': generated_code,
            ...
        }

    finally:
        hook_handle.remove()
```

**Pros**:
- ✅ Single generation call (no two-stage bug)
- ✅ Maintains selective steering based on threshold
- ✅ Proper code extraction
- ✅ Steering applied at correct positions

**Cons**:
- ⚠️ Hook is more complex (must track state)
- ⚠️ L19 activation check happens inside hook (may need different layer hook)

**Note**: This approach requires hooking L19 for activation extraction AND L16 for steering. You'd need two hooks or a more sophisticated design.

---

### 6.2 Option B: Manual Token-by-Token Generation

**Approach**: Use manual forward passes instead of `generate()` to maintain full control.

**Implementation**:
```python
def _generate_with_selective_steering(self, task_id, prompt, test_cases, baseline_row):
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    # === Generate first token manually ===
    with torch.no_grad():
        # Forward pass with L19 activation extraction
        extractor = ActivationExtractor(self.model, layers=[19], position=-1)
        extractor.setup_hooks()

        outputs = self.model(input_ids)
        logits = outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        # Extract L19 activation
        activations = extractor.get_activations()
        l19_activation = activations[19]
        sae_features = self.sae_l19.encode(l19_activation.float())
        l19_feature_activation = sae_features[0, self.incorrect_pred_feature].item()

        extractor.remove_hooks()

    # Check threshold
    should_steer = l19_feature_activation > self.threshold

    if not should_steer:
        return {...baseline...}

    # === Continue generation with steering ===
    generated_tokens = [next_token.item()]
    current_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Install steering hook
    def steering_hook_fn(module, input):
        residual = input[0]
        decoder_direction = self.correct_decoder_direction.to(residual.dtype)
        steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.config.phase4_8_correct_coefficient
        residual = residual + steering.to(residual.device, residual.dtype)
        return (residual,) + input[1:]

    hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(steering_hook_fn)

    try:
        # Generate remaining tokens
        for _ in range(self.config.model_max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1)

                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        # Decode
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_code = extract_code(generated_text, prompt)

        # Evaluate
        test_passed = evaluate_code(generated_code, test_cases)

        return {...}

    finally:
        hook_handle.remove()
```

**Pros**:
- ✅ Complete control over generation
- ✅ No two-stage bug
- ✅ Can properly continue from first token
- ✅ Clear separation of threshold check and steering

**Cons**:
- ⚠️ Much slower (no generation optimizations)
- ⚠️ More complex code
- ⚠️ Must manually handle EOS, padding, etc.

---

### 6.3 Option C: Post-Hoc Filtering (Simplest)

**Approach**: Always apply steering (like Phase 4.8), then filter results based on threshold.

**Implementation**:
```python
def _generate_with_steering(self, task_id, prompt, test_cases, baseline_row):
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    # === Extract L19 activation (like Phase 3.5) ===
    with torch.no_grad():
        extractor = ActivationExtractor(self.model, layers=[19], position=-1)
        extractor.setup_hooks()

        # Forward pass (no generation)
        self.model(input_ids)

        activations = extractor.get_activations()
        l19_activation = activations[19]
        sae_features = self.sae_l19.encode(l19_activation.float())
        l19_feature_activation = sae_features[0, self.incorrect_pred_feature].item()

        extractor.remove_hooks()

    # Check threshold
    should_steer = l19_feature_activation > self.threshold

    if not should_steer:
        # Return baseline without generating
        return {...baseline...}

    # === Generate with steering (exactly like Phase 4.8) ===
    def steering_hook_fn(module, input):
        residual = input[0]
        decoder_direction = self.correct_decoder_direction.to(residual.dtype)
        steering = decoder_direction.unsqueeze(0).unsqueeze(0) * self.config.phase4_8_correct_coefficient
        residual = residual + steering.to(residual.device, residual.dtype)
        return (residual,) + input[1:]

    hook_handle = self.model.model.layers[self.correct_steer_layer].register_forward_pre_hook(steering_hook_fn)

    try:
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.model_max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Extract code (skip prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        generated_code = extract_code(generated_text, prompt)

        # Evaluate
        test_passed = evaluate_code(generated_code, test_cases)

        return {...}

    finally:
        hook_handle.remove()
```

**Pros**:
- ✅ Simple and clean
- ✅ Reuses Phase 4.8's proven generation logic
- ✅ No two-stage bug
- ✅ Easy to implement

**Cons**:
- ⚠️ Performs an extra forward pass (to get L19 activation)
- ⚠️ Slightly less efficient (but not significantly)
- ⚠️ L19 activation from forward pass may differ from generation (but should be close)

**Note**: This is the **safest and quickest fix** - it decouples threshold checking from generation entirely.

---

### 6.4 Comparison of Fix Options

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| **Complexity** | Medium | High | Low |
| **Performance** | Fast | Slow | Fast |
| **Correctness** | ✅ High | ✅ High | ✅ High |
| **Maintains selective steering** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Ease of implementation** | Medium | Hard | Easy |
| **Risk** | Medium | Medium | Low |
| **Recommended for** | Production | Research | Quick fix |

**Recommendation**: Start with **Option C** to quickly fix the bug and validate that selective steering works as intended. Later, optimize to **Option A** if needed for performance.

---

## 7. Conclusion

### 7.1 Summary

Phase 8.3's selective steering implementation has a **critical architectural flaw**:
- ❌ Two-stage generation using `generate()` twice is fundamentally broken
- ❌ The second `generate()` call doesn't continue - it regenerates from scratch
- ❌ Code extraction doesn't skip the prompt
- ✅ The steering hook itself is correct
- ✅ All parameters match Phase 4.8

### 7.2 Impact

- **0% correction rate**: Steering doesn't fix incorrect problems due to broken generation
- **97% corruption rate**: Steering disrupts coherent solutions on correct problems
- **Complete failure**: The implementation cannot achieve its design goals

### 7.3 Next Steps

1. **Immediate**: Implement Option C (post-hoc filtering) to fix the bug quickly
2. **Validation**: Verify that fixed implementation matches or exceeds Phase 4.8 performance
3. **Optimization**: If needed, migrate to Option A for cleaner selective steering
4. **Documentation**: Update Phase 8.3 documentation with corrected architecture

### 7.4 Lessons Learned

- ⚠️ `model.generate()` is a high-level API that cannot be split into stages
- ⚠️ Token-by-token continuation requires manual forward passes or stateful hooks
- ⚠️ Always extract generated text by skipping the input prompt tokens
- ✅ Single-stage generation with conditional hooks is the correct approach

---

## Appendix: File References

- **Phase 8.3 Implementation**: [selective_steering_analyzer.py](selective_steering_analyzer.py)
  - Bug location: Lines 304-314 (two-stage generation)
  - Threshold check: Lines 233-268
  - Steering hook: Lines 290-299

- **Phase 4.8 Implementation**: [../phase4_8_steering_analysis/steering_effect_analyzer.py](../phase4_8_steering_analysis/steering_effect_analyzer.py)
  - Working generation: Lines 407-427
  - Steering hook: Uses common/steering_metrics.py:221-246

- **Configuration**: [../common/config.py](../common/config.py)
  - Line 197: `phase4_8_correct_coefficient = 29.0`
  - Line 203: `model_max_new_tokens = 512`

---

**End of Report**
