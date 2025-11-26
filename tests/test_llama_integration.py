"""
LLAMA Integration Tests for PVA-SAE

Stage 0: Validate LLAMA + LlamaScope SAE integration before modifying existing code.

Run incrementally:
    python3 tests/test_llama_integration.py --test 1  # LLAMA model loads
    python3 tests/test_llama_integration.py --test 2  # LLAMA generates code
    python3 tests/test_llama_integration.py --test 3  # Activation extraction works
    python3 tests/test_llama_integration.py --test 4  # LlamaScope SAE loads
    python3 tests/test_llama_integration.py --test 5  # SAE encoding works
    python3 tests/test_llama_integration.py --test 6  # Full PCDGE cycle
    python3 tests/test_llama_integration.py           # Run all tests
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration
# =============================================================================

LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B"
LLAMA_SAE_REPO = "fnlp/Llama-Scope"
LLAMA_HIDDEN_SIZE = 4096
LLAMA_NUM_LAYERS = 32
LLAMA_SAE_WIDTH = 32768  # 8x expansion: 4096 * 8

# Test layer - use middle layer for testing
TEST_LAYER = 15

# =============================================================================
# Test 1: LLAMA Model Loads
# =============================================================================

def test_1_llama_model_loads() -> bool:
    """Test that LLAMA-3.1-8B loads from HuggingFace."""
    print("\n" + "="*60)
    print("TEST 1: LLAMA Model Loads")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {LLAMA_MODEL_NAME}")
        print("This may take a few minutes on first run...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Load model with automatic device mapping
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"  Model loaded: {type(model).__name__}")

        # Verify architecture
        num_layers = len(model.model.layers)
        hidden_size = model.config.hidden_size
        print(f"  Layers: {num_layers} (expected: {LLAMA_NUM_LAYERS})")
        print(f"  Hidden size: {hidden_size} (expected: {LLAMA_HIDDEN_SIZE})")

        assert num_layers == LLAMA_NUM_LAYERS, f"Expected {LLAMA_NUM_LAYERS} layers, got {num_layers}"
        assert hidden_size == LLAMA_HIDDEN_SIZE, f"Expected {LLAMA_HIDDEN_SIZE} hidden size, got {hidden_size}"

        # Check layer access pattern (same as Gemma)
        layer = model.model.layers[TEST_LAYER]
        print(f"  Layer {TEST_LAYER} accessible: {type(layer).__name__}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        print("\n[PASS] Test 1: LLAMA model loads correctly")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 1: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 2: LLAMA Generates Python Code
# =============================================================================

def test_2_llama_generates_code() -> bool:
    """Test that LLAMA can generate Python code from a prompt."""
    print("\n" + "="*60)
    print("TEST 2: LLAMA Generates Python Code")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {LLAMA_MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Simple coding prompt (MBPP style)
        prompt = '''Write a python function to find the sum of all even numbers in a list.

assert sum_even([1, 2, 3, 4, 5, 6]) == 12
assert sum_even([2, 4, 6]) == 12

# Solution:
def sum_even'''

        print(f"Prompt:\n{prompt[:100]}...")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        print(f"  Input tokens: {input_ids.shape[1]}")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.0,  # Deterministic
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs.shape[1] - input_ids.shape[1]
        print(f"  Generated {new_tokens} new tokens")

        # Extract the generated code (after prompt)
        generated_code = generated_text[len(prompt):]
        print(f"\nGenerated code:\ndef sum_even{generated_code[:200]}")

        # Basic validation - should contain function body
        assert len(generated_code) > 10, "Generated code too short"
        assert "return" in generated_code.lower() or "sum" in generated_code.lower(), \
            "Generated code doesn't look like a function"

        # Clean up
        del model
        torch.cuda.empty_cache()

        print("\n[PASS] Test 2: LLAMA generates Python code")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 2: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 3: Activation Extraction via Hooks
# =============================================================================

def test_3_activation_extraction() -> bool:
    """Test that activation hooks capture residual stream from LLAMA."""
    print("\n" + "="*60)
    print("TEST 3: Activation Extraction via Hooks")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {LLAMA_MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Storage for captured activations
        activations: Dict[int, torch.Tensor] = {}
        hooks = []
        test_layers = [10, 15, 20]  # Test multiple layers

        def create_hook(layer_idx: int):
            """Create pre-hook to capture residual stream."""
            def hook_fn(module, input):
                # input[0] is residual stream: (batch, seq_len, hidden_size)
                residual = input[0]
                # Capture last token position
                activations[layer_idx] = residual[:, -1, :].detach().cpu().clone()
                return input  # Pass through unchanged
            return hook_fn

        # Register hooks
        print(f"Registering hooks for layers: {test_layers}")
        for layer_idx in test_layers:
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_pre_hook(create_hook(layer_idx))
            hooks.append(hook)

        # Run forward pass
        prompt = "def fibonacci(n):"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        print(f"Running forward pass with prompt: '{prompt}'")
        with torch.no_grad():
            _ = model(input_ids)

        # Verify activations
        print("\nCaptured activations:")
        for layer_idx in test_layers:
            assert layer_idx in activations, f"Layer {layer_idx} activation not captured"
            act = activations[layer_idx]
            print(f"  Layer {layer_idx}: shape={act.shape}, "
                  f"mean={act.mean():.4f}, std={act.std():.4f}")

            # Verify shape
            assert act.shape == (1, LLAMA_HIDDEN_SIZE), \
                f"Expected shape (1, {LLAMA_HIDDEN_SIZE}), got {act.shape}"

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        # Clean up model
        del model
        torch.cuda.empty_cache()

        print("\n[PASS] Test 3: Activation extraction works")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 3: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 4: LlamaScope SAE Loads
# =============================================================================

def test_4_llamascope_sae_loads() -> bool:
    """Test that LlamaScope SAE loads from fnlp/Llama-Scope."""
    print("\n" + "="*60)
    print("TEST 4: LlamaScope SAE Loads")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors.torch import load_file

        print(f"SAE Repository: {LLAMA_SAE_REPO}")

        # First, list files to understand structure
        print("Listing repository files...")
        files = list_repo_files(LLAMA_SAE_REPO)

        # Filter for layer 15 SAE (L15R-8x = Layer 15, Residual, 8x expansion)
        sae_folder = f"L{TEST_LAYER}R-8x"
        sae_files = [f for f in files if sae_folder in f]
        print(f"Files for {sae_folder}: {sae_files[:5]}...")

        # Find safetensors file
        safetensor_files = [f for f in sae_files if f.endswith('.safetensors')]
        assert len(safetensor_files) > 0, f"No safetensors files found for {sae_folder}"

        sae_file = safetensor_files[0]
        print(f"Downloading: {sae_file}")

        # Download SAE weights
        local_path = hf_hub_download(
            repo_id=LLAMA_SAE_REPO,
            filename=sae_file,
        )
        print(f"Downloaded to: {local_path}")

        # Load weights
        weights = load_file(local_path)
        print(f"\nSAE weights keys: {list(weights.keys())}")

        # Analyze weight shapes
        print("\nWeight shapes:")
        for key, tensor in weights.items():
            print(f"  {key}: {tensor.shape}")

        # Expected: encoder (d_model -> d_sae), decoder (d_sae -> d_model)
        # LlamaScope uses: W_enc, b_enc, W_dec, b_dec (or similar)

        # Verify we have encoder and decoder weights
        has_encoder = any('enc' in k.lower() for k in weights.keys())
        has_decoder = any('dec' in k.lower() for k in weights.keys())

        if not (has_encoder and has_decoder):
            # Some SAEs use different naming - check for weight matrices
            print("Checking alternative weight naming...")
            for key, tensor in weights.items():
                if len(tensor.shape) == 2:
                    print(f"  Matrix {key}: {tensor.shape}")

        print("\n[PASS] Test 4: LlamaScope SAE loads")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 4: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 5: SAE Encoding Works
# =============================================================================

class TopKSAE:
    """LlamaScope SAE with TopK activation."""

    def __init__(self, weights: Dict[str, torch.Tensor], k: int = 64, device: str = "cuda"):
        """
        Initialize TopK SAE from weight dictionary.

        Args:
            weights: Dictionary of weight tensors from safetensors
            k: Number of top activations to keep
            device: Device to run on
        """
        self.k = k
        self.device = device

        # Detect weight naming convention
        # Common patterns: W_enc/W_dec, encoder/decoder, weight_enc/weight_dec
        self._load_weights(weights)

    def _load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load weights handling different naming conventions."""
        keys = list(weights.keys())
        print(f"  Loading SAE weights from keys: {keys}")

        # Try common naming patterns
        enc_key = None
        dec_key = None
        b_enc_key = None
        b_dec_key = None

        for key in keys:
            key_lower = key.lower()
            if 'enc' in key_lower and 'weight' in key_lower or key_lower == 'w_enc':
                enc_key = key
            elif 'dec' in key_lower and 'weight' in key_lower or key_lower == 'w_dec':
                dec_key = key
            elif 'enc' in key_lower and 'bias' in key_lower or key_lower == 'b_enc':
                b_enc_key = key
            elif 'dec' in key_lower and 'bias' in key_lower or key_lower == 'b_dec':
                b_dec_key = key

        # If not found, try to infer from shapes
        if enc_key is None or dec_key is None:
            print("  Inferring weight roles from shapes...")
            for key, tensor in weights.items():
                if len(tensor.shape) == 2:
                    d1, d2 = tensor.shape
                    # Encoder: (d_model, d_sae) or (d_sae, d_model)
                    # Decoder: (d_sae, d_model) or (d_model, d_sae)
                    if d1 == LLAMA_HIDDEN_SIZE:
                        enc_key = key
                        print(f"    Encoder: {key} with shape {tensor.shape}")
                    elif d2 == LLAMA_HIDDEN_SIZE:
                        dec_key = key
                        print(f"    Decoder: {key} with shape {tensor.shape}")

        # Load weights
        if enc_key:
            self.W_enc = weights[enc_key].to(self.device).float()
        else:
            raise ValueError(f"Could not find encoder weights in keys: {keys}")

        if dec_key:
            self.W_dec = weights[dec_key].to(self.device).float()
        else:
            raise ValueError(f"Could not find decoder weights in keys: {keys}")

        # Load biases (optional, use zeros if not present)
        if b_enc_key:
            self.b_enc = weights[b_enc_key].to(self.device).float()
        else:
            d_sae = self.W_enc.shape[1] if self.W_enc.shape[0] == LLAMA_HIDDEN_SIZE else self.W_enc.shape[0]
            self.b_enc = torch.zeros(d_sae, device=self.device)

        if b_dec_key:
            self.b_dec = weights[b_dec_key].to(self.device).float()
        else:
            self.b_dec = torch.zeros(LLAMA_HIDDEN_SIZE, device=self.device)

        # Determine SAE dimension
        if self.W_enc.shape[0] == LLAMA_HIDDEN_SIZE:
            self.d_sae = self.W_enc.shape[1]
        else:
            self.d_sae = self.W_enc.shape[0]
            # Transpose if needed
            self.W_enc = self.W_enc.T

        print(f"  SAE dimensions: d_model={LLAMA_HIDDEN_SIZE}, d_sae={self.d_sae}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features using TopK.

        Args:
            x: Input tensor of shape (batch, d_model)

        Returns:
            Sparse feature tensor of shape (batch, d_sae)
        """
        # Ensure correct device and dtype
        x = x.to(self.device).float()

        # Pre-activation: x @ W_enc + b_enc
        pre_act = x @ self.W_enc + self.b_enc

        # TopK: keep only top k activations, zero the rest
        topk_values, topk_indices = torch.topk(pre_act, self.k, dim=-1)

        # Create sparse output
        sparse = torch.zeros_like(pre_act)
        sparse.scatter_(-1, topk_indices, topk_values)

        return sparse

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activations.

        Args:
            features: Sparse feature tensor of shape (batch, d_sae)

        Returns:
            Reconstructed activations of shape (batch, d_model)
        """
        return features @ self.W_dec + self.b_dec


def test_5_sae_encoding() -> bool:
    """Test that SAE can encode LLAMA activations to sparse features."""
    print("\n" + "="*60)
    print("TEST 5: SAE Encoding Works")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors.torch import load_file

        # Load model
        print(f"Loading model: {LLAMA_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Get device
        device = next(model.parameters()).device

        # Extract activation
        activation = None
        def capture_hook(module, input):
            nonlocal activation
            activation = input[0][:, -1, :].detach().clone()
            return input

        layer = model.model.layers[TEST_LAYER]
        hook = layer.register_forward_pre_hook(capture_hook)

        prompt = "def calculate_sum(numbers):"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        print(f"Extracting activation from layer {TEST_LAYER}...")
        with torch.no_grad():
            _ = model(input_ids)

        hook.remove()
        print(f"  Activation shape: {activation.shape}")

        # Load SAE
        print(f"\nLoading SAE for layer {TEST_LAYER}...")
        files = list_repo_files(LLAMA_SAE_REPO)
        sae_folder = f"L{TEST_LAYER}R-8x"
        safetensor_files = [f for f in files if sae_folder in f and f.endswith('.safetensors')]

        local_path = hf_hub_download(
            repo_id=LLAMA_SAE_REPO,
            filename=safetensor_files[0],
        )
        weights = load_file(local_path)

        # Create SAE
        sae = TopKSAE(weights, k=64, device=str(device))

        # Encode
        print("\nEncoding activation...")
        features = sae.encode(activation)
        print(f"  Feature shape: {features.shape}")

        # Analyze sparsity
        non_zero = (features != 0).sum().item()
        sparsity = non_zero / features.numel()
        print(f"  Non-zero features: {non_zero}")
        print(f"  Sparsity: {sparsity:.4f}")

        # TopK should give exactly k non-zero per sample
        expected_nonzero = 64  # k
        assert non_zero == expected_nonzero, f"Expected {expected_nonzero} non-zero, got {non_zero}"

        # Test decode (reconstruction)
        print("\nTesting reconstruction...")
        reconstructed = sae.decode(features)
        print(f"  Reconstructed shape: {reconstructed.shape}")

        # Calculate reconstruction error
        mse = ((activation.float().to(device) - reconstructed) ** 2).mean().item()
        print(f"  Reconstruction MSE: {mse:.6f}")

        # Clean up
        del model, sae
        torch.cuda.empty_cache()

        print("\n[PASS] Test 5: SAE encoding works")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 5: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 6: Full PCDGE Cycle
# =============================================================================

def test_6_full_pcdge() -> bool:
    """Test complete Prompt-Capture-Decompose-Generate-Evaluate cycle with LLAMA."""
    print("\n" + "="*60)
    print("TEST 6: Full PCDGE Cycle")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors.torch import load_file
        import subprocess
        import tempfile

        # Load model
        print(f"Loading model: {LLAMA_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        device = next(model.parameters()).device

        # =====================================================================
        # PROMPT: Create coding prompt (MBPP style)
        # =====================================================================
        print("\n[P]rompt: Building coding prompt...")
        prompt = '''Write a python function to find the maximum element in a list.

assert find_max([1, 2, 3, 4, 5]) == 5
assert find_max([10, 5, 20, 15]) == 20
assert find_max([-1, -5, -2]) == -1

# Solution:
def find_max'''

        print(f"  Prompt length: {len(prompt)} chars")

        # =====================================================================
        # CAPTURE: Extract activations during generation
        # =====================================================================
        print("\n[C]apture: Setting up activation hooks...")

        activations: Dict[int, torch.Tensor] = {}
        hooks = []
        capture_layers = [10, 15, 20]

        def create_capture_hook(layer_idx: int):
            def hook_fn(module, input):
                if layer_idx not in activations:  # Only capture first forward pass
                    activations[layer_idx] = input[0][:, -1, :].detach().cpu().clone()
                return input
            return hook_fn

        for layer_idx in capture_layers:
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_pre_hook(create_capture_hook(layer_idx))
            hooks.append(hook)

        # =====================================================================
        # GENERATE: Generate code solution
        # =====================================================================
        print("\n[G]enerate: Generating code...")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_text[len(prompt):]

        # Extract just the function
        full_function = f"def find_max{generated_code}"
        # Truncate at next def or class
        lines = full_function.split('\n')
        function_lines = []
        for i, line in enumerate(lines):
            if i > 0 and (line.startswith('def ') or line.startswith('class ')):
                break
            function_lines.append(line)
        full_function = '\n'.join(function_lines)

        print(f"  Generated function:\n{full_function[:300]}")

        # =====================================================================
        # DECOMPOSE: Apply SAE decomposition
        # =====================================================================
        print("\n[D]ecompose: Applying SAE to activations...")

        # Load SAE for layer 15
        files = list_repo_files(LLAMA_SAE_REPO)
        sae_folder = f"L{TEST_LAYER}R-8x"
        safetensor_files = [f for f in files if sae_folder in f and f.endswith('.safetensors')]

        local_path = hf_hub_download(
            repo_id=LLAMA_SAE_REPO,
            filename=safetensor_files[0],
        )
        weights = load_file(local_path)
        sae = TopKSAE(weights, k=64, device="cpu")

        # Decompose each captured activation
        for layer_idx, act in activations.items():
            features = sae.encode(act)
            non_zero = (features != 0).sum().item()
            print(f"  Layer {layer_idx}: {non_zero} active features")

        # =====================================================================
        # EVALUATE: Execute tests to classify correctness
        # =====================================================================
        print("\n[E]valuate: Testing generated code...")

        test_code = f'''
{full_function}

# Test cases
try:
    assert find_max([1, 2, 3, 4, 5]) == 5
    assert find_max([10, 5, 20, 15]) == 20
    assert find_max([-1, -5, -2]) == -1
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
'''

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        Path(temp_path).unlink()  # Clean up

        output = result.stdout.strip()
        is_correct = "PASS" in output

        print(f"  Test result: {'PASS (Correct)' if is_correct else 'FAIL (Incorrect)'}")
        if result.stderr:
            print(f"  Stderr: {result.stderr[:200]}")

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "-"*60)
        print("PCDGE Cycle Summary:")
        print(f"  [P]rompt: {len(prompt)} chars")
        print(f"  [C]apture: {len(activations)} layers, shape {list(activations.values())[0].shape}")
        print(f"  [D]ecompose: {sae.d_sae} SAE features")
        print(f"  [G]enerate: {len(generated_code)} chars generated")
        print(f"  [E]valuate: {'CORRECT' if is_correct else 'INCORRECT'}")

        # Clean up
        del model, sae
        torch.cuda.empty_cache()

        print("\n[PASS] Test 6: Full PCDGE cycle works")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 6: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLAMA Integration Tests for PVA-SAE")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run specific test (1-6). If not specified, runs all tests.")
    args = parser.parse_args()

    print("="*60)
    print("LLAMA Integration Tests for PVA-SAE")
    print("="*60)
    print(f"Model: {LLAMA_MODEL_NAME}")
    print(f"SAE: {LLAMA_SAE_REPO}")
    print(f"Test Layer: {TEST_LAYER}")
    print("="*60)

    tests = [
        (1, "LLAMA Model Loads", test_1_llama_model_loads),
        (2, "LLAMA Generates Code", test_2_llama_generates_code),
        (3, "Activation Extraction", test_3_activation_extraction),
        (4, "LlamaScope SAE Loads", test_4_llamascope_sae_loads),
        (5, "SAE Encoding", test_5_sae_encoding),
        (6, "Full PCDGE Cycle", test_6_full_pcdge),
    ]

    if args.test:
        # Run specific test
        test_num, test_name, test_fn = tests[args.test - 1]
        success = test_fn()
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        results = []
        for test_num, test_name, test_fn in tests:
            success = test_fn()
            results.append((test_num, test_name, success))

            if not success:
                print(f"\n[STOPPING] Test {test_num} failed. Fix before continuing.")
                break

        # Print summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        for test_num, test_name, success in results:
            status = "[PASS]" if success else "[FAIL]"
            print(f"  {status} Test {test_num}: {test_name}")

        passed = sum(1 for _, _, s in results if s)
        total = len(results)
        print(f"\nPassed: {passed}/{total}")

        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
