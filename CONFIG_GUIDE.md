# Configuration Guide

The PVA-SAE project uses a simple, clean configuration system with clear visibility and control over all settings.

## Configuration Methods

Settings are applied in this order (later overrides earlier):

1. **Defaults** - Built-in defaults in `config.py` (your single source of truth)
2. **Environment Variables** - `PVA_SAE_*` variables  
3. **CLI Arguments** - Command-line flags (highest priority)

## Quick Start

### View Current Configuration
Always check what settings will be used before running:
```bash
# Show all settings with defaults
python3 run.py phase 1 --show-config

# Show config with CLI overrides
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 50 --show-config
```

### Using CLI Arguments
Override specific settings as needed:
```bash
# Basic usage with defaults
python3 run.py phase 1

# Override specific settings
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 100 --verbose
```

### Using Environment Variables
Set environment variables for session-wide settings:
```bash
export PVA_SAE_MODEL_NAME=google/gemma-2-2b
export PVA_SAE_CHECKPOINT_FREQUENCY=25
export PVA_SAE_VERBOSE=true
python3 run.py phase 1  # Uses env vars
```

## Configuration Structure

All defaults are defined in `config.py`. Settings are organized by prefix:
- `model_*` - Model configuration
- `dataset_*` - Dataset settings
- `activation_*` - Activation extraction
- `checkpoint_*` - Checkpointing behavior
- `memory_*` - Memory management
- `sae_*` - SAE analysis (Phase 2)
- `validation_*` - Validation settings (Phase 3)

## Examples

### Phase 1: Dataset Building
```bash
# Use all defaults from config.py
python3 run.py phase 1

# Override specific settings
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 100

# Check what will run before executing
python3 run.py phase 1 --start 50 --end 150 --show-config
```

### Phase 2: SAE Analysis
```bash
# Show Phase 2 settings before running
python3 run.py phase 2 --show-config

# Override SAE settings
python3 run.py phase 2 --latent-threshold 0.05 --pile-filter
```

## Modifying Default Settings

To change default values, edit the constants in `config.py`:
```python
# common/config.py
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_CHECKPOINT_FREQUENCY = 50
# ... other defaults
```

## Debugging Configuration Issues

1. **Always use `--show-config` first** to see what settings will be used
2. **Check precedence** - CLI args override everything
3. **Validate paths** - Phase directories are shown in the config dump
4. **Use `--verbose`** for detailed logging

## Environment Variable Reference

Any config setting can be set via environment variable using the pattern `PVA_SAE_<SETTING_NAME>`:
- `PVA_SAE_MODEL_NAME` - Model to use
- `PVA_SAE_CHECKPOINT_FREQUENCY` - How often to checkpoint
- `PVA_SAE_VERBOSE` - Enable verbose logging
- etc.

For lists, use comma-separated values:
```bash
export PVA_SAE_ACTIVATION_LAYERS="13,14,16,17,20"
```