# Configuration Guide

The PVA-SAE project uses a unified configuration system that provides clear visibility and control over all settings.

## Quick Start

### View Current Configuration
```bash
# Show all settings that will be used
python3 run.py phase 1 --show-config

# Show config with CLI overrides
python3 run.py phase 1 --model gpt2 --start 0 --end 50 --show-config
```

### Use Configuration Files
```bash
# Load config from file
python3 run.py phase 1 --config-file my_config.json

# Save current config to file
python3 run.py phase 1 --model gpt2 --save-config my_config.json
```

### Environment Variables
Set any config value via environment variables:
```bash
export PVA_SAE_MODEL_NAME=gpt2
export PVA_SAE_CHECKPOINT_FREQUENCY=25
python3 run.py phase 1
```

## Configuration Precedence

Settings are applied in this order (later overrides earlier):
1. **Defaults** - Built-in defaults in `unified_config.py`
2. **Config File** - Values from `--config-file`
3. **Environment Variables** - `PVA_SAE_*` variables
4. **CLI Arguments** - Command-line flags

## Configuration Structure

Settings are organized by prefix:
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
# Basic usage
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 100

# With custom config
python3 run.py phase 1 --config-file configs/fast_test.json

# Save config for reproducibility
python3 run.py phase 1 --model gpt2 --checkpoint-frequency 25 --save-config experiment_1.json
```

### Phase 2: SAE Analysis
```bash
# Show Phase 2 specific settings
python3 run.py phase 2 --show-config

# Override SAE settings
python3 run.py phase 2 --latent-threshold 0.05 --pile-filter
```

## Creating Config Files

Example config file (`config_example.json`):
```json
{
  "model_name": "google/gemma-2-2b",
  "model_temperature": 0.0,
  "dataset_start_idx": 0,
  "dataset_end_idx": 50,
  "checkpoint_frequency": 25,
  "verbose": true
}
```

You only need to include settings you want to change from defaults.

## Debugging Configuration Issues

1. **Always use `--show-config` first** to see what settings will be used
2. **Check precedence** - CLI args override everything
3. **Validate paths** - Phase directories are shown in the config dump
4. **Use `--verbose`** for detailed logging

## Migration from Old System

The new unified config system replaces the old separate config classes. During migration:
- Old code still works via `config_adapter.py`
- New code should use `Config` directly
- All settings are now in one place for clarity