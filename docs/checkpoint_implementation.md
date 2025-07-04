# Checkpoint Implementation Guide

This document describes the checkpointing mechanism implemented for multi-GPU dataset processing in the PVA-SAE project.

## Overview

The checkpointing system allows long-running dataset generation tasks to be split into smaller chunks that can be resumed if interrupted. This is especially valuable for:
- Multi-GPU processing where different GPUs may fail independently
- Long-running jobs that may be interrupted
- Development and testing with the ability to resume from partial results

## Architecture

### Components

1. **multi_gpu_launcher.py**: Extended to support checkpointing
   - Launches `chunk_runner.py` for each GPU (checkpointing enabled by default)
   - Monitors progress and merges chunks after completion
   - Provides `--no-checkpoint` flag to disable checkpointing

2. **chunk_runner.py**: New script for sequential chunk processing
   - Processes assigned range in chunks of `checkpoint_frequency` size
   - Each chunk outputs to its own directory
   - Skips already-completed chunks on restart

3. **Phase Runners**: Modified to respect environment variables
   - `phase1_simplified/runner.py`: Checks `PHASE1_OUTPUT_DIR` 
   - `phase3_5_temperature_robustness/temperature_runner.py`: Checks `PHASE3_5_OUTPUT_DIR`
   - Falls back to config values if environment variables not set

### Directory Structure

When checkpointing is enabled, outputs are organized as:

```
data/phase1_0/
├── chunks/
│   ├── gpu0/
│   │   ├── chunk_00000-00049/
│   │   │   ├── dataset_sae_20240110_143022.parquet
│   │   │   ├── activations/
│   │   │   └── run.log
│   │   ├── chunk_00050-00099/
│   │   │   └── ...
│   │   └── completed.json
│   ├── gpu1/
│   │   └── ...
│   └── gpu2/
│       └── ...
└── dataset_gpu0_20240110_150000.parquet  (merged final output)
```

## Usage

### Basic Usage (Checkpointing Enabled by Default)

```bash
# Phase 1: Dataset building with automatic checkpointing
python multi_gpu_launcher.py --phase 1 --start 0 --end 488

# Phase 3.5: Temperature robustness with checkpointing
python multi_gpu_launcher.py --phase 3.5 --start 0 --end 389
```

### Disable Checkpointing

```bash
# Run without checkpointing (original behavior)
python multi_gpu_launcher.py --phase 1 --start 0 --end 488 --no-checkpoint
```

### Recovery

If a run is interrupted, simply run the same command again:

```bash
# Original command
python multi_gpu_launcher.py --phase 1 --start 0 --end 488

# After interruption, run same command - completed chunks are skipped
python multi_gpu_launcher.py --phase 1 --start 0 --end 488
```

Output will show:
```
GPU 0: Processing chunks for range 0-162
  Chunk 0/3: 0-49 [SKIPPED - Already completed]
  Chunk 1/3: 50-99 [SKIPPED - Already completed]
  Chunk 2/3: 100-162 [PROCESSING...]
```

## Configuration

Checkpoint frequency is controlled by `config.checkpoint_frequency` (default: 50).

To change the chunk size, modify `common/config.py`:
```python
checkpoint_frequency: int = 50  # Records per chunk
```

## How It Works

1. **Work Distribution**: 
   - Total work is divided among GPUs (same as before)
   - Each GPU's work is further divided into chunks

2. **Parallel Processing**:
   - All GPUs run in parallel
   - Each GPU processes its chunks sequentially
   - Directory isolation prevents file conflicts

3. **Recovery**:
   - Checks for existing chunk output directories
   - Skips chunks that have completed outputs
   - Continues from first missing chunk

4. **Merging**:
   - After all chunks complete, automatically merges into final datasets
   - One merged file per GPU

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Direct run.py usage**: Works exactly as before
   ```bash
   python run.py phase 1  # No checkpointing, original behavior
   ```

2. **Multi-GPU without checkpointing**: Use `--no-checkpoint`
   ```bash
   python multi_gpu_launcher.py --phase 1 --start 0 --end 488 --no-checkpoint
   ```

3. **Environment variables**: Only set when checkpointing is active

## Adapting for New Phases

To add checkpointing support to a new phase:

1. **Modify the phase runner** to check for environment variable:
   ```python
   import os
   output_dir_env = os.environ.get('PHASE_X_OUTPUT_DIR')
   if output_dir_env:
       output_dir = Path(output_dir_env)
   else:
       output_dir = Path(self.config.phase_x_output_dir)
   ```

2. **Update chunk_runner.py** to set the appropriate environment variable:
   ```python
   if phase == "X":
       env["PHASE_X_OUTPUT_DIR"] = str(chunk_dir)
   ```

3. **Update multi_gpu_launcher.py** to handle the new phase

## Troubleshooting

### Chunks not being skipped
- Check that chunk directories contain expected output files
- Verify file naming matches the pattern expected by chunk_runner.py

### Merge failures
- Ensure all chunks completed successfully
- Check for corrupted parquet files in chunk directories

### GPU assignment issues
- Verify CUDA_VISIBLE_DEVICES is set correctly
- Check GPU availability with `nvidia-smi`

## Performance Considerations

- **Overhead**: Minimal - mainly from subprocess launches between chunks
- **Disk usage**: Temporary increase due to chunk storage before merging
- **Model loading**: Happens once per GPU, not per chunk

## Future Improvements

Potential enhancements (not implemented):
- Automatic cleanup of chunk directories after successful merge
- Configurable retention of chunk files
- Progress tracking across multiple runs
- Checksum validation of chunk outputs