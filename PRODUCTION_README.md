# Production Dataset Building - Hardened Pipeline

This document describes the production-hardened MBPP dataset building pipeline designed to reliably process the full 974-record dataset.

## Overview

The hardened pipeline (`data_processing_hardened.py`) extends the original data processing code with enterprise-grade reliability features:

### Key Hardening Features

1. **Progress Resuming & Checkpointing**
   - Automatic checkpoint saves every N records (default: 50)
   - Resume from last checkpoint after interruption
   - Atomic checkpoint writes for crash safety
   
2. **Robust Error Handling & Recovery**
   - Retry failed records up to 3 times with exponential backoff
   - Continue processing after individual record failures
   - Comprehensive error tracking and reporting
   
3. **Memory Management**
   - Periodic GPU cache clearing (every 100 records)
   - Python garbage collection optimization
   - Memory usage monitoring and warnings
   
4. **Intermediate Saving & Progress Reporting**
   - Autosave partial results every 100 records
   - Real-time progress bars with ETA estimation
   - Detailed progress logging every 10 records
   
5. **Enhanced Configuration & Monitoring**
   - Configurable parameters via `HardeningConfig`
   - Resource monitoring (RAM and GPU memory)
   - Comprehensive timing statistics

## Quick Start

### Test Run (10 records)
```bash
python scripts/run_production_build.py --test-run
```

### Full Production Run (974 records)
```bash
# With default gemma-2-2b model
python scripts/run_production_build.py

# With gemma-2-9b (thesis model)
python scripts/run_production_build.py --model google/gemma-2-9b
```

### Resume from Checkpoint
```bash
# Automatically resumes from last checkpoint
python scripts/run_production_build.py --model google/gemma-2-9b
```

### Custom Configuration
```bash
python scripts/run_production_build.py \
    --start 100 \
    --end 500 \
    --checkpoint 25 \
    --max-memory 80 \
    --max-gpu-memory 25
```

## Architecture

### Core Components

#### 1. HardeningConfig
Centralizes all configuration parameters:
```python
config = HardeningConfig(
    checkpoint_frequency=50,
    autosave_frequency=100,
    max_retries=3,
    retry_backoff=1.0,
    memory_cleanup_frequency=100,
    max_memory_usage_gb=100.0,
    max_gpu_memory_usage_gb=30.0
)
```

#### 2. CheckpointManager
Handles checkpoint saving/loading:
- Saves progress state every N records
- Atomic writes to prevent corruption
- Automatic cleanup of old checkpoints

#### 3. ProgressTracker
Enhanced progress monitoring:
- Real-time ETA calculations
- Success/failure statistics
- Retry tracking

#### 4. ResourceMonitor
System resource monitoring:
- RAM usage tracking
- GPU memory monitoring
- Automatic cleanup triggers

#### 5. HardenedDatasetBuilder
Extended DatasetBuilder with all hardening features integrated.

#### 6. ProductionMBPPTester
Production-ready orchestrator for the full pipeline.

## File Structure

```
mbpp_datasets/
├── checkpoints/
│   ├── checkpoint_20250527_120000_000000.json
│   └── checkpoint_20250527_120000_050000.json
├── autosave_20250527_120000_100.parquet
├── mbpp_dataset_20250527_180000.parquet
├── mbpp_dataset_20250527_180000.json
└── mbpp_dataset_20250527_180000_extended_metadata.json

mbpp_logs/
├── mbpp_test_20250527_120000.log
└── mbpp_production_20250527_120000.log
```

## Production Workflow

### 1. Pre-flight Checks
- Verify hardware resources (128GB RAM, 4x Tesla V100)
- Ensure sufficient disk space (~10GB for full dataset)
- Check CUDA availability

### 2. Start Production Run
```bash
python scripts/run_production_build.py --model google/gemma-2-9b
```

### 3. Monitor Progress
- Watch real-time progress bar
- Check resource usage warnings
- Monitor log file for errors

### 4. Handle Interruptions
- Press Ctrl+C for graceful shutdown
- Progress automatically saved to checkpoint
- Resume by running the same command

### 5. Post-Processing
- Verify final dataset files
- Check extended metadata for statistics
- Review failed records if any

## Error Recovery

### Checkpoint Resume
If the process is interrupted:
```bash
# Simply run the same command again
python scripts/run_production_build.py --model google/gemma-2-9b
# It will automatically detect and resume from the last checkpoint
```

### Failed Records
Records that fail after max retries are logged:
- Check `failed_records` in the extended metadata
- Review specific errors in the log file
- Can be reprocessed separately if needed

### Memory Issues
If encountering memory warnings:
- Reduce `memory_cleanup_frequency` in config
- Lower batch processing if implemented
- Monitor GPU memory allocation

## Performance Optimization

### Estimated Processing Times
- **gemma-2-2b**: ~30s per record → ~8 hours for 974 records
- **gemma-2-9b**: ~45s per record → ~12 hours for 974 records

### Resource Requirements
- **RAM**: 40-60GB typical usage (128GB available)
- **GPU**: 20-25GB per GPU (32GB available per V100)
- **Disk**: ~5-10GB for full dataset with checkpoints

### Optimization Tips
1. Run during off-peak hours
2. Close unnecessary applications
3. Use SSD for checkpoint directory
4. Monitor temperature on long runs

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce memory usage
config.memory_cleanup_frequency = 50  # More frequent cleanup
config.gc_collect_frequency = 25      # More aggressive GC
```

#### 2. Slow Processing
- Check GPU utilization with `nvidia-smi`
- Verify model is on GPU not CPU
- Check for thermal throttling

#### 3. Checkpoint Corruption
- Delete corrupted checkpoint file
- Resume will use previous checkpoint
- Check disk space availability

#### 4. Network Timeouts
- Model downloading issues
- Check internet connectivity
- Pre-download model if needed

## Configuration Reference

### HardeningConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| checkpoint_frequency | 50 | Records between checkpoints |
| autosave_frequency | 100 | Records between autosaves |
| max_retries | 3 | Max retry attempts per record |
| retry_backoff | 1.0 | Exponential backoff base (seconds) |
| memory_cleanup_frequency | 100 | Records between GPU cleanup |
| gc_collect_frequency | 50 | Records between Python GC |
| progress_log_frequency | 10 | Records between progress logs |
| max_memory_usage_gb | 100.0 | RAM warning threshold |
| max_gpu_memory_usage_gb | 30.0 | GPU memory warning threshold |
| timeout_per_record | 300.0 | Max seconds per record |

## Best Practices

1. **Always Use Checkpointing**
   - Essential for long runs
   - Minimal performance overhead
   - Saves hours on recovery

2. **Monitor Resources**
   - Watch for memory warnings
   - Check GPU utilization
   - Monitor disk space

3. **Test Before Production**
   - Run with `--test-run` first
   - Verify model loading
   - Check output format

4. **Keep Logs**
   - Logs are invaluable for debugging
   - Archive logs after successful runs
   - Review warnings and errors

5. **Backup Important Runs**
   - Copy final datasets to backup location
   - Save configuration files
   - Document any custom parameters

## Example Production Session

```bash
# 1. Clean previous runs (optional)
rm -rf mbpp_datasets/checkpoints/*
rm -f mbpp_datasets/autosave_*.parquet

# 2. Start production run
python scripts/run_production_build.py \
    --model google/gemma-2-9b \
    --checkpoint 50 \
    --max-memory 100 \
    --max-gpu-memory 30

# 3. Monitor progress
# [Progress bar shows ETA and statistics]

# 4. If interrupted, resume
python scripts/run_production_build.py --model google/gemma-2-9b

# 5. Verify results
ls -la mbpp_datasets/mbpp_dataset_*.parquet
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files for detailed errors
3. Ensure all dependencies are installed
4. Verify hardware meets requirements

Remember: The hardened pipeline is designed to handle interruptions gracefully. When in doubt, just re-run the command - it will resume from the last checkpoint automatically.