# GPU Optimization Notes for PVA-SAE

## Your System Configuration
- **GPUs**: 3x Tesla V100-PCIE-32GB (not 4 as originally planned)
- **Total GPU Memory**: 95.1GB (3 × 31.7GB)
- **CPU Memory**: 128GB total

## Recommended Batch Sizes

### For 3 GPUs:
- **Conservative**: `--batch-size 12 --num-gpus 3` (4 per GPU)
- **Optimal**: `--batch-size 18 --num-gpus 3` (6 per GPU)
- **Aggressive**: `--batch-size 24 --num-gpus 3` (8 per GPU)

### Testing Commands:
```bash
# Test GPU setup
python3 run.py test-gpu --detailed

# Sequential baseline (for comparison)
python3 run.py phase 1 --start 0 --end 10

# Single GPU batch processing
python3 run.py phase 1 --start 0 --end 7 --batch-size 8

# Multi-GPU batch processing (3 GPUs)
python3 run.py phase 1 --start 0 --end 17 --batch-size 18 --num-gpus 3

# Production run with 3 GPUs
python3 run.py phase 1 --start 0 --end 973 --batch-size 18 --num-gpus 3
```

## Memory Monitoring
Monitor GPU usage during runs:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

## Expected Performance
- **Sequential**: ~1 problem per 2-3 seconds
- **Single GPU batch=8**: ~4 problems per 3-4 seconds
- **3 GPUs batch=18**: ~18 problems per 4-5 seconds (10-12x speedup)