# GPU Optimization Notes for PVA-SAE

## Your System Configuration
- **GPUs**: 3x Tesla V100-PCIE-32GB (not 4 as originally planned)
- **Total GPU Memory**: 95.1GB (3 × 31.7GB)
- **CPU Memory**: 128GB total

## Multi-GPU Parallelism Strategy

The project uses **process-level parallelism** instead of batching for optimal GPU utilization:

### Architecture:
- **Sequential Processing**: Each GPU processes one problem at a time
- **Multi-Process Parallelism**: 3 separate processes, each assigned to a different GPU
- **Range Distribution**: Dataset is split across GPUs by index ranges

### Testing Commands:
```bash
# Test GPU setup
python3 run.py test-gpu --detailed

# Sequential processing (single GPU)
python3 run.py phase 1 --start 0 --end 10

# Multi-GPU parallel processing (recommended)
python3 multi_gpu_launcher.py --start 0 --end 17 --model google/gemma-2-9b

# Production run with 3 GPUs
python3 multi_gpu_launcher.py --start 0 --end 973 --model google/gemma-2-9b
```

## Memory Monitoring
Monitor GPU usage during runs:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

## Expected Performance
- **Sequential (1 GPU)**: ~1 problem per 2-3 seconds
- **Multi-GPU (3 GPUs)**: ~3 problems per 2-3 seconds (3x speedup)
- **Effective Throughput**: ~3600-5400 problems per hour with 3 GPUs

## Why No Batching?
- **Memory Efficiency**: Sequential processing avoids memory spikes
- **Streaming Support**: Enables real-time output streaming
- **Error Isolation**: Failures don't affect entire batches
- **Simpler Architecture**: One clear processing path