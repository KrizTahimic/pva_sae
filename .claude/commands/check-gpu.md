---
description: Quick GPU status check showing memory and running processes
allowed-tools: Bash(nvidia-smi*)
---

Check GPU status and present a clean summary.

Run these two commands:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
```

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || echo "No GPU processes running"
```

Format the output as a readable table showing:
- GPU index
- GPU name
- Memory used / total (with percentage)
- GPU utilization %
- Temperature
- Any running processes (PID + memory usage)

End with a one-line summary: how many GPUs are free (>20GB available) vs busy.
