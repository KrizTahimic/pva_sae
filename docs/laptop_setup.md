# Laptop Setup

Step-by-step guide to restore this project on a new machine (e.g. laptop after GCP access ends).

## Prerequisites

- conda installed ([miniconda](https://docs.conda.io/en/latest/miniconda.html))
- `huggingface-cli` authenticated: `huggingface-cli login`
- ~50GB free disk space

## Steps

### 1. Clone repo

```bash
git clone git@github.com:KrizTahimic/sae-code-correctness.git
cd sae-code-correctness
git checkout icml
```

### 2. Create conda env

```bash
conda env create -f environment.yml
# Or if environment.yml is missing:
# conda create -n sae_cc python=3.11 && conda activate sae_cc && pip install -r requirements.txt
conda activate sae_cc
```

### 3. Download data from HuggingFace

```bash
python3 scripts/download_from_hf.py
```

This downloads:
- **156 normal phases** via `snapshot_download` (individual files directly into `data/`)
- **8 large phases** as tar archives (downloads to `/tmp/`, extracts to `data/`, deletes tar)

Resumable — re-run if interrupted. Progress tracked in `data/.download_progress.json`.

To download normal phases only (skip large archives):

```bash
python3 scripts/download_from_hf.py --skip-archives
```

### 4. Download model weights (optional)

Only needed to run generation phases (Phases 1, 3.5, 4.5, etc.). Not needed for `--viz-only`.

Models auto-download on first run via HuggingFace. To pre-download:

```bash
# Gemma-2B (~5GB)
huggingface-cli download google/gemma-2-2b

# Gemma-9B (~18GB)
huggingface-cli download google/gemma-2-9b

# LLaMA-8B (~16GB)
huggingface-cli download meta-llama/Llama-3.1-8B
```

### 5. Verify setup

```bash
# Regenerate plots without GPU (seconds, not hours)
conda activate sae_cc
python3 run.py phase 3.8 --viz-only
python3 run.py phase 4.8 --viz-only
```

## Troubleshooting

**Download interrupted?** Re-run `python3 scripts/download_from_hf.py` — it resumes from where it left off.

**Missing conda env?** Check `environment.yml` exists; otherwise install from `requirements.txt`.

**HF auth error?** Run `huggingface-cli login` and paste your token from https://huggingface.co/settings/tokens.

**OOM on generation?** Add `--start 0 --end 10` to test on subset, or reduce batch sizes in `common/config.py`.
