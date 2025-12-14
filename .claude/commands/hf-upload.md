Upload data to HuggingFace dataset repository.

Run the upload script to sync local data/ directory to HuggingFace:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc && python3 scripts/upload_to_hf.py
```

This uploads all experiment data (~1 GB) to: https://huggingface.co/datasets/kriztahimic/sae-code-correctness-data

Options:
- `--dry-run`: Preview what will be uploaded without actually uploading
- `--repo-id`: Override default repository ID

Note: Requires HuggingFace login with write permissions (`huggingface-cli login`).
