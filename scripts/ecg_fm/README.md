### ECG Founder Inference (Embeddings Extraction)

This folder contains a minimal pipeline to extract ECG embeddings using the ECGFounder model and save them as JSON files, ready for downstream multimodal workflows.

The main entry point is `inference.py`, which:
- Loads a HuggingFace dataset that provides ECG file paths via an `ecg_paths` column
- Reads ECG signals from disk using `sierraecg`
- Runs the ECGFounder model (`1lead` or `12lead`) using PyTorch/Accelerate
- Writes one JSON per ECG file to an output directory

---

## Install dependencies (pysierraecg):

```bash
cd scripts/ecg_fm/pysierraecg
pip install -e .
```

Notes:
- GPU is optional but recommended. Install a CUDA build of PyTorch if available.
- `sierraecg` is used to read ECG files (e.g., Philips/GE XML). Ensure your data is compatible.

---

## Expected Dataset Format

`inference.py` expects either a HuggingFace dataset name (from the Hub) or a local dataset directory created with `datasets.save_to_disk(...)`.

Required column:
- `ecg_paths`: List[List[str]] or List[str]
  - Each row may contain one or more file paths (relative to `--data_dir` or absolute). The script flattens all rows and processes every file.

Example: build a tiny local dataset with two ECG XML files and save it to disk.

```python
from datasets import Dataset

data = {
  "ecg_paths": [["patient_001/ecg_0001.xml"], ["patient_002/ecg_0007.xml"]]
}
ds = Dataset.from_dict(data)
ds.save_to_disk("/abs/path/to/my_ecg_hfds")
```

Directory layout on disk:
- `--data_dir` (e.g., `/abs/path/to/cohort/`) should contain the actual files referenced by `ecg_paths` (unless paths are absolute).

---

## Model Checkpoint

Provide a `.ckpt` or `.pth` checkpoint via `--checkpoint`. The loader expects a dictionary with a `"state_dict"` key (e.g., a PyTorch Lightning checkpoint). Only the backbone weights are loaded; the final linear layer is recreated for `n_classes=1` and not used for training during inference.

ECGFounder can be downloaded from: https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main

use CLI: `hf download PKUDigitalHealth/ECGFounder /abs/path/to/ecgfounder_checkpoints`

---

## Run Inference

Single GPU / CPU:

```bash
python scripts/ecg_fm/inference.py \
  --dataset /abs/path/to/my_ecg_hfds \
  --split train \
  --data_dir /abs/path/to/cohort/ecg_data \
  --output_dir /abs/path/to/ecgfounder_embeddings \
  --checkpoint /abs/path/to/ecgfounder_checkpoints/12_lead_ECGFounder.pth \
  --mode 12lead \
  --batch_size 8 \
  --num_workers 4
```

Multi-GPU (via Accelerate):

```bash
accelerate launch --num_processes 2 --multi_gpu \
  scripts/ecg_fm/inference.py \
  --dataset /abs/path/to/my_ecg_hfds \
  --split train \
  --data_dir /abs/path/to/cohort/ecg_data \
  --output_dir /abs/path/to/ecgfounder_embeddings \
  --checkpoint /abs/path/to/ecgfounder_checkpoints/12_lead_ECGFounder.pth \
  --mode 12lead \
  --batch_size 8 \
  --num_workers 4
```

Arguments:
- `--dataset`: HF dataset name or local `load_from_disk` directory
- `--split`: Split to use when loading from the HF Hub
- `--data_dir`: Base directory for ECG files when `ecg_paths` are relative
- `--output_dir`: Where JSON embeddings are saved
- `--checkpoint`: Path to ECGFounder checkpoint (`.ckpt` or `.pth`)
- `--mode`: `12lead` or `1lead`
- `--batch_size`, `--num_workers`: Dataloader params per process

---

## Input Signal Handling

- Signals are read with `sierraecg.read_file(path)` and converted to a tensor of shape `[n_leads, n_samples]`.
- Z-score normalization is applied per-file.
- For `--mode 12lead`, inputs are expected to be 12 standard leads; for `--mode 1lead`, a single lead is used.
- Batches are padded to the longest length in the batch; padding length is not currently trimmed in the saved embeddings.

---

## Output Format

One JSON file is written per input ECG file. The output file name is derived from the ECG filename (e.g., `deid_3.xml` -> `deid_3.json`). Example:

```json
{
  "embedding": [[...], [...], ...],
  "shape": [T, C],
  "description": "ECG embedding extracted from deid_3.xml using ECGFounder",
  "modality": "ecg"
}
```

Notes:
- `embedding` is a 2D list with shape `[T, C]` (sequence length by feature dimension) produced by the backbone features.
- `T` may include padding introduced at the batch level. If you need exact lengths, track them externally or post-process with your own trimming.

---

## Utilities in `sample_data/` (optional)

These helpers demonstrate how to build downstream inputs once embeddings exist:
- `create_dummy_dataset.py`: converts a dataset to a generic OpenAI-style chat format with modality references.
- `create_mimic_ecg_inputs.py`: converts to a demo embedding-chat format similar to `data/embedding_demo.json`.

They both assume your `--output_dir` already contains per-file JSON embeddings and then map dataset rows to those files.

---

## Troubleshooting

- "Expected 'ecg_paths' column in the dataset": ensure your dataset contains that column. It can be a list of strings or a list of list-of-strings.
- `ModuleNotFoundError: sierraecg`: install `sierraecg` and verify that your ECG file format is supported.
- Checkpoint errors (`KeyError: 'state_dict'`): the loader expects a checkpoint dict with `state_dict`. Convert or re-export your checkpoint accordingly.
- CUDA OOM: lower `--batch_size` and/or use fewer GPUs. Consider splitting the dataset.
- Mixed absolute/relative paths: when `ecg_paths` are relative, set `--data_dir` to the correct root.

---

## File Map

- `inference.py`: main inference script
- `ecgfounder/dataset.py`: ECG file reader and normalization
- `ecgfounder/model.py`: wrappers to create 1-lead/12-lead ECGFounder models and load checkpoints
- `ecgfounder/net1d.py`: 1D CNN backbone (with feature extractor)
- `sample_data/*`: optional dataset-to-chat conversion helpers


