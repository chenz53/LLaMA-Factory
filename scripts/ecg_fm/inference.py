import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader


# Ensure the parent directory (scripts) is on PYTHONPATH so we can import ecg_fm modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SCRIPTS_DIR = SCRIPT_DIR.parent  # .../scripts
sys.path.append(str(PROJECT_SCRIPTS_DIR))

from ecg_fm.ecgfounder.dataset import ECGDataset  # type: ignore
from ecg_fm.ecgfounder.model import ft_1lead_ECGFounder, ft_12lead_ECGFounder  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECGFounder inference and save embeddings as JSON files.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset name, script path or local dataset directory"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--data_dir", type=str, default="", help="Directory where ECG files are stored (default: '')")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory where JSON embedding files will be written"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ECGFounder checkpoint (.ckpt or .pth)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument(
        "--mode", type=str, choices=["12lead", "1lead"], default="12lead", help="ECG mode (default: 12lead)"
    )
    args = parser.parse_args()
    return args


def flatten_ecg_paths(list_of_lists: List[List[str]]) -> List[str]:
    """Flatten nested list of ECG paths coming from the dataset."""
    flat: List[str] = []
    for paths in list_of_lists:
        flat.extend(paths)
    return flat


def collate_fn(batch):
    """Custom collate fn that pads variable-length signals inside a batch."""
    signals = [item["ecg_data"] for item in batch]
    paths = [item["ecg_path"] for item in batch]
    lengths = [sig.shape[1] for sig in signals]
    ecg_data = torch.stack(signals)

    return {"ecg_data": ecg_data, "ecg_path": paths, "lengths": lengths}


def save_embedding(embedding_tensor: torch.FloatTensor, ecg_path: str, output_dir: str):
    """Save embedding tensor (1D) to a JSON file following the sample format."""
    embedding = embedding_tensor.cpu().tolist()
    description = f"ECG embedding extracted from {os.path.basename(ecg_path)} using ECGFounder"
    data = {
        "embedding": embedding,
        "shape": embedding_tensor.shape,
        "description": description,
        "modality": "ecg",
    }

    basename = Path(ecg_path).stem
    json_name = basename + ".json"
    out_path = Path(output_dir) / json_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    accelerator = Accelerator()

    # Create output directory (only once)
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load HuggingFace dataset
    if os.path.isdir(args.dataset):
        hf_ds = load_from_disk(args.dataset, keep_in_memory=False)
    else:
        # Could be a dataset hub name or script path
        hf_ds = load_dataset(args.dataset, split=args.split)

    # Ensure column exists
    if "ecg_paths" not in hf_ds.column_names:
        raise ValueError("Expected 'ecg_paths' column in the dataset")

    ecg_paths_nested: List[List[str]] = hf_ds["ecg_paths"]  # type: ignore
    ecg_paths: List[str] = flatten_ecg_paths(ecg_paths_nested)

    # Build torch dataset & dataloader
    ecg_dataset = ECGDataset(data_dir=args.data_dir, ecg_paths=ecg_paths, mode=args.mode)
    dataloader = DataLoader(
        ecg_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Build model
    device = accelerator.device
    if args.mode == "12lead":
        model = ft_12lead_ECGFounder(device=device, pth=args.checkpoint, n_classes=1, linear_prob=False)
    elif args.mode == "1lead":
        model = ft_1lead_ECGFounder(device=device, pth=args.checkpoint, n_classes=1, linear_prob=False)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}. Use '1lead' or '12lead'")

    model.eval()

    # Prepare everything with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    with torch.no_grad():
        for batch in dataloader:
            signals = batch["ecg_data"].to(device)
            _, features = model(signals)

            # Iterate through items in batch and save
            for emb, ecg_path in zip(features, batch["ecg_path"]):
                save_embedding(emb, ecg_path, args.output_dir)

    accelerator.print("Inference completed. Embeddings saved to", args.output_dir)


if __name__ == "__main__":
    main()
