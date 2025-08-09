#!/usr/bin/env python3
"""
Unified preprocessing script for converting HuggingFace datasets into OpenAI input format
compatible with the SMB-MM multimodal model.

This script supports multiple modalities and can handle various dataset formats,
converting them into the standardized OpenAI chat format with modality embeddings.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from loguru import logger


class DatasetPreprocessor:
    """Main class for preprocessing HuggingFace datasets."""

    def __init__(self):
        pass

    def convert_to_openai_format(
        self,
        sample: Dict[str, Any],
        text_cols: List[str],
        target_cols: List[str],
        modality_mapping: Dict[str, str],
        data_dirs: Dict[str, str],
    ) -> List[Dict[str, Any]] | None:
        """
        Convert a dataset sample to OpenAI chat format.

        Args:
            sample: Dataset sample
            text_cols: Text columns
            target_cols: Target columns
            modality_mapping: Mapping from dataset columns to modality names
            data_dirs: Dictionary of directories for each modality

        Returns:
            List of messages in OpenAI format, or None if no target content
        """
        messages = []

        # Concatenate text columns using " "
        text_content = " ".join(
            [str(sample[text_col]) for text_col in text_cols if text_col in sample and sample[text_col] is not None]
        )
        if text_content:
            messages.append({"role": "user", "content": [{"type": "text", "text": text_content}]})

        # Ensure the string type for target column
        target_content = " ".join(
            [
                str(sample[target_col])
                for target_col in target_cols
                if target_col in sample and sample[target_col] is not None
            ]
        )
        if target_content:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target_content}]})
        else:
            # Skip examples with no target content
            return None

        # Add modality embeddings to user messages
        for dataset_col, modality_name in modality_mapping.items():
            if dataset_col in sample and modality_name:
                modality_data = sample[dataset_col]

                # Handle file paths
                for data in modality_data:
                    if isinstance(data, str):
                        data_dir = data_dirs.get(dataset_col, "")
                        if data_dir and not os.path.isabs(data):
                            modality_path = os.path.join(data_dir, data)
                    else:
                        modality_path = data

                    # Add modality to the first user message
                    for msg in messages:
                        if msg["role"] == "user":
                            msg["content"].append({"type": modality_name, modality_name: modality_path})
                            break
                    else:
                        # If no user message exists, create one
                        messages.insert(
                            0, {"role": "user", "content": [{"type": modality_name, modality_name: modality_path}]}
                        )

        return messages

    def process_dataset(
        self,
        dataset: Dataset,
        text_cols: List[str],
        target_cols: List[str],
        modality_mapping: Dict[str, str],
        data_dirs: Dict[str, str],
        num_workers: int = 4,
    ) -> List[List[Dict[str, Any]]]:
        """
        Process entire dataset to OpenAI format.

        Args:
            dataset: HuggingFace dataset
            modality_mapping: Mapping from dataset columns to modality names
            data_dirs: Dictionary of directories for each modality
            num_workers: Number of worker threads

        Returns:
            List of conversations in OpenAI format
        """
        converted_data = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample = {
                executor.submit(
                    self.convert_to_openai_format,
                    sample,
                    text_cols,
                    target_cols,
                    modality_mapping,
                    data_dirs,
                ): sample
                for sample in dataset
            }

            for future in as_completed(future_to_sample):
                try:
                    openai_format = future.result()
                    if openai_format is not None:  # Only add conversations with target content
                        converted_data.append(openai_format)
                except Exception as e:
                    logger.error(f"Error processing sample: {str(e)}")

        return converted_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to OpenAI format for SMB-MM multimodal model"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="zekaic/mimic-iv-ecg-dataset",
        help="HuggingFace dataset name or path to local dataset",
    )
    parser.add_argument(
        "--text_cols", type=str, default='["clinical_notes", "radiotherapy_dose"]', help="Text column name"
    )
    parser.add_argument(
        "--target_cols",
        type=str,
        default='["report_0", "report_1", "report_2", "report_3", "report_4", "report_5", "report_6", "report_7", "report_8", "report_9", "report_10"]',
        help="Target column name",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process")

    # Modality configuration
    parser.add_argument(
        "--modality_mapping",
        type=str,
        default='{"ecg_files": "m1"}',
        help='JSON string mapping dataset columns to modality names (e.g., \'{"ecg_files": "m1"}\')',
    )

    # Data directory for modality files
    parser.add_argument(
        "--embeddings_dirs",
        type=str,
        default='{"ecg_files": "/workspace/mimic-iv-ecg/1.0/ecgfounder_embeddings"}',
        help='JSON string mapping modality names to directories (e.g., \'{"ecg_files": "/path/to/ecg"}\')',
    )

    # Output arguments
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    parser.add_argument(
        "--output_format", type=str, choices=["openai", "jsonl"], default="openai", help="Output format"
    )

    # Processing arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")

    # Logging
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    try:
        # Parse modality mapping and config
        modality_mapping = json.loads(args.modality_mapping)
        embeddings_dirs = json.loads(args.embeddings_dirs)
        text_cols = json.loads(args.text_cols)
        target_cols = json.loads(args.target_cols)

        # Load dataset
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split=args.split)

        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Create preprocessor
        preprocessor = DatasetPreprocessor()

        # Process dataset
        logger.info("Processing dataset...")
        converted_data = preprocessor.process_dataset(
            dataset,
            text_cols,
            target_cols,
            modality_mapping,
            embeddings_dirs,
            args.num_workers,
        )

        # Save output
        logger.info(f"Saving {len(converted_data)} conversations to {args.output_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        if args.output_format == "openai":
            with open(args.output_file, "w") as f:
                json.dump(converted_data, f, indent=2)
        elif args.output_format == "jsonl":
            with open(args.output_file, "w") as f:
                for conversation in converted_data:
                    f.write(json.dumps(conversation) + "\n")

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
