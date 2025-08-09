#!/usr/bin/env python3
"""
Unified preprocessing script for converting HuggingFace datasets into the demo
embedding-chat format compatible with `data/embedding_demo.json`.

This script supports multiple modalities and can handle various dataset formats,
converting them into the standardized format:

[
  {
    "messages": [
      {"role": "user", "content": "<m1><m1>...optional user text..."},
      {"role": "assistant", "content": "...target/label text..."}
    ],
    "embeddings": {"m1": ["/abs/or/relative/path.json", ...]}
  },
  ...
]
"""

import argparse
import json
import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from loguru import logger

random.seed(42)


class DatasetPreprocessor:
    """Main class for preprocessing HuggingFace datasets."""

    def __init__(self):
        pass

    def convert_to_embedding_demo_format(
        self,
        sample: Dict[str, Any],
        text_cols: List[str],
        target_cols: List[str],
        modality_mapping: Dict[str, str],
        data_dirs: Dict[str, str],
        include_prompts: bool = True,
    ) -> Dict[str, Any] | None:
        """
        Convert a dataset sample to the embedding demo chat format.

        Args:
            sample: Dataset sample
            text_cols: Text columns
            target_cols: Target columns
            modality_mapping: Mapping from dataset columns to modality tag names (e.g., {"ecg_files": "m1"})
            data_dirs: Dictionary mapping dataset columns to base directories for files (e.g., {"ecg_files": "/path"})

        Returns:
            Dict with keys {"messages", "embeddings"}, or None if no target content
        """
        # Concatenate text columns using " " for user text
        user_text_content = " ".join(
            [str(sample[text_col]) for text_col in text_cols if text_col in sample and sample[text_col] is not None]
        ).strip()

        # Ensure the string type for target/assistant column(s)
        assistant_text_content = "\n".join(
            [
                str(sample[target_col])
                for target_col in target_cols
                if target_col in sample and sample[target_col] is not None
            ]
        ).strip()

        if not assistant_text_content:
            # Skip examples with no target content
            return None

        # Build embeddings mapping: modality_tag -> list[str]
        embeddings: Dict[str, List[str]] = {}
        tag_prefix_parts: List[str] = []

        for dataset_col, modality_tag in modality_mapping.items():
            if not modality_tag:
                continue
            if dataset_col not in sample or sample[dataset_col] is None:
                continue

            modality_values = sample[dataset_col]
            if isinstance(modality_values, (str, bytes)):
                modality_values = [modality_values]

            base_dir = data_dirs.get(dataset_col, "")
            resolved_paths: List[str] = []
            for value in modality_values:
                value_str = str(value) + "/output.json"
                if base_dir and not os.path.isabs(value_str):
                    resolved_paths.append(os.path.join(base_dir, value_str))
                else:
                    resolved_paths.append(value_str)

            if resolved_paths:
                embeddings.setdefault(modality_tag, []).extend(resolved_paths)
                # Repeat tag per file, e.g., <m1><m1>
                tag_prefix_parts.append("".join([f"<{modality_tag}>" for _ in resolved_paths]))

        tag_prefix = "".join(tag_prefix_parts)
        user_content = f"{tag_prefix}{user_text_content}" if (user_text_content or tag_prefix) else ""

        # Optionally append a block of default ECG prompts to guide the model output
        if include_prompts:
            default_prompts = [
                "What measurements can you make from these ECG signals?",
                "Provide a one-line clinical impression based on the ECG.",
                "List the most likely differential diagnoses supported by the tracing.",
                "Draft a concise, structured ECG report summarizing the salient findings.",
            ]
            random_prompt = random.choice(default_prompts)
            if user_content:
                user_content = f"{user_content}\n\n{random_prompt}"
            else:
                user_content = f"{tag_prefix}{random_prompt}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text_content},
        ]

        return {"messages": messages, "embeddings": embeddings}

    def process_dataset(
        self,
        dataset: Dataset,
        text_cols: List[str],
        target_cols: List[str],
        modality_mapping: Dict[str, str],
        data_dirs: Dict[str, str],
        num_workers: int = 4,
        include_prompts: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process entire dataset to embedding demo format.

        Args:
            dataset: HuggingFace dataset
            modality_mapping: Mapping from dataset columns to modality names
            data_dirs: Dictionary of directories for each modality
            num_workers: Number of worker threads

        Returns:
            List of sample dicts in embedding demo format
        """
        converted_data = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample = {
                executor.submit(
                    self.convert_to_embedding_demo_format,
                    sample,
                    text_cols,
                    target_cols,
                    modality_mapping,
                    data_dirs,
                    include_prompts,
                ): sample
                for sample in dataset
            }

            for future in as_completed(future_to_sample):
                try:
                    demo_format = future.result()
                    if demo_format is not None:  # Only add examples with target content
                        converted_data.append(demo_format)
                except Exception as e:
                    logger.error(f"Error processing sample: {str(e)}")

        return converted_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to the embedding demo format (see data/embedding_demo.json)"
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
        default='["report_0", "report_1", "report_2", "report_3", "report_4", "report_5", "report_6", "report_7", "report_8", "report_9", "report_10", "report_11", "report_12", "report_13", "report_14", "report_15", "report_16", "report_17"]',
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
        default='{"ecg_files": "mimic-iv-ecg/1.0/ecgfounder_embeddings"}',
        help='JSON string mapping modality names to directories (e.g., \'{"ecg_files": "/path/to/ecg"}\')',
    )

    # Output arguments
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    parser.add_argument(
        "--output_format", type=str, choices=["json", "jsonl"], default="json", help="Output format"
    )

    # Processing arguments
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker threads")
    parser.add_argument(
        "--include_prompts",
        action="store_true",
        help="Append default ECG prompt questions to the user content",
    )
    parser.add_argument(
        "--no_include_prompts",
        action="store_true",
        help="Do not append default ECG prompt questions",
    )
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
        # Determine include_prompts effective flag (default True unless explicitly disabled)
        include_prompts = True
        if args.no_include_prompts:
            include_prompts = False
        elif args.include_prompts:
            include_prompts = True

        converted_data = preprocessor.process_dataset(
            dataset,
            text_cols,
            target_cols,
            modality_mapping,
            embeddings_dirs,
            args.num_workers,
            include_prompts,
        )

        # Save output
        logger.info(f"Saving {len(converted_data)} examples to {args.output_file}")
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if args.output_format == "json":
            with open(args.output_file, "w") as f:
                json.dump(converted_data, f, indent=2)
        elif args.output_format == "jsonl":
            with open(args.output_file, "w") as f:
                for example in converted_data:
                    f.write(json.dumps(example) + "\n")

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
