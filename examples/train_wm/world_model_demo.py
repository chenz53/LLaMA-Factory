#!/usr/bin/env python3
# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demo script for World Model fine-tuning with LLaMA Factory.

This script demonstrates how to use the world model training stage to train
a model with embedding-based autoregression instead of token-level autoregression.

Usage:
    python examples/train_wm/world_model_demo.py

The script uses the embedding_demo.json dataset as an example.
"""

import os
import sys


# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llamafactory.train.tuner import run_exp


def main():
    """Run world model training demo."""

    # Configuration for world model training
    args = [
        "--stage",
        "wm",
        "--do_train",
        "True",
        "--model_name",
        "standardmodelbio/Qwen3-MM-0.6B",  # Use a small model for demo
        "--template",
        "qwen3_embedding",
        "--dataset",
        "wm_demo",  # Changed from embedding_demo to wm_demo
        "--cutoff_len",
        "512",
        "--learning_rate",
        "1e-4",
        "--num_train_epochs",
        "2.0",
        "--max_samples",
        "100",
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "1",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.1",
        "--bf16",
        "True",
        "--logging_steps",
        "5",
        "--save_steps",
        "100",
        "--output_dir",
        "./saves/world_model_demo",
        "--overwrite_output_dir",
        "True",
        "--finetuning_type",
        "lora",
        "--lora_target",
        "all",
        "--lora_rank",
        "8",
        "--lora_alpha",
        "16",
        "--lora_dropout",
        "0.05",
        # World model specific parameters
        "--embedding_loss_weight",
        "1.0",
        "--lm_loss_weight",
        "0.1",  # Set to 0.0 to avoid batch size mismatch and focus on pure embedding prediction
        "--embedding_dim",
        "4",
        "--world_model_loss_type",
        "l2",
        # Embedding demo specific
        "--plot_loss",
        "True",
    ]

    print("🦙🏭 Starting World Model Training Demo")
    print("=" * 50)
    print(f"Dataset: wm_demo.json (Patient temporal progression)")
    print(f"Model: standardmodelbio/Qwen3-MM-0.6B")
    print(f"Training Stage: World Model (wm)")
    print(f"Loss Type: L2 Reconstruction")
    print(f"Output: ./saves/world_model_demo")
    print("=" * 50)

    # try:
    # Run the training
    run_exp(args=args)
    print("\n✅ World Model Training completed successfully!")
    print("📊 Check ./saves/world_model_demo for results")

    # except Exception as e:
    #     print(f"\n❌ Training failed with error: {e}")
    #     return 1

    # return 0


if __name__ == "__main__":
    exit(main())
