#!/usr/bin/env python3
"""
Debug script for world model training to isolate issues step by step.
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_step_by_step():
    """Test each component step by step to isolate the issue."""

    print("🔍 World Model Training Debug")
    print("=" * 40)

    # Step 1: Test imports
    print("Step 1: Testing imports...")
    try:
        from llamafactory.train.tuner import run_exp

        print("  ✅ run_exp imported successfully")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return

    # Step 2: Test argument parsing
    print("\nStep 2: Testing argument parsing...")
    try:
        args = [
            "--stage",
            "wm",
            "--do_train",
            "True",
            "--model_name",
            "standardmodelbio/Qwen3-MM-0.6B",
            "--template",
            "qwen3_embedding",
            "--dataset",
            "wm_demo",
            "--cutoff_len",
            "512",
            "--learning_rate",
            "1e-4",
            "--num_train_epochs",
            "0.1",  # Very short for testing
            "--max_samples",
            "5",  # Very small for testing
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--lr_scheduler_type",
            "cosine",
            "--warmup_ratio",
            "0.1",
            "--bf16",
            "True",
            "--logging_steps",
            "1",
            "--save_steps",
            "100",
            "--output_dir",
            "./saves/world_model_debug",
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
            "--embedding_loss_weight",
            "1.0",
            "--lm_loss_weight",
            "0.1",
            "--embedding_dim",
            "4",
            "--world_model_loss_type",
            "l2",
            "--plot_loss",
            "True",
        ]
        print(f"  ✅ Arguments prepared: {len(args)} args")
    except Exception as e:
        print(f"  ❌ Argument preparation failed: {e}")
        return

    # Step 3: Test run_exp with detailed error handling
    print("\nStep 3: Testing run_exp...")
    try:
        print("  📊 Starting training...")
        run_exp(args=args)
        print("  ✅ Training completed successfully!")
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        import traceback

        print(f"  📝 Full traceback:\n{traceback.format_exc()}")
        return

    # Step 4: Check outputs
    print("\nStep 4: Checking outputs...")
    try:
        output_dir = "./saves/world_model_debug"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"  ✅ Output directory created with {len(files)} files")
            for f in files[:5]:  # Show first 5 files
                print(f"    - {f}")
        else:
            print("  ⚠️ No output directory created")
    except Exception as e:
        print(f"  ❌ Output check failed: {e}")


if __name__ == "__main__":
    test_step_by_step()
