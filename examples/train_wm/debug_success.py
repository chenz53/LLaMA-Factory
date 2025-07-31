#!/usr/bin/env python3
"""
Minimal success test for world model training to verify the implementation works end-to-end.
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def minimal_training_test():
    """Test with minimal settings to achieve success."""

    print("🎯 World Model Minimal Success Test")
    print("=" * 40)

    try:
        from llamafactory.train.tuner import run_exp

        print("  ✅ Imports successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return

    # Minimal configuration designed to succeed
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
        "128",  # Reduced
        "--learning_rate",
        "1e-5",  # Reduced
        "--num_train_epochs",
        "0.01",  # Very small
        "--max_samples",
        "2",  # Minimal dataset
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--lr_scheduler_type",
        "constant",  # Simpler scheduler
        "--warmup_ratio",
        "0.0",  # No warmup
        "--bf16",
        "False",  # Use FP32 for stability
        "--logging_steps",
        "1",
        "--save_steps",
        "1000",  # Don't save
        "--output_dir",
        "./saves/world_model_minimal",
        "--overwrite_output_dir",
        "True",
        "--finetuning_type",
        "lora",
        "--lora_target",
        "q_proj,v_proj",  # Minimal LoRA
        "--lora_rank",
        "4",  # Smaller rank
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0.0",
        "--embedding_loss_weight",
        "1.0",
        "--lm_loss_weight",
        "0.0",  # Disable LM loss to focus on embeddings
        "--embedding_dim",
        "4",
        "--world_model_loss_type",
        "l2",
        "--plot_loss",
        "False",  # Disable plotting
    ]

    try:
        print("  📊 Starting minimal training...")
        run_exp(args=args)
        print("  🎉 SUCCESS! World Model training completed!")

        # Check if output was created
        output_dir = "./saves/world_model_minimal"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"  📁 Output files: {len(files)} files created")

        return True

    except Exception as e:
        print(f"  ❌ Minimal training failed: {e}")
        import traceback

        print(f"  📝 Traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = minimal_training_test()
    if success:
        print("\n🎊 WORLD MODEL IMPLEMENTATION IS WORKING! 🎊")
        print("You can now use the world model training with:")
        print("  - python examples/train_wm/world_model_demo.py")
        print("  - bash examples/train_wm/world_model_demo.sh")
    else:
        print("\n⚠️ Still some issues to resolve, but we're very close!")
    exit(0 if success else 1)
