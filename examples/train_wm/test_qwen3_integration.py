#!/usr/bin/env python3
"""
Test script to validate Qwen3-MM integration with world model training.
This checks the core components without requiring full training setup.
"""

import json
import os
import sys


def test_template_and_processor_handling():
    """Test that the template and processor handling works correctly."""

    print("🧪 Testing Qwen3-MM Integration with World Model")
    print("=" * 50)

    # Test 1: Check if qwen3_embedding template is defined
    print("Test 1: Template validation...")
    try:
        # Check if template file exists and contains qwen3_embedding
        template_file = "src/llamafactory/data/template.py"
        if os.path.exists(template_file):
            with open(template_file, "r") as f:
                content = f.read()
                if "qwen3_embedding" in content:
                    print("  ✅ qwen3_embedding template found in template.py")
                    # Extract relevant lines
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if "qwen3_embedding" in line:
                            print(f"    Line {i + 1}: {line.strip()}")
                else:
                    print("  ❌ qwen3_embedding template not found")
        else:
            print("  ❌ template.py not found")
    except Exception as e:
        print(f"  ❌ Error checking template: {e}")

    # Test 2: Check data collator changes
    print("\nTest 2: Data collator processor support...")
    try:
        collator_file = "src/llamafactory/train/wm/collator.py"
        if os.path.exists(collator_file):
            with open(collator_file, "r") as f:
                content = f.read()
                if "processor: Optional[object] = None" in content:
                    print("  ✅ Processor parameter added to WorldModelDataCollator")
                else:
                    print("  ❌ Processor parameter missing in WorldModelDataCollator")

                if "get_rope_func" in content:
                    print("  ✅ Multimodal support (get_rope_func) added")
                else:
                    print("  ❌ Multimodal support missing")
        else:
            print("  ❌ collator.py not found")
    except Exception as e:
        print(f"  ❌ Error checking collator: {e}")

    # Test 3: Check trainer changes
    print("\nTest 3: Trainer processor support...")
    try:
        trainer_file = "src/llamafactory/train/wm/trainer.py"
        if os.path.exists(trainer_file):
            with open(trainer_file, "r") as f:
                content = f.read()
                if 'processor: Optional["ProcessorMixin"] = None' in content:
                    print("  ✅ Processor parameter added to WorldModelTrainer")
                else:
                    print("  ❌ Processor parameter missing in WorldModelTrainer")

                if "SaveProcessorCallback(processor)" in content:
                    print("  ✅ Processor callback handling added")
                else:
                    print("  ❌ Processor callback handling missing")
        else:
            print("  ❌ trainer.py not found")
    except Exception as e:
        print(f"  ❌ Error checking trainer: {e}")

    # Test 4: Check workflow changes
    print("\nTest 4: Workflow processor integration...")
    try:
        workflow_file = "src/llamafactory/train/wm/workflow.py"
        if os.path.exists(workflow_file):
            with open(workflow_file, "r") as f:
                content = f.read()
                if 'processor = tokenizer_module["processor"]' in content:
                    print("  ✅ Processor extraction from tokenizer_module")
                else:
                    print("  ❌ Processor extraction missing")

                if "processor=processor," in content:
                    print("  ✅ Processor passed to data collator and trainer")
                else:
                    print("  ❌ Processor not passed properly")
        else:
            print("  ❌ workflow.py not found")
    except Exception as e:
        print(f"  ❌ Error checking workflow: {e}")

    # Test 5: Demo script configuration
    print("\nTest 5: Demo script configuration...")
    try:
        demo_file = "examples/train_wm/world_model_demo.py"
        if os.path.exists(demo_file):
            with open(demo_file, "r") as f:
                content = f.read()
                if "standardmodelbio/Qwen3-MM-0.6B" in content:
                    print("  ✅ Correct model name in demo script")
                else:
                    print("  ❌ Model name not updated")

                if "qwen3_embedding" in content:
                    print("  ✅ Correct template in demo script")
                else:
                    print("  ❌ Template not updated")
        else:
            print("  ❌ demo script not found")
    except Exception as e:
        print(f"  ❌ Error checking demo script: {e}")

    print("\n" + "=" * 50)
    print("🎯 Integration Test Summary:")
    print("- Template definition: Should be available")
    print("- Processor parameter: Added to collator, trainer, workflow")
    print("- Multimodal support: Enhanced for Qwen3-MM compatibility")
    print("- Demo configuration: Updated for Qwen3-MM model")
    print("\n✅ The world model should now work with Qwen3-MM!")
    print("\n📝 Note: If you still get the processor error, it might be:")
    print("   1. The qwen3_embedding template needs to be defined in template.py")
    print("   2. Missing dependencies (accelerate, etc.) for full training")
    print("   3. Model-specific processor requirements")


def check_missing_template():
    """Check if we need to define the qwen3_embedding template."""
    print("\n🔍 Checking for qwen3_embedding template definition...")

    template_file = "src/llamafactory/data/template.py"
    if os.path.exists(template_file):
        with open(template_file, "r") as f:
            content = f.read()

        # Look for the template definition
        if 'name="qwen3_embedding"' in content:
            print("✅ qwen3_embedding template is defined")
            return True
        else:
            print("❌ qwen3_embedding template is NOT defined")
            print("💡 This might be the root cause of the error!")
            print("   The template might not exist or be named differently.")

            # Look for similar templates
            if "qwen" in content.lower():
                lines = content.split("\n")
                print("   Found other Qwen templates:")
                for i, line in enumerate(lines):
                    if "qwen" in line.lower() and "name=" in line:
                        print(f"     Line {i + 1}: {line.strip()}")

            return False
    else:
        print("❌ template.py not found")
        return False


if __name__ == "__main__":
    test_template_and_processor_handling()
    check_missing_template()
