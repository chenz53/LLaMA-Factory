#!/usr/bin/env python3
"""
Test script to display what the current embedding-wise inputs look like
before feeding into the model in LLaMA-Factory.

This script demonstrates the complete embedding processing pipeline from
JSON files to final model inputs using the qwen3_embedding plugin.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

def display_json_structure_simple():
    """Display the structure of the wm_demo.json file - simplified version"""
    print("🗂️  WM_DEMO.JSON FILE STRUCTURE")
    print("=" * 60)

    # Load the main demo file - using wm_demo.json
    demo_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wm_demo.json")
    
    if not os.path.exists(demo_file_path):
        print(f"❌ Demo file not found: {demo_file_path}")
        return
        
    with open(demo_file_path, "rb") as f:
        demo_data = json.load(f)

    print(f"📄 Main demo file: {demo_file_path}")
    print(f"   Number of examples: {len(demo_data)}")
    print(f"   Structure: List of dictionaries with 'messages' and 'embeddings' keys")
    print()

    # Show first example
    print("📋 Example 1:")
    example = demo_data[0]
    print(f"   Messages: {len(example['messages'])} messages")
    for i, msg in enumerate(example["messages"]):
        print(f"     {i + 1}. {msg['role']}: {msg['content']}")
    print(f"   Embeddings structure: {type(example['embeddings'])}")
    print(f"   Embedding modalities: {list(example['embeddings'].keys())}")
    print()

    # Load and display individual embedding files from the modality mapping
    print("📁 Individual embedding files:")
    for modality, embedding_files in example["embeddings"].items():
        print(f"   Modality '{modality}':")
        for i, embedding_file in enumerate(embedding_files):
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", embedding_file)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    embedding_data = json.load(f)

                print(f"     {i + 1}. {embedding_file}:")
                print(f"        Shape: {embedding_data['shape']}")
                print(f"        Description: {embedding_data['description']}")
                print(f"        Modality: {embedding_data['modality']}")
                # Calculate num_tokens and embedding_dim from shape
                shape = embedding_data['shape']
                num_tokens = shape[0] if len(shape) > 0 else 0
                embedding_dim = shape[1] if len(shape) > 1 else 0
                print(f"        Num tokens: {num_tokens}")
                print(f"        Embedding dim: {embedding_dim}")
                print(f"        Sample values: {embedding_data['embedding'][0][:4]}...")
            else:
                print(f"     {i + 1}. {embedding_file}: ❌ FILE NOT FOUND")
            print()

# Only import PyTorch-dependent modules if we need the full functionality
try:
    import torch
    TORCH_AVAILABLE = True
    
    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.model import load_tokenizer
    from llamafactory.hparams import get_infer_args
    
except ImportError as e:
    print(f"⚠️  PyTorch dependencies not available: {e}")
    print("   Running in simplified mode - showing JSON structure only")
    TORCH_AVAILABLE = False


def get_test_embedding_plugin():
    """Get the qwen3_embedding plugin configured for testing"""
    return get_mm_plugin(
        name="qwen3_embedding",
        image_token=None,
        video_token=None,
        audio_token=None,
        embedding_tokens={"m1": "<m1>", "m2": "<m2>"},
    )


def display_regularization_process():
    """Display the embedding regularization process"""
    print("⚙️  EMBEDDING REGULARIZATION PROCESS")
    print("=" * 60)

    plugin = get_test_embedding_plugin()

    # Load embedding files
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    embedding_files = [
        os.path.join(base_path, "embedding_demo_data", "sample_embedding_1.json"),
        os.path.join(base_path, "embedding_demo_data", "sample_embedding_2.json"),
    ]

    print("🔄 Input embedding files:")
    for i, file_path in enumerate(embedding_files):
        print(f"   {i + 1}. {file_path}")
    print()

    # Process through regularization
    print("🔧 Processing through _regularize_embeddings:")
    embedding_data = plugin._regularize_embeddings(embedding_files)

    print(f"📤 Output structure:")
    print(f"   Keys: {list(embedding_data.keys())}")
    print(f"   Number of embeddings: {len(embedding_data['embeddings'])}")
    print(f"   Number of shapes: {len(embedding_data['shapes'])}")
    print()

    # Display each processed embedding
    for i, (embedding_tensor, shape) in enumerate(zip(embedding_data["embeddings"], embedding_data["shapes"])):
        print(f"📊 Embedding {i + 1}:")
        print(f"   Original shape: {shape}")
        print(f"   Tensor shape: {embedding_tensor.shape}")
        print(f"   Tensor dtype: {embedding_tensor.dtype}")
        print(f"   Tensor device: {embedding_tensor.device}")
        print(f"   Min value: {embedding_tensor.min().item():.4f}")
        print(f"   Max value: {embedding_tensor.max().item():.4f}")
        print(f"   Mean value: {embedding_tensor.mean().item():.4f}")
        print(f"   First few values: {embedding_tensor.flatten()[:8].tolist()}")
        print()


def display_multimodal_inputs():
    """Display the final multimodal inputs"""
    print("🎯 FINAL MULTIMODAL INPUTS")
    print("=" * 60)

    plugin = get_test_embedding_plugin()

    # Load embedding files
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    embedding_files = [
        os.path.join(base_path, "embedding_demo_data", "sample_embedding_1.json"),
        os.path.join(base_path, "embedding_demo_data", "sample_embedding_2.json"),
    ]

    print("🔄 Processing embeddings through _get_mm_inputs:")
    mm_inputs = plugin._get_mm_inputs(images=[], videos=[], audios=[], embeddings=embedding_files, processor=None)

    print(f"📤 Final mm_inputs dictionary:")
    print(f"   Keys: {list(mm_inputs.keys())}")
    print()

    for key, value in mm_inputs.items():
        print(f"🔑 {key}:")
        if isinstance(value, torch.Tensor):
            print(f"   Type: torch.Tensor")
            print(f"   Shape: {value.shape}")
            print(f"   Dtype: {value.dtype}")
            print(f"   Device: {value.device}")
            print(f"   Min: {value.min().item():.4f}")
            print(f"   Max: {value.max().item():.4f}")
            print(f"   Mean: {value.mean().item():.4f}")
            if value.numel() <= 50:
                print(f"   Values:\n{value}")
            else:
                print(f"   Sample values: {value.flatten()[:10].tolist()}")
        else:
            print(f"   Type: {type(value)}")
            print(f"   Value: {value}")
        print()


def display_batching_example():
    """Display how multiple embeddings are batched"""
    print("📦 BATCHING MULTIPLE EMBEDDINGS")
    print("=" * 60)

    plugin = get_test_embedding_plugin()

    # Example with different sized embeddings
    embeddings = [
        {
            "m1": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "shape": [2, 4],
            "description": "Small embedding example",
            "modality": "text",
        },
        {
            "m2": [[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4]],
            "shape": [3, 4],
            "description": "Larger embedding example",
            "modality": "document",
        },
    ]

    print("🔄 Input embeddings:")
    for i, emb in enumerate(embeddings):
        print(f"   {i + 1}. Shape: {emb['shape']}, Description: {emb['description']}")
    print()

    # Process embeddings
    embedding_data = plugin._regularize_embeddings(embeddings)

    print("📊 Individual processed embeddings:")
    for i, (tensor, shape) in enumerate(zip(embedding_data["embeddings"], embedding_data["shapes"])):
        print(f"   {i + 1}. Shape: {shape}, Tensor shape: {tensor.shape}")
        print(f"       Values: {tensor}")
        print()

    # Show how they're combined in mm_inputs
    mm_inputs = plugin._get_mm_inputs(images=[], videos=[], audios=[], embeddings=embeddings, processor=None)

    print("🎯 Final batched result:")
    embeddings_data = mm_inputs["embeddings"]
    if isinstance(embeddings_data, dict):
        # Handle case where embeddings is a dictionary
        print(f"   Final embeddings structure: {type(embeddings_data)}")
        print(f"   Keys: {list(embeddings_data.keys())}")
        for key, tensor in embeddings_data.items():
            print(f"   '{key}' tensor shape: {tensor.shape}")
            print(f"   '{key}' tensor:\n{tensor}")
    else:
        # Handle case where embeddings is a single tensor
        print(f"   Final tensor shape: {embeddings_data.shape}")
        print(f"   Final tensor:\n{embeddings_data}")
    print(f"   Shape metadata: {mm_inputs['embedding_shapes']}")


def display_integration_example():
    """Display how embeddings integrate with the complete pipeline"""
    print("🔗 INTEGRATION WITH COMPLETE PIPELINE")
    print("=" * 60)

    print("💡 In the complete LLaMA-Factory pipeline:")
    print()

    print("1️⃣ Data Loading:")
    print("   • Dataset contains 'embeddings' field with modality token mapping")
    print("   • Structure: {'m1': ['file1.json'], 'm2': ['file2.json'], ...}")
    print("   • Each JSON file contains 'embedding' array and metadata")
    print("   • Files are loaded in converter.py and processor classes")
    print()

    print("2️⃣ Data Processing:")
    print("   • _regularize_embeddings converts JSON to PyTorch tensors")
    print("   • Shape validation and tensor creation occurs")
    print("   • Multiple embeddings are prepared for batching")
    print("   • Modality tokens (m1, m2, etc.) map to specific embedding files")
    print()

    print("3️⃣ Data Collation:")
    print("   • MultiModalDataCollatorForSeq2Seq handles batching")
    print("   • Embeddings are extracted from features")
    print("   • get_mm_inputs creates final multimodal input dictionary")
    print()

    print("4️⃣ Model Input:")
    print("   • Final dictionary contains:")
    print("     - 'embeddings': torch.Tensor with shape (batch_size, num_tokens, embed_dim)")
    print("     - 'embedding_shapes': List of original shapes")
    print("     - Standard model inputs (input_ids, attention_mask, etc.)")
    print()

    print("5️⃣ Model Forward Pass:")
    print("   • Model receives embeddings alongside text tokens")
    print("   • Embeddings can represent any modality (text, audio, video, sensors)")
    print("   • Model processes embeddings according to its architecture")
    print("   • Modality tokens in text are replaced with corresponding embeddings")
    print()


def main():
    """Main function to run all demonstrations"""
    print("🚀 LLaMA-Factory Embedding Input Display Demo")
    print("=" * 70)
    print()

    # Check if demo files exist - updated to use wm_demo.json
    demo_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wm_demo.json")
    if not os.path.exists(demo_file):
        print(f"❌ Demo file not found: {demo_file}")
        print("Please ensure the embedding demo files are available in the data directory.")
        return

    try:
        # Always show the JSON structure
        display_json_structure_simple()
        
        if TORCH_AVAILABLE:
            print("\n" + "=" * 70 + "\n")
            display_regularization_process()
            print("\n" + "=" * 70 + "\n")

            display_multimodal_inputs()
            print("\n" + "=" * 70 + "\n")

            display_batching_example()
            print("\n" + "=" * 70 + "\n")

            display_integration_example()

            print("✅ SUMMARY:")
            print("=" * 70)
            print("• Embeddings are loaded from JSON files via modality token mapping")
            print("• Structure: {'m1': ['file1.json'], 'm2': ['file2.json'], ...}")
            print("• _regularize_embeddings converts them to PyTorch tensors")
            print("• Final mm_inputs contains 'embeddings' and 'embedding_shapes' keys")
            print("• Multiple embeddings are batched/stacked for efficient processing")
            print("• Ready for integration with any multimodal model architecture")
            print("• Supports arbitrary modalities beyond just image/video/audio")
        else:
            print("\n" + "=" * 70 + "\n")
            print("✅ SIMPLIFIED SUMMARY:")
            print("=" * 70)
            print("• wm_demo.json structure analyzed successfully")
            print("• Embeddings organized by modality tokens (m1, m2, etc.)")
            print("• Each modality maps to a list of embedding JSON files")
            print("• For full processing demo, ensure PyTorch dependencies are available")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
