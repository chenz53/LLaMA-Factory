#!/usr/bin/env python3
"""
Test script to verify the multimodal embedding fix works correctly.
This script tests that embeddings with different shapes are properly organized by modality.
"""

import json
import os
import sys
import tempfile

import torch


# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.extras.constants import MULTIMODAL_EMBEDDING_PLACEHOLDERS


def create_test_embedding_files():
    """Create temporary embedding files with different modalities and shapes."""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Embedding file 1: m1 modality with shape [2, 8]
    embedding1_data = {
        "m1": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        "shape": [2, 8],
        "description": "Text embedding",
        "modality": "text",
    }

    # Embedding file 2: m2 modality with shape [3, 4]
    embedding2_data = {
        "m2": [[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4]],
        "shape": [3, 4],
        "description": "Audio embedding",
        "modality": "audio",
    }

    # Embedding file 3: Another m1 modality with shape [1, 8]
    embedding3_data = {
        "m1": [[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]],
        "shape": [1, 8],
        "description": "Another text embedding",
        "modality": "text",
    }

    # Save to files
    file1 = os.path.join(temp_dir, "embedding1.json")
    file2 = os.path.join(temp_dir, "embedding2.json")
    file3 = os.path.join(temp_dir, "embedding3.json")

    with open(file1, "w") as f:
        json.dump(embedding1_data, f)
    with open(file2, "w") as f:
        json.dump(embedding2_data, f)
    with open(file3, "w") as f:
        json.dump(embedding3_data, f)

    return [
        {"file": file1, "modality_key": "m1"},
        {"file": file3, "modality_key": "m1"},
        {"file": file2, "modality_key": "m2"},
    ], temp_dir


def test_message_processing(plugin, embeddings):
    """Test how messages are processed with multimodal embedding placeholders."""

    print("💬 Testing Message Processing with Qwen3EmbeddingPlugin")
    print("=" * 50)

    # Get available placeholders
    placeholders = list(MULTIMODAL_EMBEDDING_PLACEHOLDERS.values())
    print(f"📝 Available placeholders: {placeholders}")
    print()

    # Create test messages with multimodal embedding placeholders
    test_messages = [
        {
            "role": "user",
            "content": f"Please analyze this {placeholders[0]} {placeholders[0]} and this {placeholders[1]} data.",
        },
        {"role": "assistant", "content": "I'll analyze the multimodal embedding data for you."},
    ]

    print("📨 Original messages:")
    for i, msg in enumerate(test_messages):
        print(f"   Message {i + 1} ({msg['role']}):")
        print(f"      {msg['content']}")
    print()

    # Process messages
    print("⚙️ Processing messages...")
    processed_messages = plugin.process_messages(
        messages=test_messages, images=[], videos=[], audios=[], embeddings=embeddings, processor=None
    )

    print("✅ Message processing completed!")
    print()

    print("📨 Processed messages:")
    for i, msg in enumerate(processed_messages):
        print(f"   Message {i + 1} ({msg['role']}):")
        print(f"      {msg['content']}")
        print()

    # Analyze the changes
    print("🔍 Analysis of Changes:")
    original_content = test_messages[0]["content"]
    processed_content = processed_messages[0]["content"]

    print(f"   Original:  {original_content}")
    print(f"   Processed: {processed_content}")
    print()

    # Check for expected patterns
    expected_patterns = ["<|m1_start|>", "<|m1_end|>", "<|m2_start|>", "<|m2_end|>"]
    found_patterns = []

    for pattern in expected_patterns:
        if pattern in processed_content:
            found_patterns.append(pattern)

    if found_patterns:
        print(f"✅ SUCCESS: Found expected patterns: {found_patterns}")
    else:
        print("❌ No expected embedding patterns found in processed content")

    # Count embedding tokens
    embedding_token_count = 0
    for embedding_key, embedding_token in plugin.embedding_tokens.items():
        count = processed_content.count(embedding_token)
        embedding_token_count += count
        if count > 0:
            print(f"   Found {count} instances of {embedding_key} token: '{embedding_token}'")

    print(f"   Total embedding tokens: {embedding_token_count}")
    print()


def test_multimodal_embedding_fix():
    """Test that multimodal embeddings are properly organized by modality."""

    print("🧪 Testing Multimodal Embedding Fix")
    print("=" * 50)

    # Create test embedding files
    embeddings, temp_dir = create_test_embedding_files()

    try:
        # Create plugin
        plugin = get_mm_plugin(name="qwen3_embedding", embedding_tokens={"m1": "<m1_pad>", "m2": "<m2_pad>"})

        print("📁 Created test embedding files:")
        for i, emb in enumerate(embeddings):
            print(f"   File {i + 1}: {emb['modality_key']} modality")
        print()

        # Process embeddings
        print("⚙️ Processing embeddings...")
        mm_inputs = plugin._get_mm_inputs(images=[], videos=[], audios=[], embeddings=embeddings, processor=None)

        print("✅ Processing completed!")
        print()

        # Check the results
        print("📊 Results Analysis:")
        print(f"   Keys in mm_inputs: {list(mm_inputs.keys())}")
        print(f"   mm_inputs: {mm_inputs}")
        print()

        embeddings_result = mm_inputs["embeddings"]
        print(f"🎯 Embeddings format: {type(embeddings_result)}")

        if isinstance(embeddings_result, dict):
            print("✅ SUCCESS: Embeddings are properly formatted as dictionary!")
            print()

            for modality, tensor in embeddings_result.items():
                print(f"   {modality}:")
                print(f"      Type: {type(tensor)}")
                print(f"      Shape: {tensor.shape}")
                print(f"      Dtype: {tensor.dtype}")
                print(f"      Sample values: {tensor.flatten()[:5].tolist()}")
                print()

            # Verify the expected structure
            expected_modalities = {"m1", "m2"}
            actual_modalities = set(embeddings_result.keys())

            if expected_modalities == actual_modalities:
                print("✅ SUCCESS: All expected modalities present!")

                # Check m1 has 2 embeddings stacked (shapes [2,8] and [1,8])
                m1_tensor = embeddings_result["m1"]
                if m1_tensor.shape[0] == 3:  # 2 embeddings stacked
                    print("✅ SUCCESS: m1 modality correctly stacked 2 embeddings!")
                else:
                    print(f"❌ FAILURE: m1 expected 2 stacked embeddings, got shape {m1_tensor.shape}")

                # Check m2 has 1 embedding (shape [3,4])
                m2_tensor = embeddings_result["m2"]
                if m2_tensor.shape == torch.Size([3, 4]):
                    print("✅ SUCCESS: m2 modality has correct shape [3, 4]!")
                else:
                    print(f"❌ FAILURE: m2 expected shape [3, 4], got {m2_tensor.shape}")

            else:
                print(f"❌ FAILURE: Expected modalities {expected_modalities}, got {actual_modalities}")
        else:
            print(f"❌ FAILURE: Embeddings should be dict, got {type(embeddings_result)}")
            print(f"   Value: {embeddings_result}")

        print()
        print("🔍 Additional Metadata:")
        print(f"   embedding_shapes: {mm_inputs.get('embedding_shapes', [])}")
        print(f"   embedding_types: {mm_inputs.get('embedding_types', [])}")

        # Test message processing
        print()
        print("=" * 50)
        test_message_processing(plugin, embeddings)

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)
        print()
        print("🧹 Cleaned up temporary files")


if __name__ == "__main__":
    test_multimodal_embedding_fix()
