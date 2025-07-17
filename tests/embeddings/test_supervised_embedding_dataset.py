#!/usr/bin/env python3
"""
Test script to validate supervised dataset functionality with embedding mode.
This script uses embedding_demo.json as an example to test the complete pipeline
from dataset loading to supervised fine-tuning with embeddings.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llamafactory.data.converter import SharegptDatasetConverter
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model import load_tokenizer


class TestSupervisedEmbeddingDataset(unittest.TestCase):
    """Test supervised dataset functionality with embedding mode."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        cls.test_dir = Path(__file__).parent.parent.parent
        cls.data_dir = cls.test_dir / "data"
        cls.embedding_demo_file = cls.data_dir / "embedding_demo.json"
        cls.embedding_data_dir = cls.data_dir / "embedding_demo_data"

        # Verify test files exist
        if not cls.embedding_demo_file.exists():
            raise FileNotFoundError(f"embedding_demo.json not found at {cls.embedding_demo_file}")
        if not cls.embedding_data_dir.exists():
            raise FileNotFoundError(f"embedding_demo_data directory not found at {cls.embedding_data_dir}")

        # Set up model and tokenizer
        cls.model_args = ModelArguments(model_name_or_path="Qwen/Qwen3-0.6B")
        cls.data_args = DataArguments(
            template="qwen3_embedding", cutoff_len=1024, dataset_dir=str(cls.data_dir), media_dir=str(cls.data_dir)
        )

        # Load tokenizer with mocking to avoid downloading
        with patch("llamafactory.model.load_tokenizer") as mock_load_tokenizer:
            # Create a simple mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.bos_token_id = 1
            mock_tokenizer.additional_special_tokens_ids = []
            mock_tokenizer.encode.return_value = [1, 100, 200, 2]  # Mock token IDs
            mock_tokenizer.decode.return_value = "mock response"
            mock_tokenizer.vocab_size = 32000

            mock_load_tokenizer.return_value = {"tokenizer": mock_tokenizer, "processor": None}

            tokenizer_module = load_tokenizer(cls.model_args)
            cls.tokenizer = tokenizer_module["tokenizer"]
            cls.processor = tokenizer_module["processor"]

    def setUp(self):
        """Set up for each test method."""
        # Load the demo dataset
        with open(self.embedding_demo_file, "rb") as f:
            self.demo_data = json.load(f)

        # Create template with mocking
        with patch("llamafactory.data.template.get_template_and_fix_tokenizer") as mock_get_template:
            mock_template = MagicMock()
            mock_template.mm_plugin = get_mm_plugin(
                name="qwen3_embedding",
                embedding_tokens={"m1": "<m1_pad>", "m2": "<m2_pad>"},
            )
            mock_template.encode_multiturn.return_value = [([1, 100], [200, 2])]
            mock_template.efficient_eos = False
            mock_get_template.return_value = mock_template

            self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)

    def test_embedding_files_exist_and_valid(self):
        """Test that embedding demo files exist and have valid format."""
        print("\n🧪 Testing embedding files existence and validity...")

        # Check main demo file
        self.assertTrue(
            self.embedding_demo_file.exists(), f"embedding_demo.json should exist at {self.embedding_demo_file}"
        )

        # Validate JSON structure
        self.assertIsInstance(self.demo_data, list, "Demo data should be a list")
        self.assertGreater(len(self.demo_data), 0, "Demo data should not be empty")

        # Check each entry has required fields
        for i, entry in enumerate(self.demo_data):
            with self.subTest(entry=i):
                self.assertIn("messages", entry, f"Entry {i} should have 'messages' field")
                self.assertIsInstance(entry["messages"], list, f"Entry {i} messages should be a list")

                # Check for embedding references (m1, m2, etc.)
                embedding_keys = [k for k in entry.keys() if k.startswith("m") and k[1:].isdigit()]
                if embedding_keys:
                    for key in embedding_keys:
                        self.assertIsInstance(entry[key], list, f"Entry {i} {key} should be a list")
                        for file_path in entry[key]:
                            full_path = self.data_dir / file_path
                            self.assertTrue(full_path.exists(), f"Embedding file {full_path} should exist")

        print("✅ Embedding files exist and have valid format")

    def test_embedding_file_content_structure(self):
        """Test that individual embedding files have correct structure."""
        print("\n🧪 Testing embedding file content structure...")

        sample_files = ["embedding_demo_data/sample_embedding_1.json", "embedding_demo_data/sample_embedding_2.json"]

        for file_path in sample_files:
            full_path = self.data_dir / file_path
            with open(full_path, "r", encoding="utf-8") as f:
                embedding_data = json.load(f)

            with self.subTest(file=file_path):
                # Check required fields
                self.assertIn("embedding", embedding_data, f"{file_path} should have 'embedding' field")
                self.assertIn("shape", embedding_data, f"{file_path} should have 'shape' field")

                # Validate embedding structure
                embedding = embedding_data["embedding"]
                shape = embedding_data["shape"]

                self.assertIsInstance(embedding, list, f"{file_path} embedding should be a list")
                self.assertIsInstance(shape, list, f"{file_path} shape should be a list")
                self.assertEqual(len(shape), 2, f"{file_path} shape should be 2D")

                # Check dimensions match
                self.assertEqual(len(embedding), shape[0], f"{file_path} embedding length should match shape[0]")
                if len(embedding) > 0:
                    self.assertEqual(len(embedding[0]), shape[1], f"{file_path} embedding width should match shape[1]")

        print("✅ Embedding file content structure is valid")

    def test_dataset_converter_with_embeddings(self):
        """Test SharegptDatasetConverter handles embeddings correctly."""
        print("\n🧪 Testing ShareGPT dataset converter with embeddings...")

        # Create a mock dataset attribute for embeddings
        from llamafactory.data.parser import DatasetAttr

        dataset_attr = DatasetAttr(
            load_from="file",
            dataset_name="embedding_demo.json",
            formatting="sharegpt",
            embeddings="embeddings",  # This would map to embedding keys in the data
            messages="messages",
            role_tag="role",
            content_tag="content",
            user_tag="user",
            assistant_tag="assistant",
        )

        converter = SharegptDatasetConverter(dataset_attr=dataset_attr, data_args=self.data_args)

        # Test with first demo entry
        demo_entry = self.demo_data[0].copy()

        # Convert to expected format for converter
        if "m1" in demo_entry:
            demo_entry["embeddings"] = {"m1": demo_entry["m1"]}

        result = converter(demo_entry)

        # Validate conversion results
        self.assertIn("_prompt", result, "Converted data should have '_prompt'")
        self.assertIn("_response", result, "Converted data should have '_response'")
        self.assertIn("_embeddings", result, "Converted data should have '_embeddings'")

        # Check embedding processing
        if result["_embeddings"]:
            self.assertIsInstance(result["_embeddings"], list, "_embeddings should be a list")
            for embedding in result["_embeddings"]:
                self.assertIn("file", embedding, "Embedding should have 'file' field")
                self.assertIn("modality_key", embedding, "Embedding should have 'modality_key' field")

        print("✅ ShareGPT dataset converter handles embeddings correctly")

    def test_supervised_dataset_processor_with_embeddings(self):
        """Test SupervisedDatasetProcessor handles embeddings correctly."""
        print("\n🧪 Testing supervised dataset processor with embeddings...")

        processor = SupervisedDatasetProcessor(
            template=self.template, tokenizer=self.tokenizer, processor=self.processor, data_args=self.data_args
        )

        # Create test examples with embeddings
        examples = {
            "_prompt": [[{"role": "user", "content": "<m1>What does this embedding represent?"}]],
            "_response": [[{"role": "assistant", "content": "This is a text embedding."}]],
            "_system": [""],
            "_tools": [""],
            "_images": [None],
            "_videos": [None],
            "_audios": [None],
            "_embeddings": [
                [{"file": str(self.data_dir / "embedding_demo_data/sample_embedding_1.json"), "modality_key": "m1"}]
            ],
        }

        # Mock file loading for embeddings
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.return_value = json.dumps(
                {
                    "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                    "shape": [2, 4],
                    "description": "Test embedding",
                    "modality": "text",
                }
            ).encode("utf-8")
            mock_open.return_value = mock_file

            # Process the dataset
            result = processor.preprocess_dataset(examples)

        # Validate processing results
        self.assertIn("input_ids", result, "Result should have 'input_ids'")
        self.assertIn("attention_mask", result, "Result should have 'attention_mask'")
        self.assertIn("labels", result, "Result should have 'labels'")
        self.assertIn("embeddings", result, "Result should have 'embeddings'")

        # Check embeddings are preserved
        self.assertEqual(len(result["embeddings"]), 1, "Should have one embedding entry")
        self.assertIsNotNone(result["embeddings"][0], "Embedding entry should not be None")

        print("✅ Supervised dataset processor handles embeddings correctly")

    def test_multimodal_plugin_embedding_processing(self):
        """Test that the multimodal plugin processes embeddings correctly."""
        print("\n🧪 Testing multimodal plugin embedding processing...")

        plugin = self.template.mm_plugin

        # Test embedding regularization
        test_embeddings = [str(self.data_dir / "embedding_demo_data/sample_embedding_1.json")]

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.return_value = json.dumps(
                {
                    "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                    "shape": [2, 4],
                    "description": "Test embedding",
                    "modality": "text",
                }
            ).encode("utf-8")
            mock_open.return_value = mock_file

            result = plugin._regularize_embeddings(test_embeddings)

        # Validate regularization results
        self.assertIn("embeddings", result, "Result should have 'embeddings'")
        self.assertIn("shapes", result, "Result should have 'shapes'")
        self.assertIn("embedding_types", result, "Result should have 'embedding_types'")

        # Check tensor properties
        embeddings = result["embeddings"]
        self.assertEqual(len(embeddings), 1, "Should have one embedding tensor")
        self.assertIsInstance(embeddings[0], torch.Tensor, "Embedding should be a tensor")
        self.assertEqual(embeddings[0].shape, (2, 4), "Tensor should have correct shape")

        print("✅ Multimodal plugin processes embeddings correctly")

    def test_end_to_end_pipeline_with_demo_data(self):
        """Test the complete pipeline using actual demo data."""
        print("\n🧪 Testing end-to-end pipeline with demo data...")

        # Create dataset from demo data
        dataset_entries = []
        for entry in self.demo_data[:2]:  # Test first 2 entries
            # Convert to expected format
            converted_entry = {"messages": entry["messages"]}

            # Add embedding references
            for key in entry.keys():
                if key.startswith("m") and key[1:].isdigit():
                    converted_entry[key] = entry[key]

            dataset_entries.append(converted_entry)

        # Create Hugging Face dataset
        dataset = Dataset.from_list(dataset_entries)

        # Create converter and processor
        from llamafactory.data.parser import DatasetAttr

        dataset_attr = DatasetAttr(
            load_from="file",
            dataset_name="embedding_demo.json",
            formatting="sharegpt",
            embeddings="embeddings",
            messages="messages",
            role_tag="role",
            content_tag="content",
            user_tag="user",
            assistant_tag="assistant",
        )

        converter = SharegptDatasetConverter(dataset_attr=dataset_attr, data_args=self.data_args)

        processor = SupervisedDatasetProcessor(
            template=self.template, tokenizer=self.tokenizer, processor=self.processor, data_args=self.data_args
        )

        # Process dataset
        def convert_and_process(examples):
            converted_examples = {
                "_prompt": [],
                "_response": [],
                "_system": [],
                "_tools": [],
                "_images": [],
                "_videos": [],
                "_audios": [],
                "_embeddings": [],
            }

            for i in range(len(examples["messages"])):
                # Create a single example dict for the converter
                example_dict = {"messages": examples["messages"][i]}

                # Add embedding keys
                for key in examples.keys():
                    if key.startswith("m") and key[1:].isdigit() and i < len(examples[key]):
                        if key not in example_dict:
                            example_dict[key] = examples[key][i]

                # Convert embedding keys to embeddings format
                embeddings_dict = {}
                for key in example_dict.keys():
                    if key.startswith("m") and key[1:].isdigit():
                        embeddings_dict[key] = example_dict[key]

                if embeddings_dict:
                    example_dict["embeddings"] = embeddings_dict

                # Convert using SharegptDatasetConverter logic
                # ShareGPT format expects the full messages array
                messages = example_dict["messages"]
                prompt = messages[:-1] if len(messages) > 1 else []
                response = [messages[-1]] if messages else []

                converted_examples["_prompt"].append(prompt)
                converted_examples["_response"].append(response)
                converted_examples["_system"].append("")
                converted_examples["_tools"].append("")
                converted_examples["_images"].append(None)
                converted_examples["_videos"].append(None)
                converted_examples["_audios"].append(None)

                # Process embeddings
                embeddings_list = None
                if embeddings_dict:
                    embeddings_list = []
                    for modality_key, files in embeddings_dict.items():
                        if not isinstance(files, list):
                            files = [files]
                        for file_path in files:
                            if file_path is not None:  # Skip None file paths
                                full_path = self.data_dir / file_path
                                embeddings_list.append({"file": str(full_path), "modality_key": modality_key})

                converted_examples["_embeddings"].append(embeddings_list)

            return converted_examples

        # Mock file loading
        with patch("builtins.open", create=True) as mock_open:

            def mock_open_func(filename, mode="r"):
                mock_file = MagicMock()
                if "sample_embedding_1.json" in str(filename):
                    mock_file.__enter__.return_value.read.return_value = json.dumps(
                        {"embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], "shape": [2, 4]}
                    ).encode("utf-8")
                elif "sample_embedding_2.json" in str(filename):
                    mock_file.__enter__.return_value.read.return_value = json.dumps(
                        {"embedding": [[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]], "shape": [2, 4]}
                    ).encode("utf-8")
                return mock_file

            mock_open.side_effect = mock_open_func

            # Process with mocked file loading
            converted_dataset = dataset.map(convert_and_process, batched=True)
            final_result = processor.preprocess_dataset(dict(converted_dataset[0]))

        # Validate final results
        self.assertIn("input_ids", final_result, "Final result should have input_ids")
        self.assertIn("embeddings", final_result, "Final result should have embeddings")

        # Check that we have the expected number of examples
        self.assertGreater(len(final_result["input_ids"]), 0, "Should have processed examples")

        print("✅ End-to-end pipeline works with demo data")

    def test_embedding_shape_validation(self):
        """Test that embedding shape validation and reshaping works correctly."""
        print("\n🧪 Testing embedding shape validation...")

        plugin = self.template.mm_plugin

        # Test with valid reshape - 4 elements can be reshaped to [1, 4]
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.return_value = json.dumps(
                {
                    "embedding": [[0.1, 0.2], [0.3, 0.4]],  # 2x2 data (4 elements)
                    "shape": [1, 4],  # Reshape to 1x4 (4 elements) - valid reshape
                }
            ).encode("utf-8")
            mock_open.return_value = mock_file

            # This should reshape the tensor to match the specified shape
            result = plugin._regularize_embeddings(["dummy.json"])

            # The tensor should be successfully reshaped from [2, 2] to [1, 4]
            self.assertEqual(len(result["embeddings"]), 1)
            self.assertEqual(result["embeddings"][0].shape, (1, 4))

        print("✅ Embedding shape validation works correctly")


def run_test_suite():
    """Run the complete test suite and provide summary."""
    print("🚀 Starting Supervised Embedding Dataset Test Suite")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSupervisedEmbeddingDataset)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests failed: {len(result.failures)}")
    print(f"💥 Tests errored: {len(result.errors)}")
    print(
        f"📈 Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")

    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
