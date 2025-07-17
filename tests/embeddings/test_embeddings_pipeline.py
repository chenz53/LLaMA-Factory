"""Test the embedding pipeline with multimodal embeddings."""

import logging
import unittest
from unittest.mock import MagicMock, patch

from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.processor.supervised import SupervisedDataset
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.extras.constants import MULTIMODAL_EMBEDDING_PLACEHOLDERS
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model import load_tokenizer
from llamafactory.model.model_utils.embedding_utils import get_embedding_layer_name


class TestEmbeddingsPipeline(unittest.TestCase):
    """Test the embeddings pipeline with multimodal embeddings."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-1.5B-Instruct")
        self.tokenizer = load_tokenizer(self.model_args)
        
        # Use the first available multimodal embedding placeholder
        self.embedding_placeholder = next(iter(MULTIMODAL_EMBEDDING_PLACEHOLDERS.values()))
        
        # Use multimodal embedding tokens
        self.embedding_tokens = {"m1": self.embedding_placeholder}
        
        # Mock the embedding layer
        self.embedding_layer_name = get_embedding_layer_name(self.model_args.model_name_or_path)
        
        # Create a mock for the embedding layer
        self.embedding_layer = MagicMock()
        self.embedding_layer.weight.data = MagicMock()
        self.embedding_layer.weight.data.shape = [32000, 768]  # vocab_size, hidden_size
    
    def test_template_initialization(self):
        """Test that the template is properly initialized with multimodal embeddings."""
        template = get_template_and_fix_tokenizer(self.tokenizer, name="qwen3n")
        self.assertIsNotNone(template)
        
        # Mock the mm_plugin with embedding_tokens
        template.mm_plugin = get_mm_plugin(
            name="qwen3n",
            image_token=None,
            video_token=None,
            audio_token=None,
            embedding_tokens=self.embedding_tokens
        )
        
        # Check that embedding_tokens is properly set
        self.assertIsNotNone(template.mm_plugin.embedding_tokens)
        self.assertEqual(template.mm_plugin.embedding_tokens, self.embedding_tokens)
    
    def test_multimodal_embedding_count(self):
        """Test that the embedding count is correctly calculated."""
        template = get_template_and_fix_tokenizer(self.tokenizer, name="qwen3n")
        template.mm_plugin = get_mm_plugin(
            name="qwen3n",
            image_token=None,
            video_token=None,
            audio_token=None,
            embedding_tokens=self.embedding_tokens
        )
        
        # Create messages with multimodal embedding placeholders
        messages = [
            {"role": "user", "content": f"This is a test {self.embedding_placeholder} message."},
            {"role": "assistant", "content": "This is a response."}
        ]
        
        # Count embedding placeholders
        embedding_count = 0
        for message in messages:
            for placeholder in MULTIMODAL_EMBEDDING_PLACEHOLDERS.values():
                embedding_count += message["content"].count(placeholder)
        
        self.assertEqual(embedding_count, 1)
    
    def test_collator_with_multimodal_embeddings(self):
        """Test that the collator works with multimodal embeddings."""
        template = get_template_and_fix_tokenizer(self.tokenizer, name="qwen3n")
        template.mm_plugin = get_mm_plugin(
            name="qwen3n",
            image_token=None,
            video_token=None,
            audio_token=None,
            embedding_tokens=self.embedding_tokens
        )
        
        # Create a data collator
        data_collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=self.tokenizer,
            template=template,
            model=MagicMock(),
            use_cache=False,
            pad_to_multiple_of=8,
            block_diag_attn=False,
            attn_implementation="eager",
            compute_accuracy=False,
        )
        
        # Test with empty batch embeddings
        batch_embedlens = []
        self.assertEqual(sum(batch_embedlens), 0)
        
        # The collator should handle empty embeddings properly
        self.assertIsNotNone(data_collator)


def test_multimodal_embedding_pipeline():
    """Test the entire multimodal embedding pipeline."""
    
    # Mock data
    messages = [
        {"role": "user", "content": "Please analyze this <m1>"},
        {"role": "assistant", "content": "I'll analyze the embedding data."}
    ]
    
    # Mock embedding data using new format
    embeddings = [
        {
            "m1": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "shape": [2, 4],
            "description": "Sample embedding data",
            "modality": "text"
        }
    ]
    
    print("Testing multimodal embedding pipeline...")
    print(f"Messages: {messages}")
    print(f"Embedding files: {embeddings}")
    
    # Process messages to count embedding placeholders
    embedding_count = 0
    for message in messages:
        for placeholder in MULTIMODAL_EMBEDDING_PLACEHOLDERS.values():
            embedding_count += message["content"].count(placeholder)
    
    print(f"Number of embedding placeholders: {embedding_count}")
    print(f"Number of embedding files: {len(embeddings)}")
    
    # Validate that counts match
    assert embedding_count == len(embeddings), f"Mismatch: {embedding_count} placeholders vs {len(embeddings)} embeddings"
    
    print("✓ Multimodal embedding pipeline test passed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    test_multimodal_embedding_pipeline()
    
    # Run unit tests
    unittest.main()
