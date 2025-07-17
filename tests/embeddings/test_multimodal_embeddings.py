#!/usr/bin/env python3
"""
Test script for multi-modal embedding system.
This demonstrates the new flexible embedding key system supporting m1, m2, m3, etc.
"""

import unittest
from unittest.mock import MagicMock

from llamafactory.data.mm_plugin import get_mm_plugin


class TestMultimodalEmbeddings(unittest.TestCase):
    """Test multimodal embeddings with different modality keys."""

    def test_multimodal_embedding_keys(self):
        """Test processing embeddings with different modality keys (m1, m2, m3, etc.)."""
        
        # Create plugin with multimodal embedding tokens
        plugin = get_mm_plugin(
            name="qwen3_embedding",
            image_token="<image>",
            video_token="<video>",
            audio_token="<audio>",
            embedding_tokens={"m1": "<m1>", "m2": "<m2>", "m3": "<m3>"}
        )
        
        # Test embeddings with new format including modality information
        test_embeddings = [
            {
                "file": "dummy_file_1.json",
                "modality_key": "m1"
            },
            {
                "file": "dummy_file_2.json", 
                "modality_key": "m2"
            },
            {
                "file": "dummy_file_3.json",
                "modality_key": "m3"
            }
        ]
        
        # Mock the file loading to return embedding data with "embedding" key
        def mock_open(filename, mode):
            mock_file = MagicMock()
            if "dummy_file_1.json" in filename:
                mock_file.__enter__.return_value.read.return_value = b'{"embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], "shape": [2, 4], "description": "Text embedding", "modality": "text"}'
            elif "dummy_file_2.json" in filename:
                mock_file.__enter__.return_value.read.return_value = b'{"embedding": [[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]], "shape": [2, 4], "description": "Audio embedding", "modality": "audio"}'
            elif "dummy_file_3.json" in filename:
                mock_file.__enter__.return_value.read.return_value = b'{"embedding": [[1.0, 0.9, 0.8, 0.7], [0.6, 0.5, 0.4, 0.3]], "shape": [2, 4], "description": "Video embedding", "modality": "video"}'
            return mock_file
        
        import builtins
        with unittest.mock.patch.object(builtins, 'open', mock_open):
            # Process embeddings
            result = plugin._regularize_embeddings(test_embeddings)
            
            # Verify the result structure
            self.assertIn("embeddings", result)
            self.assertIn("shapes", result)
            self.assertIn("embedding_types", result)
            
            # Verify we have the right number of embeddings
            self.assertEqual(len(result["embeddings"]), 3)
            self.assertEqual(len(result["shapes"]), 3)
            self.assertEqual(len(result["embedding_types"]), 3)
            
            # Verify the embedding types are detected correctly
            expected_types = ["m1", "m2", "m3"]
            self.assertEqual(result["embedding_types"], expected_types)
            
            # Verify shapes are correct
            expected_shapes = [(2, 4), (2, 4), (2, 4)]
            self.assertEqual(result["shapes"], expected_shapes)
            
            # Verify tensor shapes match
            for i, embedding in enumerate(result["embeddings"]):
                self.assertEqual(embedding.shape, expected_shapes[i])
            
            print("✓ Multimodal embedding keys test passed")

    def test_direct_embedding_format(self):
        """Test handling of direct embedding format."""
        
        # Create plugin with multimodal embedding tokens
        plugin = get_mm_plugin(
            name="qwen3_embedding",
            image_token="<image>",
            video_token="<video>",
            audio_token="<audio>",
            embedding_tokens={"m1": "<m1>", "m2": "<m2>"}
        )
        
        # Test with direct embedding format (using "embedding" key)
        test_embeddings = [
            {
                "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                "shape": [2, 4],
                "description": "Text embedding",
                "modality": "text"
            },
            {
                "embedding": [[0.5, 0.6, 0.7, 0.8]],
                "shape": [1, 4],
                "description": "Audio embedding",
                "modality": "audio"
            }
        ]
        
        # Process embeddings
        result = plugin._regularize_embeddings(test_embeddings)
        
        # Verify the result structure
        self.assertIn("embeddings", result)
        self.assertIn("shapes", result)
        self.assertIn("embedding_types", result)
        
        # Verify we have the right number of embeddings
        self.assertEqual(len(result["embeddings"]), 2)
        self.assertEqual(len(result["shapes"]), 2)
        self.assertEqual(len(result["embedding_types"]), 2)
        
        # Verify the embedding types are detected correctly (should be "embedding" for direct format)
        expected_types = ["embedding", "embedding"]
        self.assertEqual(result["embedding_types"], expected_types)
        
        # Verify shapes are correct
        expected_shapes = [(2, 4), (1, 4)]
        self.assertEqual(result["shapes"], expected_shapes)
        
        print("✓ Direct embedding format test passed")

    def test_get_mm_inputs_with_multimodal_embeddings(self):
        """Test _get_mm_inputs with multimodal embeddings."""
        
        # Create plugin with multimodal embedding tokens
        plugin = get_mm_plugin(
            name="qwen3_embedding",
            image_token="<image>",
            video_token="<video>",
            audio_token="<audio>",
            embedding_tokens={"m1": "<m1>", "m2": "<m2>"}
        )
        
        # Test embeddings with new format including modality information
        test_embeddings = [
            {
                "file": "dummy_file_1.json",
                "modality_key": "m1"
            },
            {
                "file": "dummy_file_2.json",
                "modality_key": "m2"
            }
        ]
        
        # Mock the file loading to return embedding data with "embedding" key
        def mock_open(filename, mode):
            mock_file = MagicMock()
            if "dummy_file_1.json" in filename:
                mock_file.__enter__.return_value.read.return_value = b'{"embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], "shape": [2, 4], "description": "Text embedding", "modality": "text"}'
            elif "dummy_file_2.json" in filename:
                mock_file.__enter__.return_value.read.return_value = b'{"embedding": [[0.9, 0.8, 0.7, 0.6]], "shape": [1, 4], "description": "Audio embedding", "modality": "audio"}'
            return mock_file
        
        import builtins
        with unittest.mock.patch.object(builtins, 'open', mock_open):
            # Mock processor
            mock_processor = MagicMock()
            
            # Get multimodal inputs
            mm_inputs = plugin._get_mm_inputs(
                images=[], 
                videos=[], 
                audios=[], 
                embeddings=test_embeddings,
                processor=mock_processor
            )
            
            # Verify the result structure
            self.assertIn("embeddings", mm_inputs)
            self.assertIn("embedding_shapes", mm_inputs)
            self.assertIn("embeddings_by_type", mm_inputs)
            self.assertIn("shapes_by_type", mm_inputs)
            
            # Verify embeddings by type
            self.assertIn("m1", mm_inputs["embeddings_by_type"])
            self.assertIn("m2", mm_inputs["embeddings_by_type"])
            
            # Verify shapes by type
            self.assertIn("m1", mm_inputs["shapes_by_type"])
            self.assertIn("m2", mm_inputs["shapes_by_type"])
            
            print("✓ Get MM inputs with multimodal embeddings test passed")


if __name__ == "__main__":
    unittest.main() 