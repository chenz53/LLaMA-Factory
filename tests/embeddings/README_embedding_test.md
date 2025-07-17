# Embedding Input Display Test Script

This test script demonstrates how embeddings are processed in LLaMA-Factory before being fed into the model.

## Overview

The script `test_embedding_display.py` shows the complete embedding processing pipeline from JSON files to final model inputs, using the existing `data/embedding_demo.json` demo file.

## What the Script Shows

### 1. JSON File Structure
- Shows how embeddings are stored in JSON format
- Displays the structure of `data/embedding_demo.json` 
- Shows individual embedding files with metadata

### 2. Regularization Process
- Demonstrates how `_regularize_embeddings` converts JSON to PyTorch tensors
- Shows tensor properties (shape, dtype, device, statistics)
- Displays the transformation from raw JSON to processed tensors

### 3. Final Multimodal Inputs
- Shows the `mm_inputs` dictionary structure
- Displays the final tensors that go to the model
- Shows how embeddings are combined with other multimodal inputs

### 4. Batching Example
- Demonstrates how multiple embeddings are batched together
- Shows how embeddings of different sizes are handled
- Displays the final stacked tensor structure

### 5. Integration Overview
- Explains how embeddings fit into the complete LLaMA-Factory pipeline
- Shows the data flow from loading to model input
- Explains the multimodal architecture integration

## Usage

```bash
# Run from the LLaMA-Factory root directory
python test_embedding_display.py
```

## Key Features Demonstrated

✅ **JSON to Tensor Conversion**: Shows how embedding JSON files are loaded and converted to PyTorch tensors

✅ **Shape Validation**: Demonstrates how embedding shapes are validated and preserved

✅ **Multimodal Integration**: Shows how embeddings integrate with the existing multimodal pipeline

✅ **Batching Support**: Demonstrates how multiple embeddings are efficiently batched

✅ **Arbitrary Modality Support**: Shows how embeddings can represent any modality (text, audio, video, sensors, etc.)

## Expected Output

The script will display:
- JSON file structure and content
- Step-by-step processing of embeddings
- Final tensor shapes and values
- Integration points with the model pipeline
- Summary of the embedding processing workflow

## Multi-Modal Embedding Keys

The framework now supports multiple embedding types using keys like `m1`, `m2`, `m3`, etc. instead of the universal `embedding` key. This allows for better organization and distinction between different modalities:

- `m1` could represent text embeddings
- `m2` could represent audio embeddings  
- `m3` could represent video embeddings
- And so on...

The system automatically detects which embedding key is used and processes them accordingly, maintaining backward compatibility with the legacy `embedding` key format.

## Example Embedding Format

```json
{
  "m1": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  ],
  "shape": [2, 8],
  "description": "Sample text embedding",
  "modality": "text",
  "num_tokens": 2,
  "embedding_dim": 8
}
```

## Final Model Input

The embeddings are processed into a format ready for the model:

```python
mm_inputs = {
    "embeddings": torch.Tensor,  # Shape: (num_tokens, embedding_dim)
    "embedding_shapes": [(2, 8)],  # Original shape metadata
    # ... other multimodal inputs
}
```

This structure allows the model to receive embeddings alongside text tokens and process them according to the model's architecture. 