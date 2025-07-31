# World Model Training with LLaMA Factory

This directory contains examples and documentation for **World Model** fine-tuning, a novel training approach that performs autoregression in embedding/latent space instead of token space.

## 🔬 What is World Model Training?

Traditional language models perform autoregression at the token level, predicting the next token given previous tokens. World Model training instead:

1. **Embeds multimodal inputs** (text, images, audio, etc.) into a shared latent space
2. **Performs autoregression on embeddings** - predicting the next embedding vector given previous embeddings  
3. **Uses reconstruction loss** (L1 or L2) between predicted and target embeddings
4. **Enables multimodal understanding** through shared embedding representations

## 🚀 Quick Start

### Basic Usage

```bash
# Run the demo with pre-configured settings
bash examples/train_wm/world_model_demo.sh
```

### Python Script

```bash
# Or run the Python demo script
python examples/train_wm/world_model_demo.py
```

### Command Line

```bash
# Custom world model training
python -m llamafactory.train.tuner \
    --stage wm \
    --model_name meta-llama/Llama-3.2-1B \
    --dataset wm_demo \
    --embedding_loss_weight 1.0 \
    --world_model_loss_type l2 \
    --embedding_dim 4 \
    --output_dir ./saves/my_world_model
```

## 📊 Data Format

World model training expects data in ShareGPT format with an additional `embeddings` field. The `wm_demo.json` dataset demonstrates temporal progression where the model learns to predict how patient states evolve over time:

```json
{
  "messages": [
    {
      "content": "<m1>Patient at day 1.",
      "role": "user"
    },
    {
      "content": "<m2>Patient at day 2.",
      "role": "assistant"
    }
  ],
  "embeddings": {
    "m1": ["embedding_demo_data/sample_embedding_1.json"],
    "m2": ["embedding_demo_data/sample_embedding_2.json"]
  }
}
```

The model learns that `m1` (day 1 state) should predict `m2` (day 2 state), enabling temporal progression modeling.

Each embedding file contains:

```json
{
  "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
  "shape": [2, 4],
  "description": "Sample embedding",
  "modality": "text"
}
```

## ⚙️ Configuration Parameters

### World Model Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage` | `"sft"` | Set to `"wm"` for world model training |
| `embedding_loss_weight` | `1.0` | Weight for embedding reconstruction loss |
| `lm_loss_weight` | `0.1` | Weight for language modeling loss |
| `embedding_dim` | `4` | Dimension of target embeddings |
| `world_model_loss_type` | `"l2"` | Reconstruction loss type (`"l1"` or `"l2"`) |

### Training Parameters

All standard LLaMA Factory training parameters are supported:

```bash
--learning_rate 1e-4
--num_train_epochs 3.0
--per_device_train_batch_size 4
--finetuning_type lora
--lora_rank 8
```

## 🏗️ Architecture

The world model trainer adds an **Embedding Prediction Head** to the base language model:

```
Input Text → LLM → Hidden States → Prediction Head → Predicted Embeddings
                                                           ↓
Target Embeddings ←────────────── Reconstruction Loss ←────┘
```

### Loss Function

The total loss combines:
1. **Embedding Reconstruction Loss**: L1/L2 loss between predicted and target embeddings
2. **Language Modeling Loss**: Standard next-token prediction loss (optional)

```python
total_loss = embedding_weight * reconstruction_loss + lm_weight * lm_loss
```

## 📁 Files Overview

- `world_model_demo.sh` - Bash script for quick demo
- `world_model_demo.py` - Python demo script  
- `README.md` - This documentation
- `../data/wm_demo.json` - Example dataset with temporal progression (Patient day 1 → day 2)
- `../data/embedding_demo.json` - Alternative dataset for general embedding training

## 🔧 Implementation Details

### Key Components

1. **WorldModelTrainer** (`src/llamafactory/train/wm/trainer.py`)
   - Custom trainer with embedding prediction capabilities
   - Handles L1/L2 reconstruction loss computation
   - Manages embedding-text alignment
   - Enhanced debugging and validation

2. **WorldModelDataCollator** (`src/llamafactory/train/wm/collator.py`)
   - Processes embedding dictionary format ({"m1": [...], "m2": [...]})
   - Creates temporal progression targets (m1 → m2, m2 → m3, etc.)
   - Handles variable-length embedding sequences
   - Robust error handling for missing/malformed files

3. **World Model Workflow** (`src/llamafactory/train/wm/workflow.py`)
   - Orchestrates the training process
   - Integrates with existing LLaMA Factory infrastructure

### Embedding Processing

The data collator automatically:
- Loads embeddings from JSON files referenced in the dataset
- Creates temporal progression targets (input: m1, target: m2)
- Handles padding and batching across different embedding lengths
- Supports multiple modalities in the same batch
- Validates embedding dimensions and provides helpful error messages

## 🎯 Use Cases

### Temporal Progression Modeling
Train models to predict how states evolve over time (e.g., patient condition progression, system state transitions).

### Multimodal Understanding
Train models to understand relationships between different modalities (text, vision, audio) in a shared embedding space.

### Cross-Modal Generation  
Enable models to generate embeddings for one modality based on inputs from another.

### Medical AI Applications
Model patient state progression, treatment response prediction, and disease trajectory forecasting.

### Representation Learning
Learn rich, semantic representations that capture world model dynamics and temporal dependencies.

## 🐛 Troubleshooting

### Common Issues

**Empty embeddings in batch**: Make sure embedding files exist and contain valid data.

**Dimension mismatch**: Ensure `embedding_dim` matches your target embedding dimensions.

**Memory issues**: Reduce batch size or use gradient accumulation for large embedding sequences.

### Debug Mode

Enable debug logging to see detailed embedding processing:

```bash
export TRANSFORMERS_VERBOSITY=debug
```

## 🔗 Related Work

World model training is inspired by:
- World Models for Reinforcement Learning
- Multimodal Foundation Models  
- Embedding-based Language Modeling
- Cross-modal Representation Learning

---

For more information, see the main [LLaMA Factory documentation](../../README.md).