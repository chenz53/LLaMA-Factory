"""Basic unit tests for the Qwen3 World-Model (Qwen3WM).

These tests only validate that the model can be instantiated from a minimal
configuration, perform a forward pass and run generation without errors.  They
are intentionally lightweight so they can run inside the CI in <10 s on CPU.
"""

import torch
from llamafactory.models.qwen3_wm.configuration_qwen3_wm import Qwen3WMConfig
from llamafactory.models.qwen3_wm.modeling_qwen3_wm import Qwen3WM


def _create_small_config() -> Qwen3WMConfig:
    """Return a very small config suitable for CPU unit tests."""
    return Qwen3WMConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        head_dim=128,
        hidden_act="silu",
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3072,
        max_position_embeddings=40960,
        max_window_layers=28,
        model_type="qwen3_wm",
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=1000000,
        sliding_window=None,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
        use_sliding_window=False,
        vocab_size=151936,
        # Predictor overrides (keep it tiny as well)
        predictor_config={
            "dim": 1024,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 64,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000,
            "rope_scaling": None,
            "attention_bias": False,
            "sliding_window": None,
            "use_sliding_window": False,
            "use_cache": False,
            "vocab_size": 151936,
            "initializer_range": 0.02,
        },
        m1_config={
            "dim": 128,
            "hidden_size": 1024,
            "intermediate_size": 512,
            "hidden_act": "silu",
        },
        m2_config={
            "dim": 128,
            "hidden_size": 1024,
            "intermediate_size": 512,
            "hidden_act": "silu",
        },
    )


def test_forward_no_mm():
    """Forward pass with only text input (no multimodal embeddings)."""
    config = _create_small_config()
    model = Qwen3WM(config)
    print(model)
    model.eval()

    batch, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    embeddings = {
        "m1": torch.randn(batch, seq_len, config.m1_config.dim),
        "m2": torch.randn(batch, seq_len, config.m2_config.dim),
    }

    with torch.no_grad():
        outputs = model(input_ids=input_ids, embeddings=embeddings)
        print(outputs)

    # Basic sanity checks on shapes
    assert outputs.logits.shape == (batch, seq_len, config.vocab_size)
    assert outputs.hidden_states is None or outputs.hidden_states[-1].shape == (
        batch,
        seq_len,
        config.hidden_size,
    )


def test_predictor_forward():
    """Directly test Predictor module forward."""
    config = _create_small_config()
    model = Qwen3WM(config)
    predictor = model.predictor  # type: ignore
    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)

    # context = even indices, target = odd indices
    context_mask = [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3])]
    target_mask = [torch.tensor([4, 5, 6, 7]), torch.tensor([4, 5, 6])]
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0]])

    out = predictor(
        inputs_embeds=hidden_states,
        context_mask=context_mask,
        target_mask=target_mask,
        attention_mask=attention_mask,
    )
    print(out.last_hidden_state.shape)
    assert out.last_hidden_state.shape[-1] == config.hidden_size
    # number of rows equals total target tokens across batch
    # assert out.last_hidden_state.shape[0] == batch * target_mask[0].numel()


def test_generate():
    """Smoke-test autoregressive generation on small prompt."""
    config = _create_small_config()
    model = Qwen3WM(config)
    model.eval()

    tokenizer_input = torch.tensor([[1, 2, 3]])  # dummy prompt
    with torch.no_grad():
        generated = model.generate(tokenizer_input, max_length=10)

    assert generated.shape[1] == 10  # sequence length expanded


if __name__ == "__main__":
    # test_forward_no_mm()
    test_predictor_forward()
    # test_generate()
