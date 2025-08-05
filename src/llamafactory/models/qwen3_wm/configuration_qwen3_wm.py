"""Qwen3-World-Model (Qwen3-WM) configuration

This file composes the official Hugging Face `Qwen3Config` with extra sub-configs
for the multimodal **connectors** and the **predictor** module used by the world
model.  The resulting `Qwen3WMConfig` can therefore be loaded with
`AutoConfig.from_pretrained(...)` like any other Transformers model while still
exposing the additional fields required by the WM architecture.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3WMConnectorConfig(PretrainedConfig):
    """Lightweight config object for a single MLP connector in a multimodal branch."""

    model_type = "qwen3_wm_connector"

    def __init__(
        self,
        *,
        dim: int = 1024,
        hidden_size: int = 384,
        intermediate_size: int = 1024,
        hidden_act: str = "silu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act


class Qwen3WMPredictorConfig(Qwen3Config):
    """Config for the lightweight predictor module (independent of the base LM)."""

    model_type = "qwen3_wm_predictor"

    def __init__(
        self,
        *,
        dim: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim


class Qwen3WMConfig(Qwen3Config):
    """Qwen3-WM = Qwen3 base language model + multimodal connectors + predictor."""

    model_type = "qwen3_wm"

    # mapping key → sub-config class – used by save / load utilities if desired
    sub_configs: Dict[str, type[PretrainedConfig]] = {
        "m1_config": Qwen3WMConnectorConfig,
        "m2_config": Qwen3WMConnectorConfig,
        "predictor_config": Qwen3WMPredictorConfig,
    }

    def __init__(
        self,
        *,
        # World-model specific sub-configs
        m1_config: Optional[Dict[str, Any]] = None,
        m2_config: Optional[Dict[str, Any]] = None,
        predictor_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialise the base LM fields with remaining kwargs
        super().__init__(**kwargs)

        # ---------------------------------------------------------------------
        # Connector sub-configs (two branches by default, m1 & m2)
        # ---------------------------------------------------------------------
        if isinstance(m1_config, dict):
            self.m1_config = Qwen3WMConnectorConfig(**m1_config)
        elif isinstance(m1_config, Qwen3WMConnectorConfig):
            self.m1_config = m1_config
        else:
            self.m1_config = Qwen3WMConnectorConfig()

        if isinstance(m2_config, dict):
            self.m2_config = Qwen3WMConnectorConfig(**m2_config)
        elif isinstance(m2_config, Qwen3WMConnectorConfig):
            self.m2_config = m2_config
        else:
            self.m2_config = Qwen3WMConnectorConfig()

        # Convenience alias expected by some helper utilities
        self.connector_config = self.m1_config

        # ---------------------------------------------------------------------
        # Predictor sub-config
        # ---------------------------------------------------------------------
        if isinstance(predictor_config, dict):
            self.predictor_config = Qwen3WMPredictorConfig(**predictor_config)
        elif isinstance(predictor_config, Qwen3WMPredictorConfig):
            self.predictor_config = predictor_config
        else:
            self.predictor_config = Qwen3WMPredictorConfig()


__all__ = [
    "Qwen3WMConnectorConfig",
    "Qwen3WMPredictorConfig",
    "Qwen3WMConfig",
]
