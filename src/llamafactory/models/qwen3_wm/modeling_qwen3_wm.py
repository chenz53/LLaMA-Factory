from typing import Dict, Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    KwargsForCausalLM,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3PreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MLPConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Predictor(Qwen3Model):
    """
    A lightweight predictor module inspired by `VJEPA2Predictor`.

    The predictor takes the encoder hidden states from the base Qwen3 language model
    and tries to predict (reconstruct) the hidden states at the positions specified
    by `target_mask`, conditioned on the tokens specified by `context_mask`.

    Differences w.r.t. VJEPA2:
    * The inputs are 1-D token sequences (no spatial patches).
    * We do not perform any sophisticated token sorting – the original ordering is
      preserved.
    * A single learnable `mask_token` is used to replace the hidden states at
      `target_mask` positions before the Transformer stack.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embed_tokens = None
        self.padding_idx = None
        self.vocab_size = None

        # A single learnable vector that is inserted at target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # Optional projection to a different predictor hidden size
        self.input_dim = getattr(config, "dim", self.hidden_size)
        if self.input_dim != self.hidden_size:
            self.input_proj = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        else:
            self.input_proj = nn.Identity()
 
        # Output projection back to the base hidden size
        if self.input_dim != self.hidden_size:
            self.output_proj = nn.Linear(self.hidden_size, self.input_dim, bias=False)
        else:
            self.output_proj = nn.Identity()

    @staticmethod
    def _apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        """Extract tokens from `tensor` given a list of indices."""
        collected = []
        for mask in masks:
            mask = mask.to(tensor.device)
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, tensor.size(-1))
            collected.append(torch.gather(tensor, dim=1, index=mask_expanded))
        return torch.cat(collected, dim=0)

    @staticmethod
    def _get_position_ids(
        hidden_states: torch.Tensor,
        target_mask: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return position IDs distinguishing target vs context.

        Parameters
        ----------
        hidden_states: torch.Tensor
            (batch, seq_len, hidden)
        target_mask: list[torch.Tensor]
            A list of length `batch`, each tensor contains the *indices* (ints) of
            target tokens **for that sample**.

        Returns
        -------
        torch.LongTensor with shape (batch, seq_len) where 1 denotes a target
        token and 0 denotes context.
        """
        device = hidden_states.device
        batch_size, seq_len, _ = hidden_states.size()
        pos_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Iterate per sample for correct scatter dimension
        for b, indices in enumerate(target_mask):
            if indices.numel() == 0:
                continue
            pos_ids[b].scatter_(0, indices.to(device), 1)
        return pos_ids

    def forward(
        self,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        """Forward pass of the predictor.

        Args:
            inputs_embeds (`torch.Tensor`): Input embeddings from the base model, shape `(batch, seq_len, hidden)`.
            attention_mask (`torch.Tensor`): Attention mask, shape `(batch, seq_len)`.
            position_ids (`torch.LongTensor`): Position IDs, shape `(batch, seq_len)`.
            past_key_values (`Cache`): Past key values, shape `(batch, num_layers, num_heads, seq_len, head_dim)`.
            use_cache (`bool`): Whether to use cache.
            cache_position (`torch.LongTensor`): Cache position, shape `(batch, seq_len)`.
            context_mask (`List[torch.Tensor]`): Token indices to be used as context (currently unused but kept for API compatibility).
            target_mask (`List[torch.Tensor]`): Token indices that should be predicted.
        """
        inputs_embeds = self.input_proj(inputs_embeds)  # (B, S, D)

        # create position embeddings to be shared across the decoder layers
        if position_ids is None:
            position_ids = self._get_position_ids(inputs_embeds, target_mask)

        # Replace target tokens with the learnable `mask_token`
        mask_token_expanded = self.mask_token.expand(inputs_embeds.size(0), -1, -1)  # (B,1,D)
        for b, t_mask in enumerate(target_mask):
            idx = t_mask.to(inputs_embeds.device).long().view(-1)
            if idx.numel() > 0:
                inputs_embeds[b, idx] = mask_token_expanded[b, 0]

        hidden_states = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        ).last_hidden_state

        # Predictor transformer
        hidden_states = self.output_proj(hidden_states)

        # Extract predictions corresponding to `target_mask`
        # pred_hidden_states = self._apply_masks(hidden_states, target_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3WM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        # Initialize connectors
        self.connectors = nn.ModuleDict(
            {
                f"m{i}_connector": MLPConnector(getattr(config, f"m{i}_config"))
                for i in range(1, 3)
            }
        )

        # Initialize model
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize predictor
        self.predictor = Predictor(config.predictor_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_multimodal(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        embeddings: Optional[Dict[str, torch.FloatTensor]] = None,
    ) -> torch.FloatTensor:
        """
        Prepare inputs for multimodal processing by merging text and modality embeddings.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            embeddings: Dictionary mapping modality keys to their embeddings

        Returns:
            torch.FloatTensor: Processed input embeddings ready for the model
        """
        if input_ids is None:
            return None

        # Get base embeddings from input_ids
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # If no context embeddings, return base embeddings
        if embeddings is None:
            return inputs_embeds

        # Merge modality embeddings
        for modality_key, embeddings in embeddings.items():
            if not modality_key.startswith("m"):
                continue

            token_id = getattr(self.config, f"{modality_key}_token_id", None)
            if token_id is None:
                continue

            # Project/merge embeddings
            merged_embeds = self.connectors[f"{modality_key}_connector"](embeddings.to(inputs_embeds.dtype))

            # Replace token embeddings with merged embeddings
            mask = input_ids == token_id
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                mask_expanded.to(inputs_embeds.device),
                merged_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        embeddings: Optional[Dict[str, torch.FloatTensor]] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            assert self.connectors is not None, "Connectors must be initialized!!"
            inputs_embeds = self.prepare_inputs_for_multimodal(input_ids, embeddings)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self, input_ids: torch.LongTensor, embeddings: Optional[Dict[str, torch.FloatTensor]] = None, **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if embeddings is not None:
            inputs_embeds = self.prepare_inputs_for_multimodal(input_ids, embeddings)
        else:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        return super().generate(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
