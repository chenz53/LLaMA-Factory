# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import (
    IGNORE_INDEX,
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    AUDIO_PLACEHOLDER,
    MULTIMODAL_EMBEDDING_PLACEHOLDERS,
)
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class WorldModelTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        # Ensure a predictor head exists for next-embedding prediction
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "n_embd", None)
        if hidden_size is None and hasattr(self.model, "get_output_embeddings"):
            hidden_size = self.model.get_output_embeddings().weight.size(1)
        if not hasattr(self.model, "next_embedding_head"):
            self.model.next_embedding_head = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(
                self.model.device
            )

        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def compute_sft_loss(self, outputs, labels):
        """Compute standard language modeling (SFT) loss using label_smoother when available."""
        if self.label_smoother is not None and labels is not None:
            return self.label_smoother(outputs, labels)
        else:
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def compute_next_embedding_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the assistant embedding representation from user embedding representation.

        hidden_states: (bsz, seq_len, hidden_size) from the last layer
        input_ids:     (bsz, seq_len)
        labels:        (bsz, seq_len) original (unmasked) labels to distinguish user vs assistant tokens
        """
        if hidden_states is None or labels is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        device = hidden_states.device
        loss = torch.tensor(0.0, device=device)

        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

        # Use *all* non-padding tokens (include standard tokens and embedded-token placeholders).
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        non_pad_mask = input_ids != pad_token_id

        # Separate user vs assistant positions based on label mask
        user_mask = non_pad_mask & (labels == IGNORE_INDEX)
        ass_mask = non_pad_mask & (labels != IGNORE_INDEX)

        preds, targets = [], []
        for b in range(input_ids.size(0)):
            if user_mask[b].any() and ass_mask[b].any():
                user_vec = hidden_states[b][user_mask[b]].mean(dim=0)
                with torch.no_grad():
                    target_vec = hidden_states[b][ass_mask[b]].mean(dim=0)
                pred_vec = self.model.next_embedding_head(user_vec)
                preds.append(pred_vec)
                targets.append(target_vec)

        if preds:
            pred_stack = torch.stack(preds)
            target_stack = torch.stack(targets)
            if self.finetuning_args.world_model_loss_type == "l1":
                loss = F.l1_loss(pred_stack, target_stack)
            else:
                loss = F.mse_loss(pred_stack, target_stack)
        return loss

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # Add batch debugging prints
        # self._debug_print_batch(inputs)

        # Mask multimodal tokens so that loss is computed only on textual content.
        labels = inputs.get("labels")
        if labels is not None:
            mm_token_ids = self._get_mm_token_ids()
            if mm_token_ids:
                with torch.no_grad():
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    for t_id in mm_token_ids:
                        mask |= labels == t_id
                    labels = labels.masked_fill(mask, IGNORE_INDEX)
                inputs["labels"] = labels
        self._debug_print_batch(inputs)

        # Forward pass with hidden states
        outputs = model(**inputs, output_hidden_states=True)

        # Compute individual losses
        sft_loss = self.compute_sft_loss(outputs, inputs.get("labels"))
        hidden_states = (
            outputs.hidden_states[-1]
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
            else None
        )
        embedding_loss = self.compute_next_embedding_loss(hidden_states, inputs["input_ids"], inputs.get("labels"))

        total_loss = (
            self.finetuning_args.lm_loss_weight * sft_loss
            + self.finetuning_args.embedding_loss_weight * embedding_loss
        )

        if kwargs.get("return_outputs", False):
            return total_loss, outputs
        return total_loss

    def _get_mm_token_ids(self) -> list[int]:
        """Return cached list of multimodal token ids to ignore in loss."""
        if hasattr(self, "_mm_token_ids"):
            return self._mm_token_ids

        tokens: set[str] = {IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, AUDIO_PLACEHOLDER}
        tokens.update(MULTIMODAL_EMBEDDING_PLACEHOLDERS.values())

        # Add wrapper tokens for embedding and vision segments
        for key in MULTIMODAL_EMBEDDING_PLACEHOLDERS:
            tokens.add(f"<|{key}_start|>")
            tokens.add(f"<|{key}_end|>")

        tokens.update({"<|vision_start|>", "<|vision_end|>"})

        # Retrieve additional tokens from the active multimodal plugin, if any
        mm_plugin = (
            getattr(getattr(self.data_collator, "template", None), "mm_plugin", None)
            if hasattr(self, "data_collator")
            else None
        )
        if mm_plugin is not None:
            for attr in ("image_token", "video_token", "audio_token"):
                tok = getattr(mm_plugin, attr, None)
                if tok:
                    tokens.add(tok)
            embedding_tokens = getattr(mm_plugin, "embedding_tokens", None)
            if embedding_tokens:
                tokens.update(embedding_tokens.values())

        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        token_ids: list[int] = []
        if tokenizer is not None:
            unk_id = tokenizer.unk_token_id
            for tok in tokens:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != unk_id:
                    token_ids.append(tid)

        self._mm_token_ids = list(set(token_ids))
        return self._mm_token_ids

    def _debug_print_batch(self, inputs: dict[str, Any]) -> None:
        """Simply print each field of the batch at rank0 only."""
        if hasattr(self, "is_world_process_zero") and self.is_world_process_zero():
            print("\n=== DEBUG BATCH ===")
            for key, value in inputs.items():
                print(f"{key}: {value}")
            print("===================\n")

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
