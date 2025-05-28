# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================

from __future__ import annotations

import argparse
from typing import Any

import deepspeed
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.datasets import PreferenceDataset, SupervisedDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class MCPGTrainer(SupervisedTrainer):
    TRAINING_TYPE = 'mcpg'
    DATASET_TYPE = SupervisedDataset

    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.scale_coeff = args.scale_coeff
        self.alpha = 0.7
        super().__init__(args, ds_train_config)

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.reference_model, _ = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )

    def init_engines(self) -> None:
        super().init_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_config,
        )

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def loss(  # pylint: disable=too-many-locals
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        rewards: torch.FloatTensor, # size = (B, 1)
    ) -> dict[str, torch.Tensor]:
        """
            Loss function for the MCPG algorithm.
        """

        sequence_log_probs = self.compute_log_probs(
            self.model.module,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(
                self.reference_model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        action_mask = attention_mask[:, 1:]
        kl_divergence_estimate = sequence_log_probs - ref_sequence_log_probs
        #kl_ratio = torch.exp(kl_divergence_estimate * action_mask) #exp
        #kl_ratio = kl_divergence_estimate * action_mask #mask 
        #kl_ratio = kl_divergence_estimate # v2
        kl_ratio = sequence_log_probs # v3
        loss = -rewards.unsqueeze(-1) * kl_ratio
        loss = torch.sum(loss * action_mask) / action_mask.sum()

        return {
            'loss': loss,
        }

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        rewards: torch.FloatTensor, # size = (B, 1)
    ) -> dict[str, Any]:
        """
            Perform a single training step.
        """
        loss_dict = self.loss(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            rewards = rewards,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():
            loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
