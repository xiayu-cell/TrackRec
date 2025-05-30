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

from safe_rlhf.datasets import PreferenceDataset, SoftmaxPreferenceDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class SDPOTrainer(SupervisedTrainer):
    TRAINING_TYPE = 'sdpo'
    DATASET_TYPE = SoftmaxPreferenceDataset

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
        better_input_ids: torch.LongTensor,  # size = (B, L)
        better_attention_mask: torch.BoolTensor,  # size = (B, L)
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for the DPO algorithm.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, torch.Tensor]: loss, reward, better sample reward, worse sample reward
        """
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)
        list_size = worse_input_ids.size(1)

        temp = []
        better_sample_rewards = []
        worse_sample_rewards = []
        sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
            self.model.module,
            input_ids=torch.cat([better_input_ids, worse_input_ids.reshape(batch_size * list_size, -1)], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask.reshape(batch_size * list_size, -1)], dim=0),
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = torch.split(sequence_log_probs, [batch_size, batch_size * list_size], 0)

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                input_ids=torch.cat([better_input_ids, worse_input_ids.reshape(batch_size * list_size, -1)], dim=0),
                attention_mask=torch.cat([better_attention_mask, worse_attention_mask.reshape(batch_size * list_size, -1)], dim=0),
            )
            (
                ref_better_sequence_log_probs,  # size = (B, L - 1)
                ref_worse_sequence_log_probs,  # size = (B, L - 1)
            ) = torch.split(ref_sequence_log_probs, [batch_size, batch_size * list_size], 0)
        worse_sequence_log_probs = worse_sequence_log_probs.reshape(batch_size, list_size, -1)
        ref_worse_sequence_log_probs = ref_worse_sequence_log_probs.reshape(batch_size, list_size, -1)
        for i in range(batch_size):
            for j in range(list_size):
                assert not torch.all(
                    torch.eq(better_input_ids[i], worse_input_ids[i, j]),
                ).item(), 'The better and worse answers are the same!'
                better_end_index = better_attention_mask[i].nonzero()[-1].squeeze().item()
                worse_end_index = worse_attention_mask[i, j].nonzero()[-1].squeeze().item()
                diverge_index = (
                    (better_input_ids[i] != worse_input_ids[i, j]).nonzero()[0].squeeze().item()
                )
                assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
                assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'

                better_seq_slice = slice(diverge_index, better_end_index + 1)
                worse_seq_slice = slice(diverge_index, worse_end_index + 1)

                # size = ()
                better_log_prob = better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
                worse_log_prob = worse_sequence_log_probs[i, j, worse_seq_slice].sum(dim=-1)
                ref_better_log_prob = ref_better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
                ref_worse_log_prob = ref_worse_sequence_log_probs[i, j, worse_seq_slice].sum(dim=-1)
                better_log_ratio = better_log_prob - ref_better_log_prob
                worse_log_ratio = worse_log_prob - ref_worse_log_prob

                temp.append(torch.exp(self.scale_coeff * (worse_log_ratio - better_log_ratio)))
                better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
                worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())
        #print(temp)
        temp1 = -torch.log(sum(temp))
        loss = -F.logsigmoid(temp1)  # size = ()

        return {
            'loss': loss,
        }

    def train_step(
        self,
        better_input_ids: torch.LongTensor,  # size = (B, L)
        better_attention_mask: torch.BoolTensor,  # size = (B, L)
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, Any]: training loss, reward, learning rate
        """
        loss_dict = self.loss(
            better_input_ids=better_input_ids,
            better_attention_mask=better_attention_mask,
            worse_input_ids=worse_input_ids,
            worse_attention_mask=worse_attention_mask,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():
            loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss,
        }
