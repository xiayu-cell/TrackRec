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
"""Dataset class for preference training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import random

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'PreferenceDataset',
    'PreferenceCollator',
    'PreferenceSample',
    'PreferenceBatch',
    'SoftmaxPreferenceSample',
    'SoftmaxPreferenceBatch',
    'SoftmaxPreferenceDataset',
    'SoftmaxPreferenceCollator',
]


class PreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: torch.LongTensor  # size = (L,)


class PreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)

    worse_input_ids: torch.LongTensor  # size = (B, L)
    worse_attention_mask: torch.BoolTensor  # size = (B, L)


class PreferenceDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PreferenceSample:
        prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        better_answer = raw_sample['answer']
        worse_answer = raw_sample['other_answer']
        better = raw_sample['better']
        if not better:
            better_answer, worse_answer = worse_answer, better_answer

        better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)
        if (
            better_input_ids.size() == worse_input_ids.size()
            and torch.all(torch.eq(better_input_ids, worse_input_ids)).item()
        ):
            raise ValueError(
                'Two responses get the same `input_ids` after tokenization.\n\n'
                f'Prompt: {prompt}\n\n'
                f'Better answer: {better_answer}\n\n'
                f'Worse answer: {worse_answer}',
            )
        return {
            'better_input_ids': better_input_ids,  # size = (L,)
            'worse_input_ids': worse_input_ids,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)


class PreferenceCollator(CollatorBase):
    def __call__(self, samples: list[PreferenceSample]) -> PreferenceBatch:
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)

        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'worse_input_ids': worse_input_ids,  # size = (B, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, L)
        }

class SoftmaxPreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: list[torch.LongTensor]  # size = (L,)


class SoftmaxPreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)

    worse_input_ids: torch.LongTensor  # size = (B, N, L)
    worse_attention_mask: torch.BoolTensor # size = (B, N, L)


class SoftmaxPreferenceDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SoftmaxPreferenceSample:
        prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        better_answer = raw_sample['answer']
        worse_answers = raw_sample['other_answers']
        better = raw_sample['better']

        better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        worse_input_ids_list = []
        rand_worse_answers = random.sample(worse_answers, 2)
        for i, worse_answer in enumerate(rand_worse_answers):
            worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)
            if (
                better_input_ids.size() == worse_input_ids.size()
                and torch.all(torch.eq(better_input_ids, worse_input_ids)).item()
            ):
                raise ValueError(
                    'Two responses get the same `input_ids` after tokenization.\n\n'
                    f'Prompt: {prompt}\n\n'
                    f'Better answer: {better_answer}\n\n'
                    f'Worse answer: {worse_answer}',
                )
            worse_input_ids_list.append(worse_input_ids)
        return {
            'better_input_ids': [better_input_ids],  # size = (L,)
            'worse_input_ids_list': worse_input_ids_list,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor], [list[dict[str, list[torch.Tensor]]]]]:
        return SoftmaxPreferenceCollator(self.tokenizer.pad_token_id)


class SoftmaxPreferenceCollator(CollatorBase):
    def __call__(self, samples: list[SoftmaxPreferenceSample]) -> SoftmaxPreferenceBatch:
        input_ids = []
        batch_size = len(samples)
        list_size = len(samples[0]['worse_input_ids_list'])
        for sample in samples:
            input_ids.append(sample['better_input_ids'][0])
            for worse_input_ids in sample['worse_input_ids_list']:
                input_ids.append(worse_input_ids)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (B * (N + 1), L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (B * (N + 1), L)

        better_indices = [_ for _ in range(0, batch_size * (list_size + 1), (list_size + 1))]
        better_input_ids = input_ids[better_indices]
        better_attention_mask = attention_mask[better_indices]

        worse_indices = [(_ + b * (list_size + 1)) for b in range(batch_size) for _ in range(1, (list_size + 1))]
        worse_input_ids = input_ids[worse_indices].reshape(batch_size, list_size, -1)
        worse_attention_mask = attention_mask[worse_indices].reshape(batch_size, list_size, -1)

        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'worse_input_ids': worse_input_ids,  # size = (B, N, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, N, L)
        }
