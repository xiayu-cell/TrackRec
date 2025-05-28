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

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.configs import IGNORE_INDEX, PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
    'KuaiyiSupervisedDataset',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class SupervisedDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SupervisedSample:
        if raw_sample.get('input') is None and raw_sample.get('dialogue') is None:
            raise ValueError('Either `input` or `dialogue` must be provided.')
        if raw_sample.get('input') is not None and raw_sample.get('dialogue') is not None:
            raise ValueError('At most one of `input` and `dialogue` can be provided.')

        if raw_sample.get('input') is not None:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            if not isinstance(input, str):
                raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str.')
            prompt = format_prompt(input=input, eos_token=self.tokenizer.eos_token)
            answer = raw_sample['answer']
            text = prompt + answer + self.tokenizer.eos_token

            input_ids = self.tokenize(text)
            labels = input_ids.clone()
            # Mask non-assistant input
            labels[: len(self.tokenize(prompt))] = IGNORE_INDEX
            rewards = torch.tensor(0.0)
            if raw_sample.get('reward') is not None:
                rewards = torch.tensor(raw_sample['reward'])
            return {'input_ids': input_ids, 'labels': labels, 'rewards': rewards}

        dialogue = raw_sample['dialogue']  # is not None
        text = PROMPT_BEGIN
        offsets = [0]
        input_ids = torch.empty(0, dtype=torch.long)
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                text += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT
            else:
                # Assistant input
                text += line + self.tokenizer.eos_token
            input_ids = self.tokenize(text)
            offsets.append(len(input_ids))

        labels = input_ids.clone()
        # Mask non-assistant input
        for begin, end in zip(offsets[::2], offsets[1::2]):
            labels[begin:end] = IGNORE_INDEX

        return {
            'input_ids': input_ids,  # size = (L,)
            'labels': labels,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)


class KuaiyiSupervisedDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SupervisedSample:
        data = raw_sample['data']
        offsets = [0]
        input_ids = torch.empty(0, dtype=torch.long)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if "system" in data[0]:
            messages[0]["content"] = data[0]["system"]
        for line in data:
            mask = line['mask']
            messages.append({"role": "user", "content": line['question']})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenize(prompt)
            if not mask:
                offsets.append(len(input_ids))

            messages.append({"role": "assistant", "content": line['answer']})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = self.tokenize(text)
            if not mask:
                offsets.append(len(input_ids))

        labels = input_ids.clone()
        for begin, end in zip(offsets[::2], offsets[1::2]):
            labels[begin:end] = IGNORE_INDEX
        rewards = torch.tensor(0.0)
        if raw_sample.get('reward') is not None:
            rewards = torch.tensor(raw_sample['reward'])
        return {'input_ids': input_ids, 'labels': labels, 'rewards': rewards}

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)


class SupervisedCollator(CollatorBase):
    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        labels = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        rewards = torch.tensor([sample['rewards'] for sample in samples])
        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
            'rewards': rewards # size = (B, )
        }
