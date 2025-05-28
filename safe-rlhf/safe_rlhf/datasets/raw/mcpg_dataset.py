# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Helpful and Harmless Dialogue Datasets from Anthropic."""

from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'LocalMCPGDataset',
]


class LocalMCPGDataset(RawDataset):
    NAME: ClassVar[str] = 'local-mcpg-dataset'
    DATA_DIR: ClassVar[str | None] = None

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(
            "json",
            data_files=path,
            split='train',
        )

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['input'],
            answer=data['answer'],
            reward=data['reward'],
            better=None
        )

    def __len__(self) -> int:
        return len(self.data)

