# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""
Librispeech language modeling dataset.
    this is an extented from huggingface's official implementation to 
    allow the use of train-960 trainscript and lm_corpus for LM training
Authors
 * Jianyuan Zhong 2021
"""

from __future__ import absolute_import, division, print_function

import datasets
import re
from typing import Optional


_CITATION = """\
@inproceedings{panayotov2015librispeech,
    title={Librispeech: an ASR corpus based on public domain audio books},
    author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
    booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
    pages={5206--5210},
    year={2015},
    organization={IEEE}
}
"""

_DESCRIPTION = """\
Language modeling resources to be used in conjunction with the LibriSpeech ASR corpus.
"""

class LibrispeechLm(datasets.GeneratorBasedBuilder):
    """Librispeech language modeling dataset."""

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            supervised_keys=("text", "text"),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path_dic = {}
        for split_name, files in self.config.data_files.items():
            if (
                split_name == "train"
            ):
                path_dic[split_name] = dl_manager.download_and_extract(files)
            else:
                path_dic[split_name] = dl_manager.download_and_extract(files)

        return [
            datasets.SplitGenerator(
                name=split_name, gen_kwargs={"archive_path": archive_path}
            )
            for split_name, archive_path in path_dic.items()
        ]

    def _generate_examples(self, archive_path):
        """Yields examples."""
        for p in archive_path:
            with open(p, "r", encoding="utf-8") as f:
                for key, line in enumerate(f):
                    line = re.sub(
                        r"\d+-\d+-\d+\s", "", line
                    )  # remove ids in transcripts
                    text = line.strip()

                    # Skip empty lines.
                    # very long sentences (>1000 char) are removed to prevent OOM
                    if text and len(text) < 1000:
                        yield key, {"text": text}
