# Copyright 2026 Knowledgator and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the mask-predict generation of OmniDeBERTa."""

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from .test_modeling_omni_deberta import OmniDebertaModelTester


if is_torch_available():
    import torch

    from transformers import OmniDebertaForMaskedLM, OmniDebertaGenerationConfig


@require_torch
class OmniDebertaGenerationTest(unittest.TestCase):
    def setUp(self):
        self.tester = OmniDebertaModelTester(self)
        self.config = self.tester.get_config()
        torch.manual_seed(0)
        self.model = OmniDebertaForMaskedLM(self.config).to(torch_device).eval()

    def prompts(self, audio=True):
        config = self.config
        input_ids = self.tester.build_input_ids(config, task_token_id=93, audio=audio)
        input_ids = torch.cat([input_ids[:, : -self.tester.num_words - 1], input_ids[:, -1:]], 1)  # no prompt words
        inputs = {
            "input_ids": input_ids,
            "pixel_values": torch.rand(self.tester.batch_size, 3, 8, 8, device=torch_device),
        }
        if audio:
            inputs["input_features"] = torch.rand(
                self.tester.batch_size, self.tester.num_frames, 8, device=torch_device
            )
        return inputs

    def test_generate_shapes(self):
        output = self.model.generate(**self.prompts(), max_new_tokens=5, return_dict_in_generate=True)
        self.assertEqual(output.sequences.shape[0], self.tester.batch_size)
        self.assertLessEqual(output.sequences.shape[1], 5)
        self.assertEqual(output.scores.shape, output.sequences.shape)
        self.assertGreater(output.num_forward_passes, 0)
        special = (self.config.mask_token_id, self.config.eos_token_id, self.config.bos_token_id)
        self.assertFalse(torch.isin(output.sequences, torch.tensor(special, device=torch_device)).any())

    def test_generate_is_deterministic_and_batch_independent(self):
        """Prompts of different lengths in one padded batch give what they give alone."""
        inputs = self.prompts(audio=False)
        input_ids = inputs["input_ids"]
        longer = torch.cat(
            [input_ids[:1, :-1], torch.tensor([[20, 21, 22]], device=torch_device), input_ids[:1, -1:]], 1
        )
        shorter = input_ids[1:2]
        pad = torch.zeros(1, longer.shape[1] - shorter.shape[1], dtype=torch.long, device=torch_device)
        batch_ids = torch.cat([longer, torch.cat([shorter, pad], 1)])
        attention_mask = (
            torch.arange(batch_ids.shape[1], device=torch_device)
            < torch.tensor([[longer.shape[1]], [shorter.shape[1]]], device=torch_device)
        ).long()
        pixel_values = inputs["pixel_values"][:2]
        batched = self.model.generate(
            batch_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_new_tokens=4
        )
        for row, ids in enumerate((longer, shorter)):
            single = self.model.generate(ids, pixel_values=pixel_values[row : row + 1], max_new_tokens=4)
            torch.testing.assert_close(batched[row, : single.shape[1]], single[0])
        again = self.model.generate(
            batch_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_new_tokens=4
        )
        torch.testing.assert_close(batched, again)

    def test_generation_config_round_trip(self):
        self.model.generation_config.block_size = 3
        self.model.generation_config.repetition_penalty = 2.0
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            reloaded = OmniDebertaForMaskedLM.from_pretrained(directory)
        self.assertEqual(reloaded.generation_config.block_size, 3)
        self.assertEqual(reloaded.generation_config.repetition_penalty, 2.0)

    def test_one_slot_per_pass(self):
        output = self.model.generate(
            **self.prompts(audio=False), max_new_tokens=3, stop_at_eos=False, drop_repeated_slots=False,
            repetition_penalty=1.0, return_dict_in_generate=True,
        )  # fmt: skip
        self.assertEqual(output.num_forward_passes, 3)
        self.assertEqual(output.sequences.shape, (self.tester.batch_size, 3))

    def test_sampling_is_seeded(self):
        inputs = self.prompts(audio=False)
        torch.manual_seed(1)
        first = self.model.generate(**inputs, max_new_tokens=4, temperature=1.0)
        torch.manual_seed(1)
        torch.testing.assert_close(first, self.model.generate(**inputs, max_new_tokens=4, temperature=1.0))

    def test_prompt_must_end_with_eos(self):
        inputs = self.prompts(audio=False)
        inputs["input_ids"] = inputs["input_ids"][:, :-1]
        with self.assertRaisesRegex(ValueError, "end with"):
            self.model.generate(**inputs)

    def test_generation_config_validated(self):
        with self.assertRaises(ValueError):
            OmniDebertaGenerationConfig(block_size=-1)
        with self.assertRaises(ValueError):
            OmniDebertaGenerationConfig(repetition_penalty=0.5)
        with self.assertRaises(ValueError):
            OmniDebertaGenerationConfig(top_k=5)
        self.assertIsInstance(self.model.generation_config, OmniDebertaGenerationConfig)
