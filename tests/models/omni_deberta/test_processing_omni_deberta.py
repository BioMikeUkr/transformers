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

import unittest

import numpy as np

from transformers import OmniDebertaProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


SPECIAL_TOKENS = {
    "boi_token": "[IMG]",
    "eoi_token": "[/IMG]",
    "boa_token": "[AUD]",
    "eoa_token": "[/AUD]",
    "blank_token": "[BLANK]",
    "transcribe_token": "[TRANSCRIBE]",
    "describe_token": "[DESCRIBE]",
    "unused_token": "[UNUSED0]",
    "embed_token": "[EMBED]",
    "segment_token": "[SEGMENT]",
    "image_token": "[IMG_PATCH]",
    "audio_token": "[AUD_FRAME]",
}


@require_vision
@require_torch
class OmniDebertaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmniDebertaProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained(
            "hf-internal-testing/tiny-random-DebertaV2Model", extra_special_tokens=SPECIAL_TOKENS
        )

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(size={"height": 32, "width": 32})

    @staticmethod
    def prepare_processor_dict():
        return {"patch_size": 16}

    def test_embedding_layout(self):
        """`[CLS] [EMBED] [IMG] patches [/IMG] [AUD] audio [/AUD] words [SEP]` when the text has no placeholders."""
        processor = self.get_processor()
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        clip = np.zeros(16000, dtype=np.float32)
        inputs = processor(text="a cat", images=image, audio=clip, return_tensors="pt")
        tokens = processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].tolist())
        num_audio = processor._num_audio_tokens(99)
        expected = ["[CLS]", "[EMBED]", "[IMG]", *["[IMG_PATCH]"] * 4, "[/IMG]", "[AUD]", *["[AUD_FRAME]"] * num_audio]
        self.assertEqual(tokens[: len(expected)], expected)
        self.assertEqual(tokens[len(expected)], "[/AUD]")
        self.assertEqual(tokens[-1], "[SEP]")
        self.assertEqual(tuple(inputs.pixel_values.shape), (1, 3, 32, 32))
        self.assertEqual(int(inputs.input_features_mask.sum()), 99)

    def test_task_marker(self):
        processor = self.get_processor()
        inputs = processor(images=np.zeros((32, 32, 3), dtype=np.uint8), text="a dog", task="segment")
        self.assertEqual(inputs.input_ids[0][1], processor.segment_token_id)
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            processor(text="a dog", task="answer")

    def test_media_per_sample(self):
        """One image per sample when the texts have no placeholders; placeholders put the media in the text."""
        processor = self.get_processor()
        images = [np.zeros((32, 32, 3), dtype=np.uint8)] * 2
        inputs = processor(text=["one", "two"], images=images)
        counts = [ids.count(processor.image_token_id) for ids in inputs.input_ids]
        self.assertEqual(counts, [4, 4])
        inputs = processor(text=f"before {processor.image_token} after", images=images[0])
        tokens = processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        self.assertLess(tokens.index("[IMG]"), len(tokens) - 2)
        self.assertNotEqual(tokens[2], "[IMG]")

    @unittest.skip(reason="the processor puts [EMBED] after [CLS], the tokenizer alone does not")
    def test_subprocessor_defaults_0_text(self):
        pass

    def test_waveform_as_list(self):
        processor = self.get_processor()
        as_list = processor(audio=list(np.zeros(16000, dtype=np.float32)))
        as_array = processor(audio=np.zeros(16000, dtype=np.float32))
        self.assertEqual(as_list.input_ids, as_array.input_ids)

    def test_long_clip_fits(self):
        """A clip longer than the room `max_length` leaves loses its end; the sequence then fills `max_length`."""
        processor = self.get_processor()
        processor.max_length = 64
        inputs = processor(text="a few words", audio=np.zeros(16000 * 10, dtype=np.float32))
        self.assertEqual(len(inputs.input_ids[0]), 64)
        short = processor(text="a few words", audio=np.zeros(16000, dtype=np.float32))
        self.assertLess(len(short.input_ids[0]), 64)

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()
        clips = [np.zeros(n, dtype=np.float32) for n in (16000, 5000, 321)]
        inputs = processor(audio=clips)
        from_call = [ids.count(processor.audio_token_id) for ids in inputs.input_ids]
        self.assertEqual(
            from_call, processor._get_num_multimodal_tokens(audio_lengths=[16000, 5000, 321])["num_audio_tokens"]
        )
        inputs = processor(images=[np.zeros((20, 30, 3), dtype=np.uint8)])
        self.assertEqual(
            [inputs.input_ids[0].count(processor.image_token_id)],
            processor._get_num_multimodal_tokens(image_sizes=[(20, 30)])["num_image_tokens"],
        )
