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
"""Testing suite for the OmniDeBERTa feature extraction."""

import unittest

import numpy as np

from transformers import OmniDebertaFeatureExtractor
from transformers.testing_utils import require_torch

from ...test_processing_common import floats_list
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


class OmniDebertaFeatureExtractionTester:
    def __init__(self, parent, batch_size=7, min_seq_length=400, max_seq_length=2000, feature_size=128):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = 0.0
        self.sampling_rate = 16000

    def prepare_feat_extract_dict(self):
        return {"feature_size": self.feature_size, "sampling_rate": self.sampling_rate}

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        """Feature-shaped inputs `(length, feature_size)` for the padding tests of the mixin."""
        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            lengths = range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            speech_inputs = [floats_list((length, self.feature_size)) for length in lengths]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


@require_torch
class OmniDebertaFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = OmniDebertaFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = OmniDebertaFeatureExtractionTester(self)

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [np.asarray(floats_list((1, length))[0]) for length in (400, 1200, 2000)]
        batch = feature_extractor(speech_inputs, return_tensors="np")
        single = feature_extractor(speech_inputs[0], return_tensors="np")
        self.assertEqual(batch.input_features.ndim, 3)
        self.assertEqual(batch.input_features.shape[-1], self.feat_extract_tester.feature_size)
        num_frames = int(single.input_features_mask[0].sum())
        np.testing.assert_allclose(
            batch.input_features[0, :num_frames], single.input_features[0, :num_frames], rtol=1e-5, atol=1e-5
        )

    def test_frame_mask(self):
        """A frame is real when its window ends on real audio: 1 s gives 99 frames (window 320, hop 160)."""
        feature_extractor = self.feature_extraction_class()
        batch = feature_extractor([np.zeros(16000), np.zeros(8000)], return_tensors="np")
        self.assertEqual(batch.input_features_mask.sum(1).tolist(), [99, 49])
        self.assertTrue((batch.input_features[1, 49:] == 0).all())

    def test_clip_shorter_than_a_frame(self):
        batch = self.feature_extraction_class()(np.zeros(100), return_tensors="np")
        self.assertEqual(batch.input_features.shape[:2], (1, 1))
        self.assertEqual(int(batch.input_features_mask.sum()), 0)

    def test_sampling_rate_checked(self):
        with self.assertRaisesRegex(ValueError, "16000"):
            self.feature_extraction_class()(np.zeros(100), sampling_rate=8000)

    @unittest.skip(reason="log-mel features are not padded or truncated by `pad`")
    def test_padding_from_list(self):
        pass

    @unittest.skip(reason="log-mel features are not padded or truncated by `pad`")
    def test_padding_from_array(self):
        pass

    @unittest.skip(reason="log-mel features are not padded or truncated by `pad`")
    def test_truncation_from_list(self):
        pass

    @unittest.skip(reason="log-mel features are not padded or truncated by `pad`")
    def test_truncation_from_array(self):
        pass
