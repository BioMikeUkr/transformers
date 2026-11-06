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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils.import_utils import is_torchvision_greater_or_equal

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


class OmniDebertaImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("size", {"height": 18, "width": 18})
        super().__init__(**kwargs)


@require_torch
@require_vision
class OmniDebertaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = OmniDebertaImageProcessingTester

    @unittest.skipIf(
        not is_torchvision_greater_or_equal("0.27"), "torchvision < 0.27 resizes with BICUBIC instead of LANCZOS"
    )
    def test_backends_equivalence(self):
        super().test_backends_equivalence()

    @unittest.skipIf(
        not is_torchvision_greater_or_equal("0.27"), "torchvision < 0.27 resizes with BICUBIC instead of LANCZOS"
    )
    def test_backends_equivalence_batched(self):
        super().test_backends_equivalence_batched()

    @unittest.skip(reason="the images are RGB")
    def test_call_numpy_4_channels(self):
        pass
