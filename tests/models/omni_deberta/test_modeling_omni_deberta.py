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
"""Testing suite for the PyTorch OmniDeBERTa model."""

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import OmniDebertaConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import (
        OmniDebertaForCTC,
        OmniDebertaForImageSegmentation,
        OmniDebertaForMaskedLM,
        OmniDebertaForQuestionAnswering,
        OmniDebertaForReplacedTokenDetection,
        OmniDebertaForSequenceClassification,
        OmniDebertaForTokenClassification,
        OmniDebertaModel,
    )


class OmniDebertaModelTester:
    """Every sequence: `[CLS] [EMBED] [IMG] patches [/IMG] [AUD] audio [/AUD] words [SEP]`."""

    def __init__(
        self,
        parent,
        batch_size=3,
        num_words=7,
        is_training=True,
        vocab_size=99,
        hidden_size=32,
        embedding_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        max_position_embeddings=64,
        position_buckets=8,
        image_size=8,
        patch_size=4,
        num_mel_bins=8,
        num_frames=12,
        num_labels=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_words = num_words
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.position_buckets = position_buckets
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_mel_bins = num_mel_bins
        self.num_frames = num_frames
        self.num_labels = num_labels
        self.num_patches = (image_size // patch_size) ** 2
        self.num_audio_tokens = ((num_frames - 1) // 2) // 2 + 1
        self.seq_length = 2 + (self.num_patches + 2) + (self.num_audio_tokens + 2) + num_words + 1

    def get_config(self, **kwargs):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "position_buckets": self.position_buckets,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_mel_bins": self.num_mel_bins,
            "audio_conv_channels": [4, 2],
            "relative_2d_layers": [1],
            "relative_2d_stride": 3,
            "modal_experts": True,
            "projection_dim": 24,
            "segment_head_dim": 8,
            "num_labels": self.num_labels,
            # the special tokens close the vocabulary, as in the released checkpoints
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "mask_token_id": 86,
            "boi_token_id": 87,
            "eoi_token_id": 88,
            "boa_token_id": 89,
            "eoa_token_id": 90,
            "blank_token_id": 91,
            "embed_token_id": 95,
            "segment_token_id": 96,
            "image_token_id": 97,
            "audio_token_id": 98,
            "expert_routed_token_ids": [0, 1, 2, 87, 88, 89, 90, 92, 93, 94, 95, 96],
        }
        config.update(kwargs)
        return OmniDebertaConfig(**config)

    def build_input_ids(self, config, task_token_id=None, image=True, audio=True):
        words = ids_tensor([self.batch_size, self.num_words], 83) + 3
        prefix = [config.bos_token_id, config.embed_token_id if task_token_id is None else task_token_id]
        if image:
            prefix += [config.boi_token_id] + [config.image_token_id] * self.num_patches + [config.eoi_token_id]
        if audio:
            prefix += [config.boa_token_id] + [config.audio_token_id] * self.num_audio_tokens + [config.eoa_token_id]
        prefix = torch.tensor(prefix, device=torch_device).expand(self.batch_size, -1)
        sep = torch.full((self.batch_size, 1), config.eos_token_id, device=torch_device)
        return torch.cat([prefix, words, sep], 1)

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = self.build_input_ids(config)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, -2:] = 0  # padding at the end of the first sequence
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        input_features = floats_tensor([self.batch_size, self.num_frames, self.num_mel_bins])
        input_features_mask = torch.ones(self.batch_size, self.num_frames, dtype=torch.bool, device=torch_device)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }
        return config, inputs

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()


@require_torch
class OmniDebertaModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            OmniDebertaModel,
            OmniDebertaForMaskedLM,
            OmniDebertaForReplacedTokenDetection,
            OmniDebertaForSequenceClassification,
            OmniDebertaForTokenClassification,
            OmniDebertaForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {}
    all_generative_model_classes = ()  # mask-predict generation is tested in test_generation_omni_deberta.py
    test_resize_embeddings = False  # media placeholders are token ids, a smaller vocabulary drops them

    def setUp(self):
        self.model_tester = OmniDebertaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OmniDebertaConfig, hidden_size=36)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is OmniDebertaForReplacedTokenDetection:
            inputs_dict["labels"] = torch.zeros_like(inputs_dict["input_ids"])
        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = OmniDebertaModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(**inputs)
        tester = self.model_tester
        self.assertEqual(result.last_hidden_state.shape, (tester.batch_size, tester.seq_length, tester.hidden_size))
        self.assertEqual(result.pooler_output.shape, (tester.batch_size, config.projection_dim))
        torch.testing.assert_close(
            result.pooler_output.norm(dim=-1), torch.ones(tester.batch_size, device=torch_device)
        )

    def test_text_only(self):
        config = self.model_tester.get_config()
        model = OmniDebertaModel(config).to(torch_device).eval()
        input_ids = self.model_tester.build_input_ids(config, image=False, audio=False)
        with torch.no_grad():
            result = model(input_ids)
        self.assertEqual(result.last_hidden_state.shape[:2], input_ids.shape)

    def test_media_placeholder_count_checked(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = OmniDebertaModel(config).to(torch_device).eval()
        inputs["pixel_values"] = inputs["pixel_values"][:2]
        with self.assertRaisesRegex(ValueError, "image patches"):
            model(**inputs)

    def test_attention_options(self):
        """The DeBERTa-v2 options give the same outputs with every attention implementation."""
        options = [
            {"pos_att_type": ["c2p", "p2c"]},
            {"pos_att_type": ["p2c"], "share_att_key": False},
            {"relative_attention": False},
            {"position_biased_input": True, "norm_rel_ebd": "none"},
            {"modal_experts": False, "embed_token_pooling": False},
            {"position_buckets": 0, "max_relative_positions": 16},
        ]
        _, inputs = self.model_tester.prepare_config_and_inputs()
        for option in options:
            with self.subTest(**option):
                config = self.model_tester.get_config(**option)
                outputs = {}
                for implementation in ("eager", "sdpa"):
                    config._attn_implementation = implementation
                    torch.manual_seed(0)
                    model = OmniDebertaModel(config).to(torch_device).eval()
                    with torch.no_grad():
                        outputs[implementation] = model(**inputs).last_hidden_state
                torch.testing.assert_close(outputs["eager"], outputs["sdpa"], rtol=1e-4, atol=1e-4)

    def test_expert_routing(self):
        """Patches and audio go to their experts; the special tokens follow the medium of their sequence."""
        config = self.model_tester.get_config()
        model = OmniDebertaModel(config).to(torch_device).eval()
        with_media = self.model_tester.build_input_ids(config)
        text = self.model_tester.build_input_ids(config, image=False, audio=False)
        text = torch.nn.functional.pad(text, (0, with_media.shape[1] - text.shape[1]), value=config.pad_token_id)
        input_ids = torch.cat([with_media[:1], text[:1]])
        modality_ids = model.get_modality_ids(input_ids)
        routing = model.get_expert_routing(input_ids, modality_ids)
        self.assertEqual(sum(routing.counts), input_ids.numel())
        counts = torch.tensor(routing.counts, device=torch_device)
        per_token = torch.repeat_interleave(torch.arange(3, device=torch_device), counts)[routing.inverse]
        per_token = per_token.view(input_ids.shape)
        self.assertTrue((per_token[modality_ids == 1] == 1).all())
        self.assertTrue((per_token[modality_ids == 2] == 2).all())
        self.assertTrue((per_token[0, :2] == 1).all())  # [CLS] [EMBED] of a sequence with an image
        self.assertTrue((per_token[1] == 0).all())  # a text sequence, padding included

    def test_positions_2d(self):
        config = self.model_tester.get_config()
        model = OmniDebertaModel(config)
        input_ids = self.model_tester.build_input_ids(config, audio=False)[:1]
        positions = model.get_positions_2d(model.get_modality_ids(input_ids))[0].tolist()
        # [CLS] [EMBED] [IMG], then a 2x2 grid with rows 3 apart, then [/IMG] after the last patch
        self.assertEqual(positions[:9], [0, 1, 2, 3, 4, 6, 7, 8, 9])

    def test_several_media_per_sequence(self):
        """Two images and two clips in one sequence: every medium fills its placeholders in reading order."""
        config = self.model_tester.get_config()
        model = OmniDebertaModel(config).to(torch_device).eval()
        tester = self.model_tester
        image = [config.boi_token_id] + [config.image_token_id] * tester.num_patches + [config.eoi_token_id]
        clip = [config.boa_token_id] + [config.audio_token_id] * tester.num_audio_tokens + [config.eoa_token_id]
        ids = [config.bos_token_id, config.embed_token_id, *image, *clip, *image, *clip, 20, 21, config.eos_token_id]
        input_ids = torch.tensor([ids], device=torch_device)
        pixel_values = floats_tensor([2, 3, config.image_size, config.image_size])
        input_features = floats_tensor([2, tester.num_frames, config.num_mel_bins])
        with torch.no_grad():
            result = model(input_ids, pixel_values=pixel_values, input_features=input_features)
            swapped = model(input_ids, pixel_values=pixel_values.flip(0), input_features=input_features)
        self.assertEqual(result.last_hidden_state.shape[1], len(ids))
        self.assertFalse(torch.allclose(result.pooler_output, swapped.pooler_output))  # the order matters
        positions = model.get_positions_2d(model.get_modality_ids(input_ids))[0]
        self.assertEqual(len(set(positions.tolist())), len(ids))  # unique, the second image included

    def test_load_one_modality(self):
        """A model built for text only loads from a full checkpoint, skips the media weights and gives the same text
        embeddings."""
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = OmniDebertaModel(config).to(torch_device).eval()
        input_ids = self.model_tester.build_input_ids(config, image=False, audio=False)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            text_only, info = OmniDebertaModel.from_pretrained(
                directory, modalities=["text"], output_loading_info=True
            )
        text_only = text_only.to(torch_device).eval()
        self.assertEqual(info["missing_keys"], set())
        self.assertIsNone(text_only.embeddings.image_stem)
        self.assertLess(text_only.num_parameters(), model.num_parameters())
        with torch.no_grad():
            torch.testing.assert_close(text_only(input_ids).pooler_output, model(input_ids).pooler_output)

    def test_for_ctc(self):
        config = self.model_tester.get_config()
        model = OmniDebertaForCTC(config).to(torch_device).eval()
        self.assertIsNone(model.model.embeddings.image_stem)
        self.assertIsNone(model.model.encoder.layer[0].image_intermediate)
        self.assertEqual(config.modalities, ["text", "image", "audio"])  # the caller's config is not changed
        input_ids = self.model_tester.build_input_ids(config, image=False)
        features = floats_tensor([self.model_tester.batch_size, self.model_tester.num_frames, config.num_mel_bins])
        labels = ids_tensor([self.model_tester.batch_size, 2], 83) + 3
        result = model(input_ids, input_features=features, labels=labels)
        self.assertEqual(
            result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_audio_tokens, config.vocab_size)
        )
        self.assertTrue(torch.isfinite(result.loss))
        with self.assertRaisesRegex(ValueError, "without the image modality"):
            model(input_ids, input_features=features, pixel_values=torch.zeros(1))

    def test_for_image_segmentation(self):
        config = self.model_tester.get_config()
        model = OmniDebertaForImageSegmentation(config).to(torch_device).eval()
        self.assertIsNone(model.model.embeddings.audio_stem)
        input_ids = self.model_tester.build_input_ids(config, task_token_id=config.segment_token_id, audio=False)
        pixel_values = floats_tensor([self.model_tester.batch_size, 3, config.image_size, config.image_size])
        num_patches = self.model_tester.num_patches
        labels = ids_tensor([self.model_tester.batch_size, num_patches], 2).float()
        labels[0, 0] = -1
        boxes = torch.full((self.model_tester.batch_size, num_patches, 4), -1.0, device=torch_device)
        boxes[:, 0] = torch.tensor([0.0, 0.0, 4.0, 4.0], device=torch_device)
        result = model(input_ids, pixel_values=pixel_values, labels=labels, boxes=boxes)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, num_patches))
        self.assertEqual(result.pred_boxes.shape, (self.model_tester.batch_size, num_patches, 4))
        self.assertTrue(torch.isfinite(result.loss))
        # two images in the first sequence, none in the second: the counts match overall, not per sequence
        image = [config.boi_token_id] + [config.image_token_id] * num_patches + [config.eoi_token_id]
        two_images = [config.bos_token_id, config.segment_token_id, *image, *image, 20, config.eos_token_id]
        no_image = [config.bos_token_id, config.segment_token_id, 20, config.eos_token_id]
        no_image += [config.pad_token_id] * (len(two_images) - len(no_image))
        input_ids = torch.tensor([two_images, no_image], device=torch_device)
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            model(input_ids, attention_mask=(input_ids != config.pad_token_id).long(), pixel_values=pixel_values[:2])

    def test_modalities_validated(self):
        with self.assertRaises(StrictDataclassClassValidationError):
            OmniDebertaConfig(modalities=["image"])
        with self.assertRaises(StrictDataclassClassValidationError):
            OmniDebertaConfig(pos_att_type=["c2c"])

    def test_pos_att_type_from_string(self):
        self.assertEqual(OmniDebertaConfig(pos_att_type="p2c|c2p").pos_att_type, ["p2c", "c2p"])

    @unittest.skip(reason="the media placeholders and the [EMBED] pooling read input_ids")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="the media placeholders and the [EMBED] pooling read input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass


class OmniDebertaForCTCModelTester(OmniDebertaModelTester):
    """Text and audio only: `[CLS] [EMBED] [AUD] audio [/AUD] words [SEP]`."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.seq_length -= self.num_patches + 2

    def prepare_config_and_inputs(self):
        config, inputs = super().prepare_config_and_inputs()
        inputs["input_ids"] = self.build_input_ids(config, image=False)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        del inputs["pixel_values"]
        return config, inputs


@require_torch
class OmniDebertaForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (OmniDebertaForCTC,) if is_torch_available() else ()
    all_generative_model_classes = ()
    test_resize_embeddings = False
    test_all_params_have_gradient = False  # the text-only special-token rows of the audio expert see no tokens

    def setUp(self):
        self.model_tester = OmniDebertaForCTCModelTester(self)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = dict(inputs_dict)
        if return_labels:
            inputs_dict["labels"] = torch.tensor([[3, 4]] * self.model_tester.batch_size, device=torch_device)
        return inputs_dict

    @unittest.skip(reason="the media placeholders and the [EMBED] pooling read input_ids")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="the media placeholders and the [EMBED] pooling read input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass


class OmniDebertaForImageSegmentationModelTester(OmniDebertaModelTester):
    """Text and one image: `[CLS] [SEGMENT] [IMG] patches [/IMG] words [SEP]`."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.seq_length -= self.num_audio_tokens + 2

    def prepare_config_and_inputs(self):
        config, inputs = super().prepare_config_and_inputs()
        inputs["input_ids"] = self.build_input_ids(config, task_token_id=config.segment_token_id, audio=False)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        del inputs["input_features"], inputs["input_features_mask"]
        return config, inputs


@require_torch
class OmniDebertaForImageSegmentationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (OmniDebertaForImageSegmentation,) if is_torch_available() else ()
    all_generative_model_classes = ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = OmniDebertaForImageSegmentationModelTester(self)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(
        reason="the head reads one image per sequence; the common test feeds input_ids only and adds rows without "
        "images. Eager and SDPA are compared on images and audio in OmniDebertaModelTest.test_attention_options."
    )
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = dict(inputs_dict)
        if return_labels:
            shape = [self.model_tester.batch_size, self.model_tester.num_patches]
            inputs_dict["labels"] = (torch.arange(shape[1], device=torch_device) % 2).float().expand(shape)
            boxes = torch.full([*shape, 4], -1.0, device=torch_device)
            boxes[:, 0] = torch.tensor([0.0, 0.0, 4.0, 4.0], device=torch_device)
            inputs_dict["boxes"] = boxes
        return inputs_dict

    @unittest.skip(reason="the media placeholders and the [EMBED] pooling read input_ids")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="the media placeholders and the [EMBED] pooling read input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass
