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
"""Processor class for OmniDeBERTa."""

import numpy as np

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring


class OmniDebertaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class OmniDebertaProcessor(ProcessorMixin):
    valid_processor_kwargs = OmniDebertaProcessorKwargs
    task_tokens = {"segment": "segment_token", "describe": "describe_token", "transcribe": "transcribe_token"}

    def __init__(
        self,
        image_processor,
        feature_extractor,
        tokenizer,
        chat_template=None,
        patch_size: int = 16,
        max_length: int = 8192,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 16):
            Side of an image patch in pixels; an image takes `(height / patch_size) * (width / patch_size)` positions.
        max_length (`int`, *optional*, defaults to 8192):
            Longest sequence of the model (`config.max_position_embeddings`). A clip that does not fit loses its end.
        """
        self.patch_size = patch_size
        self.max_length = max_length
        for name in ("image", "boi", "eoi", "audio", "boa", "eoa", "embed", *self.task_tokens):
            token = getattr(tokenizer, f"{name}_token")
            setattr(self, f"{name}_token", token)
            setattr(self, f"{name}_token_id", tokenizer.convert_tokens_to_ids(token))
        super().__init__(image_processor, feature_extractor, tokenizer, chat_template=chat_template, **kwargs)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | list[TextInput] | None = None,
        audio: AudioInput | list[AudioInput] | None = None,
        task: str | None = None,
        **kwargs: Unpack[OmniDebertaProcessorKwargs],
    ) -> BatchFeature:
        r"""
        images (`ImageInput` or `list[ImageInput]`, *optional*):
            One image per sample, a list of images per sample, or (one sample) a list of images.
        audio (`AudioInput` or `list[AudioInput]`, *optional*):
            Waveforms at the feature extractor's sampling rate, laid out like `images`.
        task (`str`, *optional*):
            `"describe"`, `"transcribe"` or `"segment"`: the sequences start with that task marker. Without a task
            they start with `[EMBED]`, whose state is the sequence embedding.

        A sample can hold any text, images and clips. `processor.image_token` / `processor.audio_token` in the text
        put the next image / clip at that place; a sample without them gets its images, then its clips, before its
        text. Clips take the room the rest of their sequence leaves within `max_length`: a long clip loses its end.
        """
        if images is None and text is None and audio is None:
            raise ValueError("You need to provide at least one of `text`, `images` or `audio`.")
        if task is not None and task not in self.task_tokens:
            raise ValueError(f"Unknown task {task!r}, expected one of {list(self.task_tokens)}.")
        prefix = getattr(self, self.task_tokens[task]) if task is not None else self.embed_token
        texts = [text] if isinstance(text, str) else (list(text) if text is not None else None)

        if texts is not None and any(self.image_token in t or self.audio_token in t for t in texts):
            # the placeholders say where the media go; the base processor checks their counts
            image_lists, audio_lists = self._as_batch(images, 1), self._as_batch(audio, 1)
        else:
            if texts is not None:
                num_samples = len(texts)
            else:  # media only: one sample per item
                media = images if images is not None else audio
                num_samples = len(media) if isinstance(media, (list, tuple)) and not _is_waveform(media) else 1
            image_lists, audio_lists = self._as_batch(images, num_samples), self._as_batch(audio, num_samples)
            texts = [
                self.image_token * len(sample_images) + self.audio_token * len(sample_audio) + sample
                for sample, sample_images, sample_audio in zip(texts or [""] * num_samples, image_lists, audio_lists)
            ]
        flat_images = [image for sample_images in image_lists for image in sample_images] or None
        flat_audio = [
            np.asarray(clip, dtype=np.float32) if isinstance(clip, (list, tuple)) else clip
            for sample_audio in audio_lists
            for clip in sample_audio
        ] or None
        texts = [prefix + sample for sample in texts]
        if flat_audio is not None:
            flat_audio = self._fit_audio(texts, flat_audio)
        return super().__call__(images=flat_images, text=texts, audio=flat_audio, **kwargs)

    def _fit_audio(self, texts: list[str], clips: list) -> list:
        """Cut the clips of every sequence (in reading order) to the room its other tokens leave in `max_length`."""
        if sum(text.count(self.audio_token) for text in texts) != len(clips):
            return clips  # the base processor reports the mismatch
        size = self.image_processor.size
        image_positions = (size["height"] // self.patch_size) * (size["width"] // self.patch_size) + 2
        fitted, clip_index = [], 0
        for text in texts:
            num_clips = text.count(self.audio_token)
            words = text.replace(self.image_token, " ").replace(self.audio_token, " ")
            used = 2 + len(self.tokenizer(words, add_special_tokens=False)["input_ids"])  # [CLS] and [SEP]
            room = self.max_length - used - text.count(self.image_token) * image_positions - 2 * num_clips
            for clip in clips[clip_index : clip_index + num_clips]:
                waveform = clip["array"] if isinstance(clip, dict) else clip
                if self._num_audio_tokens(self._num_frames(len(waveform))) > max(room, 1):
                    waveform = waveform[: self._samples_for_audio_tokens(max(room, 1))]
                room -= self._num_audio_tokens(self._num_frames(len(waveform)))
                fitted.append(waveform)
            clip_index += num_clips
        return fitted

    def _num_frames(self, num_samples: int) -> int:
        """Mel frames of a waveform of `num_samples` samples whose window lies on real audio."""
        extractor = self.feature_extractor
        return max(
            0, (num_samples + extractor.frame_length // 2 - extractor.frame_length - 1) // extractor.hop_length + 1
        )

    def _samples_for_audio_tokens(self, num_tokens: int) -> int:
        """Fewest samples that make `num_tokens` audio positions."""
        extractor = self.feature_extractor
        frames = 4 * (num_tokens - 1) + 1
        return (frames - 1) * extractor.hop_length + extractor.frame_length + 1 - extractor.frame_length // 2

    @staticmethod
    def _as_batch(values, num_samples: int) -> list[list]:
        """None, one item, one item per sample or a list per sample -> one list per sample."""
        if values is None:
            return [[] for _ in range(num_samples)]
        if not isinstance(values, (list, tuple)) or _is_waveform(values):
            values = [values]
        values = list(values)
        if len(values) == num_samples:
            return [list(v) if isinstance(v, (list, tuple)) and not _is_waveform(v) else [v] for v in values]
        if num_samples == 1:
            return [values]
        raise ValueError(f"Got {len(values)} media items for {num_samples} samples.")

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        height, width = image_inputs["pixel_values"][image_idx].shape[-2:]
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        return f"{self.boi_token}{self.image_token * num_patches}{self.eoi_token}"

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_tokens = self._num_audio_tokens(int(audio_inputs["input_features_mask"][audio_idx].sum()))
        return f"{self.boa_token}{self.audio_token * num_tokens}{self.eoa_token}"

    @staticmethod
    def _num_audio_tokens(num_frames: int) -> int:
        """Audio positions made from `num_frames` mel frames by the two stride-2 convolutions of the audio stem."""
        return ((num_frames - 1) // 2) // 2 + 1 if num_frames > 0 else 0

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        """
        Number of placeholder tokens of images and clips of the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                `(height, width)` of every image.
            audio_lengths (`list[int]`, *optional*):
                Samples of every clip.
        """
        data = {}
        if image_sizes is not None:
            size = self.image_processor.size
            num_patches = (size["height"] // self.patch_size) * (size["width"] // self.patch_size)
            data["num_image_tokens"] = [num_patches] * len(image_sizes)
            data["num_image_patches"] = [1] * len(image_sizes)
        if audio_lengths is not None:
            data["num_audio_tokens"] = [self._num_audio_tokens(self._num_frames(n)) for n in audio_lengths]
        return MultiModalData(**data)


def _is_waveform(value) -> bool:
    return len(value) > 0 and isinstance(value[0], (int, float, np.number))


__all__ = ["OmniDebertaProcessor"]
