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
"""Feature extractor class for OmniDeBERTa."""

import warnings

import numpy as np

from ...audio_utils import mel_filter_bank, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available
from ...utils.import_utils import requires


if is_torch_available():
    import torch


@requires(backends=("torch",))
class OmniDebertaFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an OmniDeBERTa feature extractor: log-mel spectrograms of mono waveforms (Hann window, HTK mel scale,
    no normalization), one frame every `hop_length` samples, the first frame centered on the first sample.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            Number of mel bins.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate of the waveforms, in hertz.
        frame_length (`int`, *optional*, defaults to 320):
            Window length in samples (20 ms).
        hop_length (`int`, *optional*, defaults to 160):
            Step between two frames in samples (10 ms).
        fft_length (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        max_frequency (`float`, *optional*, defaults to 8000.0):
            Top of the mel scale, in hertz.
        mel_floor (`float`, *optional*, defaults to 0.001):
            Added to the mel energies before the logarithm.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value the features of padded frames take.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        frame_length=320,
        hop_length=160,
        fft_length=512,
        max_frequency=8000.0,
        mel_floor=1e-3,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fft_length = fft_length
        self.max_frequency = max_frequency
        self.mel_floor = mel_floor
        self.window = window_function(frame_length, "hann", periodic=True).astype(np.float32)
        with warnings.catch_warnings():  # the top mel filter falls between two FFT bins, as in training
            warnings.simplefilter("ignore")
            self.mel_filters = mel_filter_bank(
                num_frequency_bins=fft_length // 2 + 1,
                num_mel_filters=feature_size,
                min_frequency=0.0,
                max_frequency=max_frequency,
                sampling_rate=sampling_rate,
                norm=None,
                mel_scale="htk",
            ).astype(np.float32)

    def to_dict(self):
        output = super().to_dict()
        output.pop("window", None)
        output.pop("mel_filters", None)
        return output

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Turn one waveform or a list of them (float in [-1, 1], mono) into `input_features` of shape
        `(clips, frames, feature_size)`, padded to the longest clip, and `input_features_mask` (True for the frames
        whose window lies on real audio).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]` or `list[list[float]]`):
                The waveform or waveforms.
            sampling_rate (`int`, *optional*):
                Sampling rate of the waveforms; must be `self.sampling_rate`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to return.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The waveforms must be sampled at {self.sampling_rate} Hz, got {sampling_rate}. Resample them first."
            )
        if isinstance(raw_speech, (list, tuple)) and len(raw_speech) == 0:
            raise ValueError("Got no waveform.")
        is_batched = isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        is_batched = is_batched or (isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1)
        if not is_batched:
            raw_speech = [raw_speech]
        waveforms = [torch.as_tensor(np.asarray(wave, dtype=np.float32)).reshape(-1) for wave in raw_speech]
        lengths = torch.tensor([len(wave) for wave in waveforms])
        batch = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        # at least one window: clips shorter than a frame give one frame, masked as padding
        batch = torch.nn.functional.pad(
            batch, (0, max(0, self.frame_length + 1 - self.frame_length // 2 - batch.shape[1]))
        )

        pad = self.frame_length // 2
        frames = torch.nn.functional.pad(batch, (pad, 0)).unfold(-1, self.frame_length + 1, self.hop_length)
        frames = frames[..., :-1] * torch.from_numpy(self.window)
        magnitude = torch.fft.rfft(frames, n=self.fft_length).abs()
        features = torch.log(magnitude @ torch.from_numpy(self.mel_filters) + self.mel_floor)
        # a frame is real when the last sample of its window is real audio
        window_end = torch.arange(features.shape[1]) * self.hop_length + self.frame_length
        mask = window_end[None, :] < (lengths[:, None] + pad)
        features = features.masked_fill(~mask[..., None], self.padding_value)
        return BatchFeature(
            {"input_features": features.numpy(), "input_features_mask": mask.numpy()}, tensor_type=return_tensors
        )


__all__ = ["OmniDebertaFeatureExtractor"]
