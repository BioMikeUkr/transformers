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
"""Mask-predict text generation of OmniDeBERTa (image descriptions, speech transcripts)."""

import copy
import math
import weakref
from dataclasses import dataclass
from typing import Any

import torch

from ...generation import GenerationConfig
from ...modeling_outputs import ModelOutput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class OmniDebertaGenerationConfig(GenerationConfig):
    r"""
    Configuration of [`OmniDebertaGenerationMixin.generate`].

    Args:
        max_new_tokens (`int`, *optional*):
            Number of `[MASK]` slots put after the prompt. By default `tokens_per_second` x the clip length +
            `extra_tokens` for audio, `default_new_tokens` otherwise. The model ends the text with `[SEP]`, so slots
            too many cost nothing but time.
        tokens_per_second (`float`, *optional*, defaults to 5.0):
            Slots per second of audio when `max_new_tokens` is not set.
        extra_tokens (`int`, *optional*, defaults to 12):
            Slots added to those of the audio length when `max_new_tokens` is not set.
        seconds_per_frame (`float`, *optional*, defaults to 0.01):
            Duration of one mel frame of the feature extractor (its `hop_length` / `sampling_rate`), to measure clips.
        default_new_tokens (`int`, *optional*, defaults to 48):
            Slots of a sequence without audio when `max_new_tokens` is not set.
        tokens_per_step (`int`, *optional*, defaults to 1):
            Slots every forward pass fixes in every sequence: the most confident ones. Slots are never masked again.
        block_size (`int`, *optional*, defaults to 16):
            In sequences with audio (transcripts), only the leftmost `block_size` open slots compete
            (semi-autoregressive order); 0 for all of them.
        description_block_size (`int`, *optional*, defaults to 4):
            The same for sequences without audio (image descriptions), which read better with shorter blocks.
        repetition_penalty (`float`, *optional*, defaults to 10.0):
            A candidate among the `num_candidates` best loses `log(repetition_penalty)` for every fixed neighbor that is
            the same token (distance 1) or, with a `tokenizer` given to `generate`, the same word ignoring case
            (within `repetition_window`). 1.0 for none.
        repetition_window (`int`, *optional*, defaults to 6):
            Distance up to which fixed neighbors count for the word penalty.
        num_candidates (`int`, *optional*, defaults to 64):
            Candidates per slot the repetition penalty is applied to.
        drop_repeated_slots (`bool`, *optional*, defaults to `True`):
            Remove a slot instead of filling it when its best token repeats a fixed neighbor; a slot filled with
            `[BLANK]` is always removed.
        temperature (`float`, *optional*, defaults to 0.0):
            0 for the best token of every slot; above 0 the token is sampled among the `num_candidates` best, from
            softmax(penalized log-probability / temperature). Seed with `torch.manual_seed`.
        stop_at_eos (`bool`, *optional*, defaults to `True`):
            Whether a slot may take `[SEP]`; the text ends at the first one.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether to return an [`OmniDebertaGenerationOutput`] instead of the token ids.
    """

    def __init__(self, **kwargs):
        # The parent attributes (sampling, beams, caches) do not apply to mask-predict generation.
        self.max_new_tokens: int | None = kwargs.pop("max_new_tokens", None)
        self.tokens_per_second: float = kwargs.pop("tokens_per_second", 5.0)
        self.extra_tokens: int = kwargs.pop("extra_tokens", 12)
        self.seconds_per_frame: float = kwargs.pop("seconds_per_frame", 0.01)
        self.default_new_tokens: int = kwargs.pop("default_new_tokens", 48)
        self.tokens_per_step: int = kwargs.pop("tokens_per_step", 1)
        self.block_size: int = kwargs.pop("block_size", 16)
        self.description_block_size: int = kwargs.pop("description_block_size", 4)
        self.repetition_penalty: float = kwargs.pop("repetition_penalty", 10.0)
        self.repetition_window: int = kwargs.pop("repetition_window", 6)
        self.num_candidates: int = kwargs.pop("num_candidates", 64)
        self.drop_repeated_slots: bool = kwargs.pop("drop_repeated_slots", True)
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.stop_at_eos: bool = kwargs.pop("stop_at_eos", True)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.pad_token_id: int | None = kwargs.pop("pad_token_id", None)
        self.eos_token_id: int | None = kwargs.pop("eos_token_id", None)
        self._commit_hash: str | None = kwargs.pop("_commit_hash", None)
        self._from_model_config: bool | None = kwargs.pop("_from_model_config", None)
        self.transformers_version: str | None = kwargs.pop("transformers_version", None)
        if len(kwargs) > 0:
            raise ValueError(f"Unexpected kwargs: {list(kwargs)}")
        self.validate()

    def validate(self, **unused_kwargs):
        for name in ("tokens_per_step", "num_candidates", "repetition_window"):
            if getattr(self, name) < 1:
                raise ValueError(f"`{name}` must be a positive integer, got {getattr(self, name)}")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError(f"`max_new_tokens` must be a positive integer, got {self.max_new_tokens}")
        for name in ("block_size", "description_block_size"):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be >= 0, got {getattr(self, name)}")
        if self.temperature < 0:
            raise ValueError(f"`temperature` must be >= 0, got {self.temperature}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"`repetition_penalty` must be >= 1.0, got {self.repetition_penalty}")

    def get_generation_mode(self, *args, **kwargs):
        raise NotImplementedError("OmniDebertaGenerationConfig does not support `get_generation_mode`")

    def from_model_config(self, *args, **kwargs):
        raise NotImplementedError("OmniDebertaGenerationConfig does not support `from_model_config`")


@auto_docstring(
    custom_intro="""
    Output of [`OmniDebertaGenerationMixin.generate`].
    """
)
@dataclass
class OmniDebertaGenerationOutput(ModelOutput):
    r"""
    sequences (`torch.LongTensor` of shape `(batch_size, generated_length)`):
        The generated tokens of every sequence, without the prompt and the final `[SEP]`, padded with
        `config.pad_token_id`.
    scores (`torch.FloatTensor` of shape `(batch_size, generated_length)`):
        Probability of every generated token when it was fixed (after the repetition penalty); 0 on padding.
    num_forward_passes (`int`):
        Forward passes of the model.
    """

    sequences: torch.LongTensor | None = None
    scores: torch.FloatTensor | None = None
    num_forward_passes: int | None = None


_WORD_KEYS = weakref.WeakKeyDictionary()


def word_keys(tokenizer, vocab_size: int) -> torch.LongTensor:
    """Per token id: an id of its word ignoring case and the leading space; -1 for tokens with fewer than two letters
    or digits (punctuation and short pieces may repeat). Computed once per tokenizer."""
    if tokenizer in _WORD_KEYS and len(_WORD_KEYS[tokenizer]) == vocab_size:
        return _WORD_KEYS[tokenizer]
    pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    index, keys = {}, []
    for piece in pieces:
        word = (piece or "").lstrip("▁").lower()
        keys.append(index.setdefault(word, len(index)) if sum(c.isalnum() for c in word) >= 2 else -1)
    _WORD_KEYS[tokenizer] = torch.tensor(keys)
    return _WORD_KEYS[tokenizer]


class OmniDebertaGenerationMixin:
    """
    Mask-predict generation: `[MASK]` slots after the prompt; every forward pass fixes the most confident slots
    (left to right in blocks) until the text ends with `[SEP]`. Slots too many are removed (`[BLANK]`, repeats).
    The tokens from `config.mask_token_id` on are the special tokens and are never generated, except `[BLANK]`.
    """

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        generation_config: OmniDebertaGenerationConfig | None = None,
        tokenizer=None,
        **kwargs: Any,
    ) -> torch.LongTensor | OmniDebertaGenerationOutput:
        r"""
        Generate the text that follows the prompt of every sequence, e.g. the description of an image
        (`processor(images=image, task="describe")`) or the transcript of a clip (`task="transcribe"`).

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Prompts made by [`OmniDebertaProcessor`], ending with `[SEP]`; the slots go before it.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                1 for the tokens of the prompts, 0 for padding.
            pixel_values, input_features, input_features_mask (*optional*):
                The media of the prompts, as made by the processor.
            generation_config ([`OmniDebertaGenerationConfig`], *optional*):
                Overrides `model.generation_config`; keyword arguments that match its attributes override both.
            tokenizer (*optional*):
                The tokenizer of the model; with it the repetition penalty also compares words, not only token ids.

        Returns:
            `torch.LongTensor` of shape `(batch_size, generated_length)`, or an [`OmniDebertaGenerationOutput`]
            with `return_dict_in_generate=True`. Decode with `processor.batch_decode(..., skip_special_tokens=True)`.
        """
        generation_config = copy.deepcopy(generation_config or self.generation_config)
        unused = generation_config.update(**kwargs)
        if unused:
            raise ValueError(f"Unexpected arguments: {list(unused)}")
        generation_config.validate()
        config = self.config
        device = input_ids.device
        mask_id, blank_id, eos_id, pad_id = (
            config.mask_token_id,
            config.blank_token_id,
            config.eos_token_id,
            config.pad_token_id,
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if input_features is not None and input_features_mask is None:
            input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.bool, device=device)

        # per sequence: its tokens, its media and where its slots start
        rows, images, clips = [], [], []
        image_index = clip_index = 0
        patches_per_image = (config.image_size // config.patch_size) ** 2
        for ids in (row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask)):
            num_images = ids.count(config.image_token_id) // patches_per_image
            num_clips = ids.count(config.boa_token_id)
            images.append(list(range(image_index, image_index + num_images)))
            clips.append(list(range(clip_index, clip_index + num_clips)))
            image_index, clip_index = image_index + num_images, clip_index + num_clips
            if ids[-1] != eos_id:
                raise ValueError("Every prompt must end with [SEP] (made by the processor with its defaults).")
            length = generation_config.max_new_tokens
            if length is None and num_clips:
                frames = input_features_mask[clips[-1]].sum().item() if input_features_mask is not None else 0
                seconds = frames * generation_config.seconds_per_frame
                length = math.ceil(generation_config.tokens_per_second * seconds) + generation_config.extra_tokens
            elif length is None:
                length = generation_config.default_new_tokens
            if generation_config.max_new_tokens is None:
                length = max(4, length)
            length = max(0, min(length, config.max_position_embeddings - len(ids)))
            start = len(ids) - 1
            rows.append(
                {"ids": ids[:-1] + [mask_id] * length + [eos_id], "start": start, "probs": [0.0] * length,
                 "audio": num_clips > 0}
            )  # fmt: skip

        keys = None
        if tokenizer is not None and generation_config.repetition_penalty > 1:
            keys = word_keys(tokenizer, config.vocab_size)
            keys = (keys.to(device), keys.tolist())  # on the device for the penalty, on the host for the repeats
        banned = torch.zeros(config.vocab_size, dtype=torch.bool, device=device)
        banned[[pad_id, config.bos_token_id]] = True
        if not generation_config.stop_at_eos:
            banned[eos_id] = True
        banned[mask_id:] = True
        banned[blank_id] = False  # a slot too many: removed below

        live = [index for index, row in enumerate(rows) if mask_id in row["ids"][row["start"] :]]
        num_passes = 0
        while live:
            batch = self._slot_batch([rows[i] for i in live], pad_id, device)
            batch_images = [image for i in live for image in images[i]]
            batch_clips = [clip for i in live for clip in clips[i]]
            hidden = self.model(
                **batch,
                pixel_values=pixel_values[batch_images] if batch_images else None,
                input_features=input_features[batch_clips] if batch_clips else None,
                input_features_mask=input_features_mask[batch_clips] if batch_clips else None,
            ).last_hidden_state
            num_passes += 1
            still_live = []
            for position, index in enumerate(live):
                row = rows[index]
                start, end = row["start"], len(row["ids"]) - 1
                logits = self.lm_head(hidden[position, start:end]).float().to(device)
                logits[:, banned] = torch.finfo(logits.dtype).min
                if not clips[index]:  # [BLANK] is trained on transcripts only: in a description it cuts words out
                    logits[:, blank_id] = torch.finfo(logits.dtype).min
                self._step(row, logits, keys, generation_config, mask_id, blank_id, eos_id)
                text = row["ids"][row["start"] : -1]
                if mask_id not in text:
                    continue
                if generation_config.stop_at_eos and eos_id in text and mask_id not in text[: text.index(eos_id)]:
                    continue
                still_live.append(index)
            live = still_live

        texts, scores = [], []
        for row in rows:
            text, probs = row["ids"][row["start"] : -1], row["probs"]
            end = text.index(eos_id) if generation_config.stop_at_eos and eos_id in text else len(text)
            texts.append(text[:end])
            scores.append(probs[:end])
        width = max([len(text) for text in texts] + [0])
        sequences = torch.full((len(texts), width), pad_id, dtype=torch.long, device=device)
        token_scores = torch.zeros(len(texts), width, device=device)
        for i, (text, probs) in enumerate(zip(texts, scores)):
            sequences[i, : len(text)] = torch.tensor(text, dtype=torch.long)
            token_scores[i, : len(probs)] = torch.tensor(probs)
        if generation_config.return_dict_in_generate:
            return OmniDebertaGenerationOutput(sequences=sequences, scores=token_scores, num_forward_passes=num_passes)
        return sequences

    def adjust_generation_fn(
        self,
        generation_config,
        from_auto_class,
        from_pipeline,
        pretrained_model_name_or_path,
        cache_dir,
        force_download,
        proxies,
        local_files_only,
        token,
        revision,
        subfolder,
        trust_remote_code,
        **kwargs,
    ):
        """Called by `from_pretrained`: load the checkpoint's `generation_config.json` (as in DiffusionGemma)."""
        del trust_remote_code  # unused
        if generation_config is not None:
            self.generation_config = self.generation_config_class.from_dict(generation_config.to_dict())
        elif pretrained_model_name_or_path is not None:
            try:
                self.generation_config = self.generation_config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    **kwargs,
                )
            except OSError:
                logger.info("Generation config file not found, using the default generation config.")

    @staticmethod
    def _slot_batch(rows: list[dict], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
        width = max(len(row["ids"]) for row in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        for i, row in enumerate(rows):
            input_ids[i, : len(row["ids"])] = torch.tensor(row["ids"], device=device)
            attention_mask[i, : len(row["ids"])] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def _step(row: dict, logits: torch.Tensor, keys, generation_config, mask_id: int, blank_id: int, eos_id: int):
        """One pass for one sequence: fix (or remove) its most confident open slots."""
        ids, start = row["ids"], row["start"]
        text = ids[start:-1]
        if generation_config.stop_at_eos:
            # one end: [SEP] only after the last fixed word, nothing but [SEP] after the first fixed [SEP]
            words = [i for i, token in enumerate(text) if token not in (mask_id, eos_id)]
            if words:
                logits[: words[-1] + 1, eos_id] = torch.finfo(logits.dtype).min
            if eos_id in text:
                logits[text.index(eos_id) + 1 :, eos_id + 1 :] = torch.finfo(logits.dtype).min
        raw = logits.argmax(-1).tolist()
        best, best_prob = OmniDebertaGenerationMixin._choose(
            ids, start, logits, keys[0] if keys else None, generation_config, mask_id, eos_id
        )
        open_slots = [i for i, token in enumerate(text) if token == mask_id]
        block_size = generation_config.block_size if row["audio"] else generation_config.description_block_size
        if block_size:
            open_slots = open_slots[:block_size]
        chosen = sorted(open_slots, key=lambda i: -best_prob[i])[: generation_config.tokens_per_step]
        for i in sorted(chosen, reverse=True):
            repeated = generation_config.drop_repeated_slots and OmniDebertaGenerationMixin._repeats(
                ids, start + i, raw[i], keys[1] if keys else None, mask_id, eos_id
            )
            if best[i] == blank_id or repeated:
                del ids[start + i], row["probs"][i]  # a slot too many: remove it
            else:
                ids[start + i], row["probs"][i] = best[i], best_prob[i]

    @staticmethod
    def _choose(ids, start, logits, keys, generation_config, mask_id, eos_id):
        """Best token per slot and its probability, after the repetition penalty."""
        log_probs = torch.log_softmax(logits, -1)
        penalty, temperature = generation_config.repetition_penalty, generation_config.temperature
        if penalty <= 1 and temperature <= 0:
            best = log_probs.max(-1)
            return best.indices.tolist(), best.values.exp().tolist()
        values, tokens = log_probs.topk(generation_config.num_candidates, -1)
        if penalty <= 1:
            return OmniDebertaGenerationMixin._pick(values, tokens, temperature)
        full = torch.tensor(ids, device=logits.device)
        slots = torch.arange(logits.shape[0], device=logits.device) + start
        window = generation_config.repetition_window if keys is not None else 1
        for distance in [d for d in range(-window, window + 1) if d]:
            neighbor = slots + distance
            token = full[neighbor.clamp(0, len(ids) - 1)]
            fixed = (neighbor >= 0) & (neighbor < len(ids)) & (token != mask_id) & (token > eos_id)
            same = torch.zeros_like(tokens, dtype=torch.bool)
            if keys is not None:
                key = keys[token][:, None]
                same = (key >= 0) & (keys[tokens] == key)
            if abs(distance) == 1:
                same |= tokens == token[:, None]
            values = values - math.log(penalty) * (same & fixed[:, None])
        return OmniDebertaGenerationMixin._pick(values, tokens, temperature)

    @staticmethod
    def _pick(values: torch.Tensor, tokens: torch.Tensor, temperature: float) -> tuple[list[int], list[float]]:
        """The best candidate of every slot, or one sampled with `temperature`; with its probability."""
        if temperature > 0:
            index = torch.multinomial(torch.softmax(values / temperature, -1), 1)[:, 0]
        else:
            index = values.argmax(-1)
        token = tokens.gather(1, index[:, None])[:, 0]
        prob = values.gather(1, index[:, None])[:, 0].exp()
        return token.tolist(), prob.tolist()

    @staticmethod
    def _repeats(ids, position, token, keys, mask_id, eos_id) -> bool:
        """Whether `token` at `position` repeats a fixed neighbor (same id, or same word ignoring case)."""
        for neighbor in (position - 1, position + 1):
            if 0 <= neighbor < len(ids) and ids[neighbor] not in (mask_id, eos_id):
                if ids[neighbor] == token:
                    return True
                if keys is not None and keys[ids[neighbor]] >= 0 and keys[ids[neighbor]] == keys[token]:
                    return True
        return False


__all__ = ["OmniDebertaGenerationConfig", "OmniDebertaGenerationMixin", "OmniDebertaGenerationOutput"]
