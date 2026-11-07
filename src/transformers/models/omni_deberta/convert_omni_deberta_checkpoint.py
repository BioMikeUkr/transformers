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
"""
Convert an OmniDeBERTa training checkpoint (`OmniDebertaForPreTraining` of the training code, or the discriminator of
its ELECTRA wrapper) to the transformers format.

    python src/transformers/models/omni_deberta/convert_omni_deberta_checkpoint.py --input_dir SOURCE --output_dir OUTPUT

The output holds the weights of every head of the source (word prediction, referring segmentation, replaced token
detection); every model class loads the ones it uses. The processor is saved next to them. A model with the word
prediction head also gets its generation config, and the files that make it a SentenceTransformers model
(`modules.json`, `sentence_bert_config.json`, `config_sentence_transformers.json`): its embedding is `pooler_output`
for every modality. With `--encoder_only` (an embedding model) the task heads are left out and the output is an
`OmniDebertaModel` with the SentenceTransformers files.
"""

import argparse
import json
import os
import re
import shutil
import tempfile

from safetensors.torch import load_file, save_file

from transformers import (
    AutoTokenizer,
    OmniDebertaConfig,
    OmniDebertaForImageSegmentation,
    OmniDebertaForMaskedLM,
    OmniDebertaForReplacedTokenDetection,
    OmniDebertaGenerationConfig,
    OmniDebertaModel,
)
from transformers.models.omni_deberta.feature_extraction_omni_deberta import OmniDebertaFeatureExtractor
from transformers.models.omni_deberta.image_processing_omni_deberta import OmniDebertaImageProcessor
from transformers.models.omni_deberta.processing_omni_deberta import OmniDebertaProcessor


# training-only config fields
DROPPED_CONFIG = {
    "bf16_residual", "pruned_from_layers", "generator_config", "architectures", "auto_map", "transformers_version",
    "model_type", "torch_dtype", "dtype", "_name_or_path", "code_layer", "image_codebook_size", "audio_codebook_size",
    "task_markers", "cls_token_id", "sep_token_id",
}  # fmt: skip
RENAMED_CONFIG = {
    "image_token_id": "boi_token_id",
    "image_end_token_id": "eoi_token_id",
    "audio_token_id": "boa_token_id",
    "audio_end_token_id": "eoa_token_id",
}
# pre-training-only weights: code heads, mask vectors of hidden patches / frames, FSDP padding; the tied decoder
DROPPED_WEIGHTS = re.compile(
    r"^(image_code_head|audio_code_head)\.|_mask_token$|^_fsdp_align_pad$|^lm_head\.decoder\."
)
RENAMED_WEIGHTS = [
    (r"^deberta\.segment_(query|key|box)\.", r"segment_\1."),
    (r"^deberta\.", "model."),
    (r"^model\.embeddings\.(patch_proj|row_embeddings|column_embeddings)\.", r"model.embeddings.image_stem.\1."),
    (r"^model\.embeddings\.image_layer_norm\.", "model.embeddings.image_stem.layer_norm."),
    (r"^model\.embeddings\.audio_layer_norm\.", "model.embeddings.audio_stem.layer_norm."),
    (r"^(segment_\w+)\.0\.", r"\1.layer_norm."),
    (r"^(segment_\w+)\.1\.", r"\1.fc1."),
    (r"^(segment_\w+)\.3\.", r"\1.fc2."),
]
# tokens after [MASK] (128000) in id order; the training tokenizer has no names for 128001-128010 and no 128011-128012
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
FIRST_SPECIAL_ID = 128001

# The prompts the model was trained with (those of its teacher, EmbeddingGemma 2), by task; the keys are task names
# understood by the MTEB harness.
TASK_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "BitextMining": "task: search result | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
}
# Renders a conversation (SentenceTransformers' "message" inputs: several modalities in one embedding) as text with
# media placeholders, in content order. The system text (the task prompt) is kept only when the conversation has text:
# images and clips were embedded without prompts in training, as by the teacher.
CHAT_TEMPLATE = (
    "{%- set ns = namespace(has_text=false) -%}"
    "{%- for msg in messages if msg.get('role') != 'system' -%}"
    "{%- if msg.get('content') is string -%}{%- if msg['content'] -%}{%- set ns.has_text = true -%}{%- endif -%}"
    "{%- else -%}{%- for item in msg['content'] if item.get('type') == 'text' and item.get('text') -%}"
    "{%- set ns.has_text = true -%}{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{%- if ns.has_text -%}{%- for msg in messages if msg.get('role') == 'system' -%}"
    "{%- if msg.get('content') is string -%}{{ msg['content'] }}"
    "{%- else -%}{%- for item in msg['content'] if item.get('type') == 'text' -%}{{ item['text'] }}{%- endfor -%}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{%- for msg in messages if msg.get('role') != 'system' -%}"
    "{%- if msg.get('content') is string -%}{{ msg['content'] }}"
    "{%- else -%}"
    "{%- set existing_text = msg['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | join -%}"
    "{%- set has_manual_placeholders = ('[IMG_PATCH]' in existing_text) or ('[AUD_FRAME]' in existing_text) -%}"
    "{%- for item in msg['content'] -%}"
    "{%- if item.get('type') == 'text' -%}{{ item['text'] }}"
    "{%- elif not has_manual_placeholders and item.get('type') == 'image' -%}[IMG_PATCH]"
    "{%- elif not has_manual_placeholders and item.get('type') == 'audio' -%}[AUD_FRAME]"
    "{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}"
)
ST_REQUIREMENTS = {
    "sentence-transformers": {
        "specifier": ">=6.1.0",
        "reason": "Older versions ignore the order of multimodal dict inputs and cannot put several images or texts "
        "into a single embedding.",
    },
}


def convert_config(input_dir: str) -> OmniDebertaConfig:
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as file:
        old = json.load(file)
    fields = {RENAMED_CONFIG.get(key, key): value for key, value in old.items() if key not in DROPPED_CONFIG}
    config = OmniDebertaConfig(**fields)
    config.architectures = ["OmniDebertaForMaskedLM"]
    return config


def convert_state_dict(state: dict) -> dict:
    if any(key.startswith("generator.") for key in state):  # ELECTRA wrapper: keep the discriminator
        state = {key: value for key, value in state.items() if not key.startswith("generator.")}
    converted = {}
    for key, value in state.items():
        if DROPPED_WEIGHTS.search(key):
            continue
        for pattern, replacement in RENAMED_WEIGHTS:
            key = re.sub(pattern, replacement, key)
        converted[key] = value.contiguous()
    return converted


def convert_tokenizer(input_dir: str):
    # call defaults of the processor travel with the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        input_dir,
        extra_special_tokens=SPECIAL_TOKENS,
        padding=True,
        model_input_names=["input_ids", "attention_mask"],
    )
    for offset, token in enumerate(SPECIAL_TOKENS.values()):
        if tokenizer.convert_tokens_to_ids(token) != FIRST_SPECIAL_ID + offset:
            raise ValueError(f"{token} did not get the id {FIRST_SPECIAL_ID + offset}")
    return tokenizer


def check_weights(output_dir: str, state: dict):
    """Every converted weight is used by a head class and none is missing: catches a wrong rename."""
    heads = [OmniDebertaForMaskedLM] if any(key.startswith("lm_head.") for key in state) else []
    if any(key.startswith("segment_") for key in state):
        heads.append(OmniDebertaForImageSegmentation)
    if any(key.startswith("rtd_head.") for key in state):
        heads.append(OmniDebertaForReplacedTokenDetection)
    pooler = any(key.startswith("model.pooler.") for key in state)
    base, info = OmniDebertaModel.from_pretrained(output_dir, add_pooling_layer=pooler, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"OmniDebertaModel: missing {info['missing_keys']}, mismatched {info['mismatched_keys']}")
    used = {f"model.{key}" for key in base.state_dict()}
    for head in heads:
        model, info = head.from_pretrained(output_dir, output_loading_info=True)
        if info["missing_keys"] or info["mismatched_keys"]:
            raise ValueError(f"{head.__name__}: missing {info['missing_keys']}, mismatched {info['mismatched_keys']}")
        used |= set(model.state_dict())
    unused = set(state) - used
    if unused:
        raise ValueError(f"Converted weights no model class uses: {sorted(unused)[:10]}")


def convert(input_dir: str, output_dir: str, encoder_only: bool = False, tokenizer_dir: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    config = convert_config(input_dir)
    state = convert_state_dict(load_file(os.path.join(input_dir, "model.safetensors")))
    if encoder_only:
        state = {key: value for key, value in state.items() if key.startswith("model.")}
        config.architectures = ["OmniDebertaModel"]
    elif "rtd_head.classifier.weight" in state:
        # the discriminator's embedding projection is the start point's, untrained for its encoder: left out
        state = {key: value for key, value in state.items() if not key.startswith("model.pooler.")}
        config.architectures = ["OmniDebertaForReplacedTokenDetection"]
    has_lm_head = any(key.startswith("lm_head.") for key in state)
    config.save_pretrained(output_dir)
    save_file(state, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})
    if has_lm_head:
        OmniDebertaGenerationConfig().save_pretrained(output_dir)
    check_weights(output_dir, state)

    processor = OmniDebertaProcessor(
        image_processor=OmniDebertaImageProcessor(size={"height": config.image_size, "width": config.image_size}),
        feature_extractor=OmniDebertaFeatureExtractor(feature_size=config.num_mel_bins),
        tokenizer=convert_tokenizer(tokenizer_dir or input_dir),
        patch_size=config.patch_size,
        max_length=config.max_position_embeddings,
        chat_template=CHAT_TEMPLATE,
    )
    processor.save_pretrained(output_dir)
    if has_lm_head or encoder_only:  # the discriminator's embedding was not trained
        save_sentence_transformer(output_dir)


def save_sentence_transformer(output_dir: str):
    """Make `output_dir` a SentenceTransformers model whose embedding is `pooler_output` (the normalized projection of
    the `[EMBED]` state) for text, images and audio: one `Transformer` module, no pooling or normalization module.
    SentenceTransformers saves into a scratch directory: it would rewrite the weights without the task heads."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules.transformer import Transformer

    transformer = Transformer(
        output_dir,
        modality_config={
            **{
                modality: {"method": "forward", "method_output_name": "pooler_output"}
                for modality in ("text", "image", "audio")
            },
            "message": {"method": "forward", "method_output_name": "pooler_output", "format": "structured"},
        },
        module_output_name="sentence_embedding",
    )
    model = SentenceTransformer(modules=[transformer], prompts=TASK_PROMPTS, similarity_fn_name="cosine")
    with tempfile.TemporaryDirectory() as scratch:
        model.save_pretrained(scratch)
        for name in ("modules.json", "sentence_bert_config.json", "config_sentence_transformers.json"):
            shutil.copy(os.path.join(scratch, name), os.path.join(output_dir, name))
    # `requirements` is read but not written by SentenceTransformers
    path = os.path.join(output_dir, "config_sentence_transformers.json")
    with open(path, encoding="utf-8") as file:
        st_config = json.load(file)
    st_config["requirements"] = ST_REQUIREMENTS
    with open(path, "w", encoding="utf-8") as file:
        json.dump(st_config, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Directory of the training checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory to write the converted model to.")
    parser.add_argument(
        "--encoder_only", action="store_true", help="Keep only the encoder (an embedding model), no task heads."
    )
    parser.add_argument(
        "--tokenizer_dir", help="Directory of the tokenizer, when the checkpoint has none (default: `--input_dir`)."
    )
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, encoder_only=args.encoder_only, tokenizer_dir=args.tokenizer_dir)
