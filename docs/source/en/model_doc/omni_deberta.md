<!--Copyright 2026 Knowledgator and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was contributed to Hugging Face Transformers on 2026-11-06.*

# OmniDeBERTa

OmniDeBERTa is a family of multimodal encoders from Knowledgator. One [DeBERTa-v2](./deberta-v2) encoder with
disentangled relative attention reads text, images and audio in a single sequence:
`[CLS] [EMBED] [IMG] patches [/IMG] [AUD] audio [/AUD] text [SEP]`. Images are cut into 16x16 patches (one position
each, with 2-D relative positions in some layers), audio goes through a small convolutional stem (one position per
40 ms of log-mel frames). Every layer has one feed-forward expert per modality, chosen per token by a fixed rule, so
`OmniDeberta-300M-A150M` has 300M parameters of which 150M are active for a token.

The same model gives:

- a sequence embedding for text, images, audio or any mix of them (`pooler_output`: the normalized projection of the
  `[EMBED]` state), usable for retrieval across modalities;
- masked word prediction conditioned on all modalities, and text generation by mask-predict: image descriptions
  (`task="describe"`) and speech transcripts (`task="transcribe"`);
- task heads for classification, token classification and question answering over any modality, speech recognition
  with CTC ([`OmniDebertaForCTC`], built for text and audio only), referring segmentation
  ([`OmniDebertaForImageSegmentation`], built for text and images only) and replaced token detection.

## Usage examples

**Embeddings**

```python
import numpy as np
import torch
from PIL import Image

from transformers import AutoModel, AutoProcessor


model_id = "knowledgator/OmniDeberta-300M-A150M"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto")

image = Image.new("RGB", (320, 240), "red")
speech = np.zeros(16000, dtype=np.float32)  # 1 s at 16 kHz

with torch.inference_mode():
    texts = model(**processor(text=["a red square", "silence"], return_tensors="pt").to(model.device)).pooler_output
    images = model(**processor(images=[image], return_tensors="pt").to(model.device)).pooler_output
    clips = model(**processor(audio=[speech], return_tensors="pt").to(model.device)).pooler_output

print(texts @ images.T, texts @ clips.T)  # cosine similarities: the embeddings are normalized
```

A sample may hold several media and text: without placeholders its images, then its clips come before its text;
`processor.image_token` and `processor.audio_token` in the text put the next image or clip at that place.

**Sentence Transformers**

The checkpoints load in [Sentence Transformers](https://sbert.net) (>= 6.1) without remote code: the embedding is
`pooler_output` for every modality, and the prompts the model was trained with are stored by task. As in training,
a prompt applies to inputs with text: images and clips alone are embedded without it.

```python
from PIL import Image
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("knowledgator/OmniDeberta-300M-A150M")
image = Image.new("RGB", (320, 240), "red")

queries = model.encode(["a red square"], prompt_name="query")
documents = model.encode([image, {"image": image, "text": "a red square on a white page"}])
print(model.similarity(queries, documents))
```

**Image descriptions and speech transcripts**

Generation is mask-predict: `[MASK]` slots after the prompt, filled over several forward passes, left to right in
blocks, until the model ends the text with `[SEP]`. See [`OmniDebertaGenerationConfig`] for the options.

```python
import torch
from PIL import Image

from transformers import AutoModelForMaskedLM, AutoProcessor


model_id = "knowledgator/OmniDeberta-300M-A150M"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, device_map="auto")

image = Image.new("RGB", (320, 240), "red")
inputs = processor(images=[image], task="describe", return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, tokenizer=processor.tokenizer)
print(processor.batch_decode(output, skip_special_tokens=True))
```

For a transcript, pass `audio=[waveform]` (16 kHz) and `task="transcribe"`; the number of slots follows the length of
the clip.

**Masked word prediction**

```python
inputs = processor(images=[image], text=f"The square is {processor.tokenizer.mask_token}.", return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs.to(model.device)).logits
masked = (inputs.input_ids[0] == processor.tokenizer.mask_token_id).nonzero().item()
print(processor.decode(logits[0, masked].argmax()))
```

## Notes

- A model for some modalities only loads from any checkpoint: `AutoModel.from_pretrained(model_id, modalities=["text"])`
  (in Sentence Transformers `config_kwargs={"modalities": ["text"]}`) builds no image or audio stem and no image or audio
  experts, and skips their weights: 148M parameters instead of 300M for `OmniDeberta-300M-A150M`, the same text
  embeddings.
- The image processor resizes with LANCZOS, as in training. torchvision supports it from 0.27 on; with older versions
  [`AutoImageProcessor`] picks the PIL backend ([`OmniDebertaImageProcessorPil`]).
- The relative position terms are an additive attention bias: the model supports the `eager`, `sdpa` and `flex_attention`
  implementations, not FlashAttention.

## OmniDebertaConfig

[[autodoc]] OmniDebertaConfig

## OmniDebertaImageProcessor

[[autodoc]] OmniDebertaImageProcessor

## OmniDebertaImageProcessorPil

[[autodoc]] OmniDebertaImageProcessorPil

## OmniDebertaFeatureExtractor

[[autodoc]] OmniDebertaFeatureExtractor
    - __call__

## OmniDebertaProcessor

[[autodoc]] OmniDebertaProcessor
    - __call__

## OmniDebertaGenerationConfig

[[autodoc]] OmniDebertaGenerationConfig

## OmniDebertaGenerationMixin

[[autodoc]] OmniDebertaGenerationMixin
    - generate

## OmniDebertaGenerationOutput

[[autodoc]] OmniDebertaGenerationOutput

## OmniDebertaModel

[[autodoc]] OmniDebertaModel
    - forward

## OmniDebertaForMaskedLM

[[autodoc]] OmniDebertaForMaskedLM
    - forward
    - generate

## OmniDebertaForReplacedTokenDetection

[[autodoc]] OmniDebertaForReplacedTokenDetection
    - forward

## OmniDebertaForSequenceClassification

[[autodoc]] OmniDebertaForSequenceClassification
    - forward

## OmniDebertaForTokenClassification

[[autodoc]] OmniDebertaForTokenClassification
    - forward

## OmniDebertaForQuestionAnswering

[[autodoc]] OmniDebertaForQuestionAnswering
    - forward

## OmniDebertaForCTC

[[autodoc]] OmniDebertaForCTC
    - forward

## OmniDebertaForImageSegmentation

[[autodoc]] OmniDebertaForImageSegmentation
    - forward
