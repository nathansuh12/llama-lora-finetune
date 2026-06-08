# LLaMA LoRA Fine-tuning

A from-scratch implementation of **LoRA (Low-Rank Adaptation)** in PyTorch, applied to fine-tune **Llama-3.2-3B-Instruct** for binary sentiment classification on SST-2.

Rather than using `peft`, the LoRA layers here are written by hand — the goal was to understand the mechanism end to end: how the low-rank update is parameterized, where it gets injected into the transformer, and how the trainable parameter count collapses while the base weights stay frozen.

## How LoRA works here

LoRA freezes a pretrained linear layer's weight `W` and learns a low-rank update instead of touching `W` directly. For an input `x`, the modified layer computes:

```
output = W·x  +  (α / r) · (x · A · B)
```

where `A ∈ R^(in × r)` and `B ∈ R^(r × out)`, with `r` (the rank) much smaller than the layer dimensions. `A` is initialized from a scaled normal distribution and `B` is initialized to zeros, so the LoRA path contributes nothing at the start of training and the model begins identical to the pretrained base.

Two small modules implement this:

- **`LoRALayer`** — holds the trainable `A` and `B` matrices and applies the `(α / r)` scaling.
- **`LinearWithLoRA`** — wraps an existing `nn.Linear`, runs the frozen layer and the LoRA path in parallel, and sums them. Swapping a base linear layer for LoRA is just `layer = LinearWithLoRA(layer, rank, alpha)`.

## What gets fine-tuned

The base model is fully frozen, then LoRA is injected into the **query** and **value** projections (`q_proj`, `v_proj`) of all 28 decoder layers. Only the LoRA `A`/`B` matrices are marked trainable.

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` (~3.2B params) |
| LoRA target modules | `q_proj`, `v_proj` (all 28 layers) |
| Rank `r` | 8 |
| Alpha `α` | 16 (scaling = α/r = 2.0) |
| **Trainable params** | **~2.29M (≈ 0.07% of the model)** |

## Task & data

The model is fine-tuned on the [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) sentiment dataset, reframed as an instruction-following task. Each example is formatted as:

```
Classify the sentiment as positive or negative.
Text: <sentence>
Sentiment: <positive|negative>
```

Training uses 5,000 examples, tokenized to a max length of 64 with causal-LM labels (padding masked to `-100`). At inference, the model is prompted with everything up to `Sentiment:` and generates a single token.

## Training setup

| | |
|---|---|
| Trainer | Hugging Face `Trainer` |
| Batch size | 2 per device |
| Epochs | 3 |
| Learning rate | 3e-4 |
| Precision | bf16 |
| Hardware | NVIDIA RTX 3060 Ti (8 GB) |
| Runtime | ~14.5 min (final training loss ≈ 1.54) |

## Setup

Llama-3.2 is a gated model — you'll need a Hugging Face account, acceptance of Meta's license on the [model page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), and to be logged in:

```bash
huggingface-cli login
```

Dependencies:

```bash
pip install torch torchvision torchaudio   # CUDA build matching your setup
pip install transformers datasets numpy
```

Developed with `torch 2.12.0+cu126`, `transformers`, and `datasets`. A CUDA-capable GPU is recommended.

## Usage

The full pipeline lives in `loradev.ipynb`, run top to bottom:

1. Define `LoRALayer` and `LinearWithLoRA`.
2. Load Llama-3.2-3B-Instruct and freeze all parameters.
3. Inject LoRA into the attention projections and unfreeze the LoRA params.
4. Load and format SST-2, then tokenize.
5. Train with the HF `Trainer`.
6. Run inference with `classify_sentiment(text)`.

```python
classify_sentiment("A genuinely moving, beautifully acted film.")
# -> "positive"
```

## Notes & limitations

- The notebook is exploratory and includes some scratch/inspection cells (toy LoRA examples, environment checks) alongside the main pipeline.
- LoRA dropout is defined as a hyperparameter but not applied inside `LoRALayer` — adding it would more closely match the reference implementation.
- Labels are computed over the full prompt+answer sequence rather than the answer tokens alone, which is fine for this small demo but masks the answer span if you want stricter supervision.
