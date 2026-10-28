# DistilRLHF

**RLHF from scratch in PyTorch:** supervised fine-tuning, a preference-based reward model, and PPO for DistilGPT2 on Alpaca, written as plain PyTorch with no Hugging Face Trainer or TRL.

**Goal:** make DistilGPT2 better at following instructions using human feedback and PPO.

> **Status: in progress.** Supervised fine-tuning (SFT) is implemented. The human-feedback stages are **not done yet**: the reward model and PPO fine-tuning still need to be built.

---

## Pipeline

```
Pretrained DistilGPT2 → Supervised Fine-Tuning → Human Feedback (preference data) → Reward Model → PPO Fine-Tuning → Evaluation
```

| Stage | Status |
| --- | --- |
| Supervised fine-tuning (SFT) | ✅ Implemented |
| Training infrastructure (validation, checkpoint/resume) | 🚧 In progress |
| Human feedback (preference data) | ⬜ Not done |
| Reward model | ⬜ Not done |
| PPO fine-tuning (RLHF) | ⬜ Not done |
| Evaluation | ⬜ Not done |

## Stack

- Python, PyTorch
- Hugging Face Transformers and Datasets (model and tokenizer loading, data only)
- Model: [`distilbert/distilgpt2`](https://huggingface.co/distilbert/distilgpt2)
- Data: [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)
- Hardware: Apple M1 Pro (PyTorch MPS); NVIDIA GPU via Google Colab planned

---

## What's implemented (as of October 28, 2026)

### 1. Data preparation

The Alpaca dataset is split 80 / 10 / 10 into train, validation and test sets with a fixed seed (42). That gives about 41.6k training examples and about 5.2k each for validation and test.

Each example is split into a **prompt** and a **response**. The prompt is the instruction, plus the input when the example has one, followed by the `### Response:` marker. The response is Alpaca's reference output.

```
### Instruction:
{instruction}
### Input:
{input}          ← only included when the example has an input
### Response:
{output}
```

### 2. Tokenization with response-only loss

We tokenize the prompt and response separately so we can identify which tokens belong to the response. However, it's the `labels` containing `-100` for prompt tokens that actually causes the loss to be calculated only on response tokens, because DistilGPT2's cross-entropy loss function ignores labels with a value of `-100`. Cross-entropy is used because next-token prediction is a classification problem: at each position, the model picks one token out of its vocabulary.

| Field | Contents |
| --- | --- |
| `input_ids` | prompt tokens + response tokens + EOS |
| `attention_mask` | 1 for every real token (0 for padding, added at batch time) |
| `labels` | `-100` for every prompt token, then the response tokens + EOS |

`input_ids` = prompt (instruction + input) + response + EOS. Since DistilGPT2 predicts the next token, we give it the correct previous response tokens to help predict the next one. So `labels` and `input_ids` need to be the same size, to match the number of tokens.

DistilGPT2's cross-entropy loss function knows to:

- **Ignore the instruction and input:** their labels are `-100`.
- **Calculate loss on the response:** its labels contain the correct token IDs.
- **Calculate loss on EOS:** this teaches the model when to finish its response.

The loss skips any position whose label is `-100`. Because every prompt position is labelled `-100`, the model is never graded on predicting the prompt, only on predicting the response. Appending EOS (`<|endoftext|>`, id 50256) to the response teaches the model when to stop.

Sequences longer than DistilGPT2's 1,024-token context are dropped rather than truncated, which removed one training example.

### 3. Custom batching

A hand-written collator pads each batch to its longest sequence:

- `input_ids` are padded with the EOS token (DistilGPT2 has no dedicated pad token)
- `attention_mask` is padded with `0`, so the model ignores padding
- `labels` must match `input_ids` in length, one label for every input token, so they are padded too. They are padded with `-100` so padding never contributes to the loss

Batches are served by a standard PyTorch `DataLoader` (batch size 4, shuffled).

### 4. SFT training loop

The loop is plain PyTorch:

```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
loss = outputs.loss          # causal next-token cross-entropy

optimizer.zero_grad()
loss.backward()
optimizer.step()             # AdamW, lr 5e-5, weight decay 0.01
```

---

## Next steps

- Finish supervised fine-tuning on the full training set.
- Test an A100 GPU on Google Colab to see whether it trains faster than the M1 Pro.

---

## References

- [Hugging Face: Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [Hugging Face: Generation with LLMs](https://huggingface.co/docs/transformers/llm_tutorial)
- Taori et al., *Stanford Alpaca: An Instruction-following LLaMA Model* (2023)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Ouyang et al., *Training language models to follow instructions with human feedback* (2022)
