# Efficient Fine-Tuning Zoo

Reproducible benchmark of parameter-efficient fine-tuning (PEFT) methods on NLP tasks using HuggingFace Transformers and PEFT.

**Methods covered:** LoRA · QLoRA · Adapters (AdaLoRA) · Prompt Tuning · Full fine-tuning (baseline)  
**Tasks:** SST-2 (sentiment) · AG News (topic classification) · SAMSum (dialogue summarisation)

---

## Quickstart

```bash
git clone https://github.com/parisazeynaly/Efficient-ft-zoo.git
cd Efficient-ft-zoo
make setup

# Train a single run
python train.py --task sst2 --method lora --epochs 3 --batch_size 16

# Or use a config file
python train.py --config configs/lora_sst2.yaml

# Evaluate a checkpoint
python -m src.eval --checkpoint outputs/sst2_lora --task sst2 --model distilbert-base-uncased

# Run the full benchmark
make benchmark
```

---

## Project structure

```
Efficient-ft-zoo/
├── train.py                  # entry point — CLI + YAML config support
├── src/
│   ├── data.py               # dataset loading & tokenisation (SST-2, AG News, SAMSum)
│   ├── methods.py            # PEFT method wrappers (LoRA, QLoRA, Adapter, Prompt Tuning)
│   ├── trainer.py            # training loop with W&B logging
│   └── eval.py               # evaluation (accuracy/F1 for classification, ROUGE for seq2seq)
├── configs/                  # YAML configs for reproducible experiments
│   ├── lora_sst2.yaml
│   ├── qlora_agnews.yaml
│   ├── adapter_samsum.yaml
│   └── prompt_tuning_sst2.yaml
├── outputs/                  # saved checkpoints (gitignored)
├── results/                  # metrics and benchmark table
├── requirements.txt
├── Makefile
└── setup.sh
```

---

## Methods

| Method | Key idea | Trainable params |
|--------|----------|-----------------|
| Full | All parameters updated | 100% |
| LoRA | Low-rank update matrices on attention weights | ~0.5% |
| QLoRA | LoRA on a 4-bit quantised base model | ~0.5% |
| Adapter (AdaLoRA) | Adaptive rank allocation via SVD decomposition | ~0.5% |
| Prompt Tuning | Only soft prompt embeddings are trained | ~0.01% |

---

## Benchmark results

*(Fill in after running `make benchmark`)*

| Task | Method | Trainable % | Accuracy / ROUGE-L | Train time (s) |
|------|--------|------------|-------------------|----------------|
| SST-2 | full | 100.00% | — | — |
| SST-2 | lora | ~0.5% | — | — |
| SST-2 | qlora | ~0.5% | — | — |
| SST-2 | adapter | ~0.5% | — | — |
| SST-2 | prompt_tuning | ~0.01% | — | — |
| AG News | lora | ~0.5% | — | — |
| SAMSum | adapter | ~0.5% | — | — |

---

## Configuration

All hyperparameters are set via YAML configs in `configs/`. Example:

```yaml
task: sst2
method: lora
model: distilbert-base-uncased
epochs: 3
batch_size: 16
lr: 2e-4
output_dir: outputs/sst2_lora
wandb: false
```

CLI arguments override config values:
```bash
python train.py --config configs/lora_sst2.yaml --epochs 5 --wandb
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for QLoRA (4-bit quantisation requires bitsandbytes)
- See `requirements.txt` for all dependencies

---

## Related work

- [Hu et al. 2021 — LoRA](https://arxiv.org/abs/2106.09685)
- [Dettmers et al. 2023 — QLoRA](https://arxiv.org/abs/2305.14314)
- [Zhang et al. 2023 — AdaLoRA](https://arxiv.org/abs/2303.10512)
- [Lester et al. 2021 — Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [HuggingFace PEFT library](https://github.com/huggingface/peft)
