# Efficient-ft-zoo
Reproducible benchmark of parameter-efficient fine-tuning methods (LoRA, QLoRA, Adapters, Prompt-tuning) on NLP tasks using HuggingFace Transformers.
# Efficient Fine-Tuning Zoo (LLMs)
Reproducible benchmark of LoRA / QLoRA / Adapters / Prompt-tuning on small NLP tasks (SST-2, AG News, SAMSum).

## Quickstart
```bash
pip install -r requirements.txt
python train.py --task sst2 --method lora --epochs 1 --batch_size 16
python train.py --task agnews --method qlora --epochs 1 --batch_size 8
python -m src.eval
