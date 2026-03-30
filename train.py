# train.py
"""
Entry point for fine-tuning a model with a chosen PEFT method.

Examples:
    python train.py --task sst2 --method lora --epochs 3 --batch_size 16
    python train.py --task agnews --method qlora --epochs 1 --batch_size 8
    python train.py --task samsum --method adapter --model google/flan-t5-small --epochs 2
    python train.py --config configs/lora_sst2.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from src.data import TASK_CONFIG, get_dataset, get_dataloader
from src.methods import build_model
from src.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="Efficient Fine-Tuning Zoo")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file. CLI args override config values.")
    parser.add_argument("--task", type=str, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--method", type=str,
                        choices=["lora", "qlora", "adapter", "prompt_tuning", "full"])
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    return parser.parse_args()


def merge_config(args):
    """Load YAML config if provided; CLI args take precedence."""
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    # CLI args override config
    for k, v in vars(args).items():
        if v is not None and k != "config":
            cfg[k] = v
    return cfg


def main():
    args = parse_args()
    cfg = merge_config(args)

    task = cfg["task"]
    method = cfg["method"]
    model_name = cfg.get("model", "distilbert-base-uncased")
    epochs = int(cfg.get("epochs", 3))
    batch_size = int(cfg.get("batch_size", 16))
    lr = float(cfg.get("lr", 2e-4))
    max_length = int(cfg.get("max_length", 128))
    output_dir = cfg.get("output_dir") or f"outputs/{task}_{method}"
    use_wandb = bool(cfg.get("wandb", False))

    print(f"\n{'='*50}")
    print(f"  Task:    {task}")
    print(f"  Method:  {method}")
    print(f"  Model:   {model_name}")
    print(f"  Epochs:  {epochs}")
    print(f"  Device:  (auto-detected)")
    print(f"{'='*50}\n")

    # --- Tokenizer & data ---
    print("[main] Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, eval_ds = get_dataset(task, tokenizer, max_length=max_length)
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = get_dataloader(eval_ds, batch_size=batch_size, shuffle=False)

    # --- Model ---
    num_labels = TASK_CONFIG[task]["num_labels"]
    print(f"[main] Building model with method={method!r}...")
    model, trainable, total = build_model(model_name, task, method, num_labels)
    pct = 100 * trainable / total
    print(f"[main] Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    # --- Train ---
    run_name = f"{task}_{method}_{model_name.split('/')[-1]}"
    history = train(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=epochs,
        lr=lr,
        output_dir=output_dir,
        use_wandb=use_wandb,
        run_name=run_name,
    )

    # --- Save run summary ---
    summary = {
        "task": task,
        "method": method,
        "model": model_name,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(pct, 4),
        "epochs": epochs,
        "final_train_loss": history["train_loss"][-1],
        "final_eval_loss": history["eval_loss"][-1],
        "total_time_s": history["total_time_s"],
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[main] Run summary saved to {output_dir}/run_summary.json")
    print(f"[main] Final eval loss: {history['eval_loss'][-1]:.4f}")
    print(f"[main] Total time: {history['total_time_s']}s")


if __name__ == "__main__":
    main()
