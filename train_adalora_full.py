"""
Full LoRA-style training pipeline with AdaLoRA adapters.

Matches train.py data prep, objective, and hyperparameters (epochs, lr,
warmup, fp16, batching, logging/eval/save) but swaps LoRA for AdaLoRA
with the current pruning schedule (warmup_fraction=0.3, update_interval=100).

Outputs:
 - Standard HF Trainer checkpoint dir (for resume)
 - Adapter-only checkpoint adalora_full_checkpoint.pt (A/B + rank_mask + meta)

Optional quick check:
    python train_adalora_full.py --dry_run   # runs a tiny max_steps loop
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.adalora import AdaLoRAController
from src.adalora_lora import AdaLoRAConv1D
from src.adalora_model import inject_adalora
from src.model import print_trainable_parameters


# Hyperparameters (mirrors train.py)
MODEL_ID = "gpt2-medium"
LORA_RANK_INIT = 4
LORA_RANK_TARGET = 2
LORA_ALPHA = 32
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-4
WARMUP_RATIO_TRAIN = 0.1  # Trainer warmup (same as train.py)
MAX_LEN = 128
LOGGING_STEPS = 50

# AdaLoRA controller settings (unchanged from existing AdaLoRA)
ADALORA_UPDATE_INTERVAL = 100
ADALORA_WARMUP_FRACTION = 0.3


def preprocess_function(examples, tokenizer):
    # Same prompt/labeling as train.py
    texts = [
        f"{mr} || {target} {tokenizer.eos_token}"
        for mr, target in zip(examples["meaning_representation"], examples["target"])
    ]
    batch = tokenizer(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )

    batch["labels"] = []
    sep_token_id = tokenizer.encode(" ||")[0]
    for i in range(len(batch["input_ids"])):
        input_ids = batch["input_ids"][i]
        labels = list(input_ids)

        try:
            sep_index = labels.index(sep_token_id) + 1
        except ValueError:
            sep_index = 0

        for j in range(sep_index):
            labels[j] = -100
        for j, tok in enumerate(input_ids):
            if tok == tokenizer.pad_token_id:
                labels[j] = -100

        batch["labels"].append(labels)

    return batch


class AdaLoRACallback(TrainerCallback):
    def __init__(self, controller: AdaLoRAController):
        self.controller = controller

    def on_step_end(self, args, state, control, **kwargs):
        self.controller.step_end(state.global_step, state.max_steps)
        return control


def build_training_args(output_dir, dry_run=False):
    # Keep same strategy as train.py (epoch-based) unless dry_run forces max_steps
    if dry_run:
        kwargs = dict(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            max_steps=5,
            evaluation_strategy="steps",
            eval_steps=2,
            save_strategy="steps",
            save_steps=5,
            logging_steps=1,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO_TRAIN,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )
    else:
        kwargs = dict(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LR,
            num_train_epochs=EPOCHS,
            logging_steps=LOGGING_STEPS,
            save_strategy="epoch",
            eval_strategy="epoch",
            warmup_ratio=WARMUP_RATIO_TRAIN,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run tiny sanity loop (max_steps=5)")
    args = parser.parse_args()

    # Data
    dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = dataset["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_dataset = dataset["validation"].map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Model
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad = False

    model, lora_modules = inject_adalora(model, rank=LORA_RANK_INIT, alpha=LORA_ALPHA)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Parameter Check (AdaLoRA full) ---")
    print_trainable_parameters(model)

    # Controller
    r_init_total = len(lora_modules) * LORA_RANK_INIT
    r_target_total = len(lora_modules) * LORA_RANK_TARGET
    controller = AdaLoRAController(
        lora_modules,
        r_init_total=r_init_total,
        r_target_total=r_target_total,
        update_interval=ADALORA_UPDATE_INTERVAL,
        warmup_ratio=ADALORA_WARMUP_FRACTION,
    )

    # Training
    training_args = build_training_args("./adalora_gpt2_e2e_full", dry_run=args.dry_run)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[AdaLoRACallback(controller)],
    )

    trainer.train()

    # Adapter-only checkpoint for eval_metrics_adalora.py compatibility
    state = model.state_dict()
    adalora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaLoRAConv1D):
            for k, v in module.state_dict().items():
                adalora_state[f"{name}.{k}"] = v.cpu()

    meta = {
        "model_id": MODEL_ID,
        "rank_init": LORA_RANK_INIT,
        "rank_target": LORA_RANK_TARGET,
        "alpha": LORA_ALPHA,
        "update_interval": ADALORA_UPDATE_INTERVAL,
        "warmup_ratio": ADALORA_WARMUP_FRACTION,
        "max_length": MAX_LEN,
    }

    os.makedirs("./adalora_gpt2_e2e_full", exist_ok=True)
    torch.save({"state_dict": adalora_state, "meta": meta}, "adalora_full_checkpoint.pt")

    # Also keep HF trainer checkpoint (already in output_dir via Trainer)
    print("Saved adapter checkpoint to adalora_full_checkpoint.pt")


if __name__ == "__main__":
    main()
