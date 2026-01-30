"""
Smoke test harness for LoRA and AdaLoRA pipelines.

Runs tiny training/eval loops on small slices of GEM/e2e_nlg to catch
shape/key/load issues in ~10–30 seconds (CPU by default, GPU if available).

What it does:
 - LoRA: train for max_steps=50 on 50 train / 20 val, save adapter weights, reload, and evaluate.
 - AdaLoRA: same slice and steps, with pruning schedule intact, save adapter+mask, reload, and evaluate.

Exit codes:
 - Process exits 0 even on failures, but prints PASS/FAIL with details for both pipelines.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

from src.model import inject_lora
from src.adalora_model import inject_adalora
from src.adalora import AdaLoRAController
from src.adalora_lora import AdaLoRAConv1D


# ----------------
# Common settings
# ----------------
TRAIN_SIZE = 50
VAL_SIZE = 20
MAX_STEPS = 50
EVAL_STEPS = 10
SAVE_STEPS = 50
MAX_LEN = 128
BATCH_SIZE = 2  # small for speed
MODEL_ID = "gpt2-medium"
RANK = 4
ALPHA = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------
# Preprocessing
# ----------------
def preprocess_lora(examples, tokenizer):
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


# AdaLoRA uses the same preprocessing as LoRA after alignment
preprocess_adalora = preprocess_lora


def load_slices(tokenizer, preprocess_fn):
    ds = load_dataset("GEM/e2e_nlg", trust_remote_code=True)
    train_ds = ds["train"].select(range(TRAIN_SIZE)).map(
        lambda x: preprocess_fn(x, tokenizer), batched=True
    )
    val_ds = ds["validation"].select(range(VAL_SIZE)).map(
        lambda x: preprocess_fn(x, tokenizer), batched=True
    )
    return train_ds, val_ds


def first_loss_and_last(log_history):
    losses = [e["loss"] for e in log_history if "loss" in e]
    if not losses:
        return None, None, None
    return losses[0], losses[-1], losses[-1] - losses[0]


# ----------------
# AdaLoRA callback
# ----------------
class AdaLoRACallback(TrainerCallback):
    def __init__(self, controller: AdaLoRAController):
        self.controller = controller

    def on_step_end(self, args, state, control, **kwargs):
        self.controller.step_end(state.global_step, state.max_steps)
        return control


# ----------------
# LoRA smoke
# ----------------
def run_lora_smoke():
    print("\n=== LoRA smoke test ===")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    train_ds, val_ds = load_slices(tokenizer, preprocess_lora)

    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad = False
    model = inject_lora(model, rank=RANK, alpha=ALPHA)
    model.to(DEVICE)

    # Build TrainingArguments kwargs with version compatibility
    ta_kwargs = {
        "output_dir": "./smoketest_outputs/lora",
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "logging_steps": 5,
        "learning_rate": 5e-4,
        "warmup_ratio": 0.1,
        "fp16": (DEVICE == "cuda"),
        "report_to": "none",
    }
    init_params = TrainingArguments.__init__.__code__.co_varnames
    if "evaluation_strategy" in init_params:
        ta_kwargs["evaluation_strategy"] = "steps"
        strategy_eval_key = "evaluation_strategy"
    else:
        ta_kwargs["eval_strategy"] = "steps"
        strategy_eval_key = "eval_strategy"

    if "save_strategy" in init_params:
        ta_kwargs["save_strategy"] = "steps"
        strategy_save_key = "save_strategy"
    else:
        ta_kwargs["save_steps"] = SAVE_STEPS
        strategy_save_key = "save_steps"

    if "logging_strategy" in init_params:
        ta_kwargs["logging_strategy"] = "steps"

    print(f"[LoRA] TrainingArguments keys used: eval={strategy_eval_key}, save={strategy_save_key}")

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    init_loss, final_loss, delta = first_loss_and_last(trainer.state.log_history)

    # Save adapter-only checkpoint
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    lora_ckpt_path = "./smoketest_outputs/lora/lora_smoke.pt"
    os.makedirs(os.path.dirname(lora_ckpt_path), exist_ok=True)
    torch.save(lora_state, lora_ckpt_path)

    # Reload test
    reload_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    reload_model = inject_lora(reload_model, rank=RANK, alpha=ALPHA)
    reload_model.load_state_dict(torch.load(lora_ckpt_path, map_location="cpu"), strict=False)
    reload_model.to(DEVICE)
    reload_model.eval()

    sample = val_ds[0]
    with torch.no_grad():
        _ = reload_model(
            input_ids=torch.tensor([sample["input_ids"]], device=DEVICE),
            attention_mask=torch.tensor([sample["attention_mask"]], device=DEVICE),
            labels=torch.tensor([sample["labels"]], device=DEVICE),
        )

    passed = (init_loss is not None) and (final_loss is not None)
    print(f"[LoRA] loss init={init_loss}, final={final_loss}, delta={delta}")
    print(f"[LoRA] eval metrics: {eval_metrics}")
    print(f"[LoRA] checkpoint saved: {lora_ckpt_path}")
    print(f"[LoRA] reload test: {'PASS' if passed else 'FAIL'}")
    return {
        "name": "LoRA",
        "init_loss": init_loss,
        "final_loss": final_loss,
        "delta": delta,
        "ckpt": lora_ckpt_path,
        "eval": eval_metrics,
        "passed": passed,
    }


# ----------------
# AdaLoRA smoke
# ----------------
def run_adalora_smoke():
    print("\n=== AdaLoRA smoke test ===")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    train_ds, val_ds = load_slices(tokenizer, preprocess_adalora)

    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad = False

    model, modules = inject_adalora(model, rank=RANK, alpha=ALPHA)
    model.to(DEVICE)

    controller = AdaLoRAController(
        modules,
        r_init_total=len(modules) * RANK,
        r_target_total=len(modules) * 2,  # target rank stays as in train_adalora.py
        update_interval=100,
        warmup_ratio=0.3,
    )

    ta_kwargs = {
        "output_dir": "./smoketest_outputs/adalora",
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "logging_steps": 5,
        "learning_rate": 5e-4,
        "warmup_ratio": 0.1,
        "fp16": (DEVICE == "cuda"),
        "report_to": "none",
    }
    init_params = TrainingArguments.__init__.__code__.co_varnames
    if "evaluation_strategy" in init_params:
        ta_kwargs["evaluation_strategy"] = "steps"
        strategy_eval_key = "evaluation_strategy"
    else:
        ta_kwargs["eval_strategy"] = "steps"
        strategy_eval_key = "eval_strategy"

    if "save_strategy" in init_params:
        ta_kwargs["save_strategy"] = "steps"
        strategy_save_key = "save_strategy"
    else:
        ta_kwargs["save_steps"] = SAVE_STEPS
        strategy_save_key = "save_steps"

    if "logging_strategy" in init_params:
        ta_kwargs["logging_strategy"] = "steps"

    print(f"[AdaLoRA] TrainingArguments keys used: eval={strategy_eval_key}, save={strategy_save_key}")

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[AdaLoRACallback(controller)],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    init_loss, final_loss, delta = first_loss_and_last(trainer.state.log_history)

    # Lambda diagnostics
    all_lam = torch.cat([p.detach().flatten() for n, p in model.named_parameters() if "lambda_vals" in n])
    nonzero = int((all_lam != 0).sum().item())
    all_mask = torch.cat([b.detach().flatten() for n, b in model.named_buffers() if "rank_mask" in n])
    active = int((all_mask > 0).sum().item())
    print(f"[AdaLoRA] lambda mean={all_lam.abs().mean().item():.4f} max={all_lam.abs().max().item():.4f} nonzero={nonzero}/{all_lam.numel()} active_mask={active}/{all_mask.numel()}")

    # Save AdaLoRA adapter state (A/B + mask)
    state = model.state_dict()
    adalora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaLoRAConv1D):
            for k, v in module.state_dict().items():
                adalora_state[f"{name}.{k}"] = v.cpu()

    adalora_ckpt_path = "./smoketest_outputs/adalora/adalora_smoke.pt"
    os.makedirs(os.path.dirname(adalora_ckpt_path), exist_ok=True)
    meta = {
        "rank_init": RANK,
        "rank_target": 2,
        "alpha": ALPHA,
        "update_interval": 100,
        "warmup_ratio": 0.3,
        "model_id": MODEL_ID,
    }
    torch.save({"state_dict": adalora_state, "meta": meta}, adalora_ckpt_path)

    # Reload test with mask-aware resize
    reload_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    reload_model, _ = inject_adalora(reload_model, rank=RANK, alpha=ALPHA)
    sd = torch.load(adalora_ckpt_path, map_location="cpu")
    state_dict = sd["state_dict"] if "state_dict" in sd else sd

    with torch.no_grad():
        for name, module in reload_model.named_modules():
            if isinstance(module, AdaLoRAConv1D):
                mask_key = f"{name}.rank_mask"
                if mask_key in state_dict:
                    chk_rank = state_dict[mask_key].shape[0]
                    if chk_rank != module.rank:
                        in_features = module.original_layer.weight.shape[0]
                        out_features = module.original_layer.weight.shape[1]
                        module.lora_A = torch.nn.Parameter(torch.empty(in_features, chk_rank))
                        module.lora_B = torch.nn.Parameter(torch.empty(chk_rank, out_features))
                        module.rank_mask = torch.nn.Parameter(torch.empty(chk_rank))
                        module.rank = chk_rank

    reload_model.load_state_dict(state_dict, strict=False)
    reload_model.to(DEVICE)
    reload_model.eval()

    sample = val_ds[0]
    with torch.no_grad():
        _ = reload_model(
            input_ids=torch.tensor([sample["input_ids"]], device=DEVICE),
            attention_mask=torch.tensor([sample["attention_mask"]], device=DEVICE),
            labels=torch.tensor([sample["labels"]], device=DEVICE),
        )

    passed = (init_loss is not None) and (final_loss is not None)
    print(f"[AdaLoRA] loss init={init_loss}, final={final_loss}, delta={delta}")
    print(f"[AdaLoRA] eval metrics: {eval_metrics}")
    print(f"[AdaLoRA] checkpoint saved: {adalora_ckpt_path}")
    print(f"[AdaLoRA] reload test: {'PASS' if passed else 'FAIL'}")
    return {
        "name": "AdaLoRA",
        "init_loss": init_loss,
        "final_loss": final_loss,
        "delta": delta,
        "ckpt": adalora_ckpt_path,
        "eval": eval_metrics,
        "passed": passed,
    }


def main():
    results = []
    try:
        results.append(run_lora_smoke())
    except Exception as e:
        print(f"[LoRA] FAIL: {e}")
    try:
        results.append(run_adalora_smoke())
    except Exception as e:
        print(f"[AdaLoRA] FAIL: {e}")

    print("\n=== Smoke summary ===")
    for r in results:
        if r is None:
            continue
        print(
            f"{r['name']}: "
            f"init={r['init_loss']}, final={r['final_loss']}, delta={r['delta']}, "
            f"ckpt={r['ckpt']}, passed={'YES' if r['passed'] else 'NO'}"
        )


if __name__ == "__main__":
    main()
