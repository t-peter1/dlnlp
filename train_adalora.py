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


# Configuration (aligned with LoRA hyperparams; AdaLoRA rank schedule unchanged)
MODEL_ID = "gpt2-medium"
LORA_RANK_INIT = 4
LORA_RANK_TARGET = 2
LORA_ALPHA = 32
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-4
UPDATE_INTERVAL = 100
WARMUP_RATIO = 0.3
TRAIN_MAX_LEN = 128


def preprocess_function(examples, tokenizer):
    texts = [
        f"{mr} || {target} {tokenizer.eos_token}"
        for mr, target in zip(examples["meaning_representation"], examples["target"])
    ]

    batch = tokenizer(
        texts,
        max_length=TRAIN_MAX_LEN,
        truncation=True,
        padding="max_length",
    )

    # Label masking to train only on the target (matches train.py)
    batch["labels"] = []
    sep_token_id = tokenizer.encode(" ||")[0]
    for i in range(len(batch["input_ids"])):
        input_ids = batch["input_ids"][i]
        labels = list(input_ids)

        try:
            sep_index = labels.index(sep_token_id) + 1
        except ValueError:
            sep_index = 0

        # Mask prompt (MR) and separator
        for j in range(sep_index):
            labels[j] = -100

        # Mask padding
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
        #self.controller.maybe_svd_compress(state.global_step, state.max_steps)
        return control


def main():
    # Data
    dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = dataset["train"].select(range(5000)).map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    val_dataset = dataset["validation"].select(range(500)).map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    # Model
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad = False

    model, lora_modules = inject_adalora(model, rank=LORA_RANK_INIT, alpha=LORA_ALPHA)
    model.to("cuda")

    print("--- Parameter Check (AdaLoRA) ---")
    print_trainable_parameters(model)

    # Controller
    r_init_total = len(lora_modules) * LORA_RANK_INIT
    r_target_total = len(lora_modules) * LORA_RANK_TARGET
    controller = AdaLoRAController(
        lora_modules,
        r_init_total=r_init_total,
        r_target_total=r_target_total,
        update_interval=UPDATE_INTERVAL,
        warmup_ratio=WARMUP_RATIO,
    )

    # Training
    training_args = TrainingArguments(
        output_dir="./adalora_gpt2_e2e",
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        warmup_ratio=0.1,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[AdaLoRACallback(controller)],
    )

    trainer.train()

    # Save AdaLoRA weights + masks (adapter-only checkpoint)
    model = trainer.model
    state = model.state_dict()
    mask_keys = [k for k in state if "rank_mask" in k]
    print(f"[save] total keys: {len(state)}, rank_mask keys: {len(mask_keys)}, sample: {mask_keys[:10]}")

    adalora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaLoRAConv1D):
            for k, v in module.state_dict().items():
                adalora_state[f"{name}.{k}"] = v.cpu()

    saved_mask_count = sum("rank_mask" in k for k in adalora_state)
    print(f"[save] filtered keys: {len(adalora_state)}, rank_mask saved: {saved_mask_count}")

    meta = {
        "rank_init": LORA_RANK_INIT,
        "rank_target": LORA_RANK_TARGET,
        "alpha": LORA_ALPHA,
        "update_interval": UPDATE_INTERVAL,
        "warmup_ratio": WARMUP_RATIO,
        "model_id": MODEL_ID,
        "train_warmup_ratio": 0.1,
        "max_length": TRAIN_MAX_LEN,
    }

    torch.save({"state_dict": adalora_state, "meta": meta}, "adalora_checkpoint.pt")

    # Optionally keep full HF save for convenience
    trainer.save_model("./adalora_gpt2_e2e")
    tokenizer.save_pretrained("./adalora_gpt2_e2e")


if __name__ == "__main__":
    main()
