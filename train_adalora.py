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


# Configuration (kept close to train.py defaults)
MODEL_ID = "gpt2-medium"
LORA_RANK_INIT = 4
LORA_RANK_TARGET = 2
LORA_ALPHA = 32
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-4
UPDATE_INTERVAL = 100
WARMUP_RATIO = 0.3


def preprocess_function(examples, tokenizer):
    inputs = [
        f"{mr} || {ref} {tokenizer.eos_token}"
        for mr, ref in zip(examples["meaning_representation"], examples["target"])
    ]
    return tokenizer(inputs, max_length=512, truncation=True, padding="max_length")


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
        fp16=False,
        bf16=False,
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

    model = trainer.model
    model.eval()

    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "svd_compress"):
                m.svd_compress(target_rank=m.rank)
                if hasattr(m, "rank_mask"):
                    del m.rank_mask

    trainer.save_model("./adalora_gpt2_e2e")
    tokenizer.save_pretrained("./adalora_gpt2_e2e")


if __name__ == "__main__":
    main()
