import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from src.model import inject_lora, print_trainable_parameters

#Configuration
MODEL_ID = "gpt2-medium"
LORA_RANK = 4
LORA_ALPHA = 32
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-4

# 2. Prepare Data (E2E NLG)
# E2E dataset structure: "meaning_representation" (input) -> "human_reference" (target)
dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # GEM/e2e_nlg uses 'meaning_representation' for input and 'target' for output
    inputs = [f"{mr} || {ref} {tokenizer.eos_token}" for mr, ref in zip(examples['meaning_representation'], examples['target'])]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    return model_inputs

# Subsetting for faster experimentation (Optional but recommended)
train_dataset = dataset["train"].select(range(5000)).map(preprocess_function, batched=True)
val_dataset = dataset["validation"].select(range(500)).map(preprocess_function, batched=True)

# 3. Prepare Model
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

print("Freezing base model weights...")
for param in model.parameters():
    param.requires_grad = False
    
model = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
model.to("cuda")

print("--- Parameter Check ---")
print_trainable_parameters(model)
# You should see ~0.04% trainable parameters

# 4. Training
training_args = TrainingArguments(
    output_dir="./lora_gpt2_e2e",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True, # V100 supports fp16, gives 2x speedup
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting training...")
trainer.train()

# 5. Save only the LoRA weights (not the whole model!)
# This demonstrates the storage efficiency claim of the paper
torch.save(model.state_dict(), "lora_weights_only.pt")