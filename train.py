import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from src.model import inject_lora, print_trainable_parameters

# Config
MODEL_ID = "gpt2-medium"
LORA_RANK = 4
LORA_ALPHA = 32
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-4

# prepare Data (E2E NLG)
# E2E dataset structure: "meaning_representation" (input) -> "human_reference" (target)
dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

"""
def preprocess_function(examples):
    # GEM/e2e_nlg uses 'meaning_representation' for input and 'target' for output
    inputs = [f"{mr} || {ref} {tokenizer.eos_token}" for mr, ref in zip(examples['meaning_representation'], examples['target'])]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    return model_inputs
"""
def preprocess_function(examples):
    texts = [f"{mr} || {target} {tokenizer.eos_token}" 
             for mr, target in zip(examples['meaning_representation'], examples['target'])]
    
    batch = tokenizer(texts, max_length=128, truncation=True, padding="max_length")
    
    # Label Masking
    batch["labels"] = []
    for i in range(len(batch["input_ids"])):
        input_ids = batch["input_ids"][i]
        labels = list(input_ids)
        
        try:
            sep_token_id = tokenizer.encode(" ||")[0]
            # find the first occurrence of the separator
            sep_index = labels.index(sep_token_id) + 1
        except ValueError:
            sep_index = 0
            
        # Mask the Prompt (MR) and the Separator
        for j in range(sep_index):
            labels[j] = -100
            
        # Mask all Padding tokens 
        for j in range(len(labels)):
            if input_ids[j] == tokenizer.pad_token_id:
                labels[j] = -100
                
        batch["labels"].append(labels)
    
    return batch


train_dataset = dataset["train"].map(preprocess_function, batched=True)
val_dataset = dataset["validation"].map(preprocess_function, batched=True)

model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

print("Freezing base model weights...")
for param in model.parameters():
    param.requires_grad = False
    
model = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
model.to("cuda")

print("--- Parameter Check ---")
print_trainable_parameters(model)

# training
training_args = TrainingArguments(
    output_dir="./lora_gpt2_e2e",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    warmup_ratio=0.1,
    fp16=True,
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

# only the LoRA weights
lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora" in k}
torch.save(lora_state_dict, "lora_weights_r4_preproccess.pt")