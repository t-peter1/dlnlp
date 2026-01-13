import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# CONFIG
MODEL_ID = "gpt2-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Raw GPT-2 (No LoRA injection!)
print("Loading raw GPT-2 Medium...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.to(device)
model.eval()

# 2. The Input
text_input = "name[Blue Spice] || eatType[pub] || food[Chinese] || priceRange[cheap]"
print(f"\nINPUT: {text_input}")

input_ids = tokenizer.encode(text_input, return_tensors='pt').to(device)
attention_mask = torch.ones_like(input_ids)

# 3. Generate
print("Generating...")
with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False, # Deterministic
        num_beams=1
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("-" * 40)
print(f"OUTPUT: {generated_text}")
print("-" * 40)