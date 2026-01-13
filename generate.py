import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.model import inject_lora

# 1. Setup
MODEL_ID = "gpt2-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)

# 2. Load the Base Model
print("Loading base model...")
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

# 3. Inject the LoRA Layers (Structure must match training!)
model = inject_lora(model, rank=4, alpha=32)

# 4. Load your Trained Weights
print("Loading trained LoRA weights...")
# Note: strict=False allows us to load even if some keys are missing/extra
# (useful since we are mixing frozen and trained weights)
model.load_state_dict(torch.load("lora_weights_only.pt"), strict=False)
model.to(device)
model.eval()

# 5. Test It
# Input: A structured meaning representation (MR)
text_input = "name[The Eagle] || eatType[coffee shop] || food[Japanese] || priceRange[cheap]"
print(f"\nINPUT: {text_input}")

input_ids = tokenizer.encode(text_input, return_tensors='pt').to(device)

# Generate
print("Generating...")
with torch.no_grad():
    output_tokens = model.generate(
        input_ids, 
        max_length=100, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("-" * 30)
print(f"OUTPUT: {generated_text}")
print("-" * 30)