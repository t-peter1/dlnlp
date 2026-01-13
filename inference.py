import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.model import inject_lora

# --- CONFIG ---
MODEL_ID = "gpt2-medium"
WEIGHTS_PATH = "lora_weights_only_fixed.pt"  # Your small ~1.5MB file
RANK = 4                                    # Must match what you trained with!
ALPHA = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_review(text_input):
    # 1. Load the empty Base Model (Heavy)
    print("Loading base GPT-2...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Modify the Architecture (Inject LoRA)
    # This creates the "slots" (A and B matrices) for our weights to go into
    model = inject_lora(model, rank=RANK, alpha=ALPHA)

    # 3. Load the Saved Weights (The "Cheat Sheet")
    print(f"Loading LoRA weights from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    
    # strict=False is KEY here. 
    # It tells PyTorch: "It's okay if the file doesn't have weights for the whole model,
    # just load the ones that match."
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Move to GPU
    model.to(device)
    model.eval()

    # 4. Generate Text
    print(f"\nINPUT: {text_input}")
    input_ids = tokenizer.encode(text_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=120,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Clean formatting to show just the output
    # (The model usually repeats the input, then adds "||", then the sentence)
    print("-" * 40)
    print(f"OUTPUT: {generated_text}")
    print("-" * 40)

if __name__ == "__main__":
    # Test Case
    test_prompt = "name[Blue Spice] || eatType[pub] || food[Chinese] || priceRange[cheap]"
    generate_review(test_prompt)