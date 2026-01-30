import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.model import inject_lora

# --- CONFIG ---
MODEL_ID = "gpt2-medium"
WEIGHTS_PATH = "lora_weights_only_fixed.pt"
RANK = 4 
ALPHA = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_review(text_input):
    print("Loading base GPT-2...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # inject lora
    model = inject_lora(model, rank=RANK, alpha=ALPHA)

    print(f"Loading LoRA weights from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

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
    
    print("-" * 40)
    print(f"OUTPUT: {generated_text}")
    print("-" * 40)

if __name__ == "__main__":
    # Test Case
    test_prompt = "name[Blue Spice] || eatType[pub] || food[Chinese] || priceRange[cheap]"
    generate_review(test_prompt)