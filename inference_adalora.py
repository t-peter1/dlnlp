import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.adalora_model import inject_adalora


# --- CONFIG ---
MODEL_ID = "gpt2-medium"
CKPT_PATH = "adalora_checkpoint.pt"
RANK_INIT = 4
ALPHA = 32

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_review(text_input):
    # Load checkpoint meta
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("meta", {})
    else:
        state_dict = checkpoint
        meta = {}
        
    rank = meta.get("rank_init", RANK_INIT)
    alpha = meta.get("alpha", ALPHA)

    # Base model + AdaLoRA injection
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model, _ = inject_adalora(model, rank=rank, alpha=alpha)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(text_input, return_tensors="pt").to(device)
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
            top_p=0.95,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("-" * 40)
    print(f"INPUT: {text_input}")
    print(f"OUTPUT: {generated_text}")
    print("-" * 40)


if __name__ == "__main__":
    sample = "name[Blue Spice] || eatType[pub] || food[Chinese] || priceRange[cheap]"
    generate_review(sample)
