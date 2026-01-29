import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.adalora_model import inject_adalora
from src.adalora_lora import AdaLoRAConv1D



# --- CONFIG ---
MODEL_ID = "gpt2-medium"
CKPT_PATH = "adalora_checkpoint.pt"
RANK_INIT = 4
ALPHA = 32

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_review(text_input):
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    from src.adalora_model import AdaLoRAConv1D
    model, _ = inject_adalora(model, rank=RANK_INIT, alpha=ALPHA)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, AdaLoRAConv1D):
                mask_key = f"{name}.rank_mask"
                if mask_key in state_dict:
                    final_rank = state_dict[mask_key].shape[0]
                    
                    if final_rank != module.rank:
                        in_features = module.original_layer.nx
                        out_features = module.original_layer.nf
                        
                        module.lora_A = nn.Parameter(torch.empty(in_features, final_rank))
                        module.lora_B = nn.Parameter(torch.empty(final_rank, out_features))
                        module.rank_mask = nn.Parameter(torch.empty(final_rank))
                        module.rank = final_rank

    model.load_state_dict(state_dict, strict=False)

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
