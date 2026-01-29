import torch
from transformers import GPT2LMHeadModel
from src.adalora_model import inject_adalora

CKPT_PATH = "adalora_checkpoint.pt"
MODEL_ID = "gpt2-medium"

print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

print("Top-level type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys())[:20])

# Handle both dict formats
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    lora_state = ckpt["state_dict"]
    meta = ckpt.get("meta", {})
else:
    lora_state = ckpt
    meta = {}

print("Meta:", meta)

print("Loading base model...")
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

# Use rank from meta if available, otherwise fall back
r_init = meta.get("rank_init", meta.get("r_init", 4))
alpha = meta.get("alpha", 32)

print(f"Injecting AdaLoRA (r_init={r_init}, alpha={alpha})...")
model, adalora_layers = inject_adalora(model, rank=r_init, alpha=alpha)

print("Loading LoRA + mask state...")
model.load_state_dict(lora_state, strict=False)

# Inspect ranks
active_ranks = []
for i, layer in enumerate(adalora_layers):
    rm = layer.rank_mask.detach().cpu()
    active = int((rm > 0).sum().item())
    active_ranks.append(active)
    print(f"Layer {i:02d}: active rank = {active}")

print("\nSummary:")
print(f"Num AdaLoRA layers: {len(active_ranks)}")
print(f"Min rank: {min(active_ranks)}")
print(f"Mean rank: {sum(active_ranks)/len(active_ranks):.2f}")
print(f"Max rank: {max(active_ranks)}")

mask_keys = [k for k in lora_state.keys() if "rank_mask" in k]
print("rank_mask keys:", len(mask_keys))
print("example:", mask_keys[:3])