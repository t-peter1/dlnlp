import torch

# 1. Load your huge 1.5GB file
print("Loading big model file...")
full_state_dict = torch.load("lora_weights_only.pt")

# 2. Filter out everything except the LoRA weights
print("Extracting LoRA weights...")
lora_only = {k: v for k, v in full_state_dict.items() if "lora" in k}

# 3. Save the small file
print(f"Original keys: {len(full_state_dict)} -> LoRA keys: {len(lora_only)}")
torch.save(lora_only, "lora_weights_only_fixed.pt")

print("Done! Check the file size of 'lora_weights_only_fixed.pt'")