import torch.nn as nn
from .lora import LoRAConv1D

def inject_lora(model, rank=4, alpha=32):
    """
    Replaces specific layers in the GPT-2 model with LoRA layers.
    """
    for name, module in model.named_modules():
        # The paper targets attention projection weights.
        # In HF GPT-2, the attention projection is named 'c_attn'.
        if name.endswith('c_attn'):
            # We access the parent module to replace the child
            parent_name = name.rsplit('.', 1)[0]
            node = model
            for part in parent_name.split('.'):
                node = getattr(node, part)
            
            # Replace the layer
            original_layer = getattr(node, 'c_attn')
            lora_layer = LoRAConv1D(original_layer, rank=rank, alpha=alpha)
            setattr(node, 'c_attn', lora_layer)
            
            print(f"Replaced {name} with LoRA layer (r={rank})")
            
    return model

def print_trainable_parameters(model):
    """
    Helper to verify the parameter reduction
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")