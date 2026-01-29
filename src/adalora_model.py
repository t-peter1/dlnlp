import torch.nn as nn
from .adalora_lora import AdaLoRAConv1D


def inject_adalora(model, rank=4, alpha=32):
    """
    Replace GPT-2 attention projection layers (c_attn) with AdaLoRAConv1D.
    Returns the modified model and a list of injected AdaLoRA modules.
    """
    injected = []

    for name, module in model.named_modules():
        if name.endswith("c_attn"):
            parent_name = name.rsplit(".", 1)[0]
            node = model
            for part in parent_name.split("."):
                node = getattr(node, part)

            original_layer = getattr(node, "c_attn")
            lora_layer = AdaLoRAConv1D(original_layer, rank=rank, alpha=alpha)
            setattr(node, "c_attn", lora_layer)
            injected.append(lora_layer)

    return model, injected
