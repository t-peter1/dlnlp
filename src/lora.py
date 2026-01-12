import torch
import torch.nn as nn
import math

class LoRAConv1D(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=32):
        """
        Wraps a Hugging Face GPT-2 Conv1D layer with LoRA.
        
        Args:
            original_layer: The frozen Conv1D layer from GPT-2
            rank (int): The rank 'r' from the paper
            alpha (int): The scaling factor 'alpha'
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Dimensions for GPT-2 Conv1D: weight is [hidden_size, out_features]
        # (Note: This is transposed compared to standard nn.Linear)
        self.in_features = original_layer.weight.shape[0]
        self.out_features = original_layer.weight.shape[1]
        
        # 1. Initialize LoRA matrices A and B
        # Matrix A: [in_features, r]
        # Matrix B: [r, out_features]
        self.lora_A = nn.Parameter(torch.zeros((self.in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, self.out_features)))
        
        # Scaling constant
        self.scaling = self.alpha / self.rank
        
        # 2. Initialize weights
        # The paper initializes A with random Gaussian, B with zero.
        # This ensures the LoRA update starts as 0 (no change to model behavior).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 3. Freeze the original layer
        self.original_layer.weight.requires_grad = False
        self.original_layer.bias.requires_grad = False

    def forward(self, x):
        # x shape: [batch_size, seq_len, in_features]
        
        # 1. Compute original output (Frozen)
        original_output = self.original_layer(x)
        
        # 2. Compute LoRA path: x @ A @ B * scaling
        # We process the matrix multiplication in steps for clarity
        # Step 1: x @ A -> [batch, seq, rank]
        lora_hidden = x @ self.lora_A 
        
        # Step 2: hidden @ B -> [batch, seq, out_features]
        lora_update = lora_hidden @ self.lora_B
        
        # 3. Combine
        return original_output + (lora_update * self.scaling)