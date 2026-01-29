import math
import torch
import torch.nn as nn


class AdaLoRAConv1D(nn.Module):
    """
    LoRA variant with rank masking to support AdaLoRA.
    Mirrors GPT-2 Conv1D wrapping style used in src/lora.py.
    """

    def __init__(self, original_layer, rank=4, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # GPT-2 Conv1D weight shape: [in_features, out_features]
        self.in_features = original_layer.weight.shape[0]
        self.out_features = original_layer.weight.shape[1]

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros((self.in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, self.out_features)))

        # Mask buffer (1 keeps direction active, 0 prunes it)
        self.register_buffer("rank_mask", torch.ones(rank))

        # Scaling
        self.scaling = self.alpha / self.rank

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base weights
        self.original_layer.weight.requires_grad = False
        self.original_layer.bias.requires_grad = False

    def forward(self, x):
        # Base path
        base_out = self.original_layer(x)

        # Apply masks to LoRA factors
        mask_col = self.rank_mask.view(1, -1)  # for columns of A
        mask_row = self.rank_mask.view(-1, 1)  # for rows of B

        lora_a = self.lora_A * mask_col
        lora_b = self.lora_B * mask_row

        lora_hidden = x @ lora_a
        lora_update = lora_hidden @ lora_b

        return base_out + (lora_update * self.scaling)

    def get_importance(self):
        """
        Simple magnitude-based importance per rank direction.
        importance[i] = ||A[:, i]||_1 * ||B[i, :]||_1
        """
        a_score = self.lora_A.abs().sum(dim=0)
        b_score = self.lora_B.abs().sum(dim=1)
        return a_score * b_score

    def update_mask(self, new_mask):
        """
        Replace mask with a new tensor (float, same shape).
        """
        if new_mask.numel() != self.rank_mask.numel():
            raise ValueError("new_mask must match current rank dimension")
        self.rank_mask.data.copy_(new_mask.to(self.rank_mask.device).float())

    def prune_by_indices(self, idxs_to_zero):
        """
        Zero out specified rank directions (by index) in the mask.
        """
        if not idxs_to_zero:
            return
        mask = self.rank_mask.clone()
        mask[idxs_to_zero] = 0.0
        self.rank_mask.data.copy_(mask)

    def svd_compress(self, target_rank: int):
        with torch.no_grad():
            mask_col = self.rank_mask.view(1, -1)
            mask_row = self.rank_mask.view(-1, 1)
    
            A = self.lora_A * mask_col          # [in, r]
            B = self.lora_B * mask_row          # [r, out]
    
            delta_w = A @ B                     # [in, out]
    
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    
            k = min(target_rank, S.numel())
            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]
    
            sqrt_S = torch.diag(torch.sqrt(S_k))
    
            new_A = U_k @ sqrt_S                # [in, k]
            new_B = sqrt_S @ Vh_k               # [k, out]
    
            # Replace parameters
            self.rank = k
            self.lora_A = nn.Parameter(new_A.contiguous())
            self.lora_B = nn.Parameter(new_B.contiguous())
    
            # IMPORTANT: replace buffer correctly
            if "rank_mask" in self._buffers:
                del self._buffers["rank_mask"]
    
            self.register_buffer(
                "rank_mask",
                torch.ones(k, device=self.lora_A.device),
            )
    
            self.scaling = self.alpha / k

            assert self.lora_A.shape[1] == self.lora_B.shape[0] == self.rank_mask.numel()

