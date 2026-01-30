import math
import torch
import torch.nn as nn


class AdaLoRAConv1D(nn.Module):
    """
    Explicit SVD-form adapter: ΔW = P Λ Q
      P: [out_features, r]
      Λ: [r] (trainable singular values / gates)
      Q: [r, in_features]
    """

    def __init__(self, original_layer, rank=4, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        self.in_features = original_layer.weight.shape[0]
        self.out_features = original_layer.weight.shape[1]

        # SVD-form parameters
        self.Q = nn.Parameter(torch.zeros((rank, self.in_features)))   # right singular vectors
        self.P = nn.Parameter(torch.zeros((self.out_features, rank)))  # left singular vectors
        self.lambda_vals = nn.Parameter(torch.full((rank,), 0.1))      # singular values (gates; small but nonzero)
        self.register_buffer("rank_mask", torch.ones(rank))            # binary mask for irreversible pruning

        # Scaling
        self.scaling = self.alpha / self.rank

        # Initialization
        nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        nn.init.constant_(self.lambda_vals, 0.1)

        # Freeze base weights
        self.original_layer.weight.requires_grad = False
        self.original_layer.bias.requires_grad = False

    def forward(self, x):
        # x: [batch, seq, in_features]
        if x.size(-1) != self.in_features:
            raise ValueError(f"Input hidden size {x.size(-1)} != expected {self.in_features}")

        base_out = self.original_layer(x)

        # ΔW = P Λ Q ; apply Λ once
        h = torch.matmul(x, self.Q.transpose(-1, -2))                  # [b, s, r]
        gate = self.lambda_vals * self.rank_mask                       # [r]
        h = h * gate.view(1, 1, -1)                                    # [b, s, r]
        update = torch.matmul(h, self.P.transpose(-1, -2))             # [b, s, out]

        return base_out + update * self.scaling

    def get_importance(self):
        # importance per rank i: ||P[:, i]||_1 * ||Q[i, :]||_1
        p_score = self.P.abs().sum(dim=0)      # [r]
        q_score = self.Q.abs().sum(dim=1)      # [r]
        return p_score * q_score

    def update_mask(self, new_lambda):
        if new_lambda.numel() != self.lambda_vals.numel():
            raise ValueError("new_lambda must match current rank dimension")
        self.lambda_vals.data.copy_(new_lambda.to(self.lambda_vals.device).float())

    def prune_by_indices(self, idxs_to_zero):
        if not idxs_to_zero:
            return
        mask = self.rank_mask.data
        mask[idxs_to_zero] = 0.0
        self.rank_mask.data.copy_(mask)

    def svd_compress(self, target_rank: int):
        """
        Optional offline compression; truncates to target_rank via SVD on the current delta.
        """
        with torch.no_grad():
            delta_w = self.P @ torch.diag(self.lambda_vals) @ self.Q   # [out, in]
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
            k = min(target_rank, S.numel())
            self.rank = k
            self.P = nn.Parameter(U[:, :k].contiguous())               # [out, k]
            self.lambda_vals = nn.Parameter(S[:k].contiguous())       # [k]
            self.Q = nn.Parameter(Vh[:k, :].contiguous())             # [k, in]
            self.scaling = self.alpha / k
            assert self.Q.shape[0] == self.rank
