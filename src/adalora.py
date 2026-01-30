import math
from typing import List, Optional

import torch


class AdaLoRAController:
    """
    Minimal AdaLoRA-style rank allocator using magnitude importance.
    """

    def __init__(
        self,
        modules: List,
        r_init_total: int,
        r_target_total: int,
        update_interval: int = 100,
        warmup_ratio: float = 0.3,
        svd_interval=1000,
        svd_start_ratio=0.5
    ):
        self.modules = modules
        self.r_init_total = r_init_total
        self.r_target_total = r_target_total
        self.update_interval = max(1, update_interval)
        self.warmup_ratio = max(0.0, min(1.0, warmup_ratio))
        self.svd_interval = svd_interval
        self.svd_start_ratio = svd_start_ratio

    def _current_budget(self) -> float:
        # Count active directions by mask
        total = 0
        for m in self.modules:
            total += int((m.rank_mask > 0).sum().item())
        return float(total)

    def _target_budget(self, step: int, total_steps: Optional[int]) -> float:
        if total_steps is None or total_steps <= 0:
            return float(self.r_target_total)

        warmup_steps = int(total_steps * self.warmup_ratio)
        if step <= warmup_steps:
            return float(self.r_init_total)

        remaining = max(total_steps - warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / remaining)
        return float(
            self.r_init_total - (self.r_init_total - self.r_target_total) * progress
        )

    def step_end(self, global_step: int, total_steps: Optional[int] = None):
        if global_step == 0 or global_step % self.update_interval != 0:
            return

        target_budget = self._target_budget(global_step, total_steps)
        current_budget_before = self._current_budget()

        if current_budget_before <= target_budget + 1e-6:
            return

        prune_count = int(round(current_budget_before - target_budget))
        if prune_count <= 0:
            return

        candidates = []
        for mod in self.modules:
            importance = mod.get_importance().detach().cpu()
            mask = mod.rank_mask.detach().cpu()
            for idx, (imp, mval) in enumerate(zip(importance, mask)):
                if mval <= 0:
                    continue
                candidates.append((float(imp), mod, idx))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0])
        to_prune = candidates[:prune_count]

        grouped = {}
        for _, mod, idx in to_prune:
            grouped.setdefault(mod, []).append(idx)

        for mod, idxs in grouped.items():
            mod.prune_by_indices(idxs)

        # ---- Safe, informative logging (only when pruning happens) ----
        if total_steps is not None and total_steps > 0:
            warmup_steps = int(total_steps * self.warmup_ratio)
            warmup_over = global_step > warmup_steps
        else:
            warmup_steps = None
            warmup_over = None

        current_budget_after = self._current_budget()
        print(
            f"[adalora] step={global_step} "
            f"warmup_over={warmup_over} "
            f"target_budget={target_budget:.1f} "
            f"budget_before={current_budget_before:.1f} "
            f"budget_after={current_budget_after:.1f} "
            f"pruned={len(to_prune)}"
        )

    def maybe_svd_compress(self, global_step, total_steps):
        if total_steps is None:
            return
    
        progress = global_step / total_steps
        if progress < self.svd_start_ratio:
            return
    
        if global_step % self.svd_interval != 0:
            return
    
        for mod in self.modules:
            current_rank = int((mod.rank_mask > 0).sum().item())
            target_rank = max(1, current_rank - 1)
            mod.svd_compress(target_rank)
