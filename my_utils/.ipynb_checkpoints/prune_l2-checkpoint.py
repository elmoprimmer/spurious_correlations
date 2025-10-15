# L2 pruning for torchvision ViT-B/16
# - Supports two modes:
#     layer_type='mlp'  -> prunes hidden neurons in the MLP (fc1/fc2) of each block
#     layer_type='attn' -> prunes attention heads in each block
# - Rate is a fraction (0 < rate < 1), and we prune the global bottom r neurons/heads.
#
# Usage:
#   import torch
#   from torchvision.models import vit_b_16
#
#   model = vit_b_16(weights=None)  # or with weights if available
#   pruner = L2VitB16Pruner(layer_type='mlp')  # or 'attn'
#   summary = pruner.prune(model, rate=0.2)
#   print(summary)

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

try:
    # torchvision >= 0.13 has these under models.vision_transformer
    from torchvision.models.vision_transformer import VisionTransformer
except Exception:
    VisionTransformer = None  # best-effort guard


class L2VitB16Pruner:
    def __init__(self, layer_type: str = "linear"):
        layer_type = layer_type.lower()
        if layer_type not in {"linear", "attn"}:
            raise ValueError("layer_type must be 'linear' or 'attn'")
        self.layer_type = layer_type

    # ---------- public API ----------
    @torch.no_grad()
    def prune(self, model: nn.Module, rate: float) -> Dict[str, int]:
        """
        Prune the model in-place based on L2 norms.
        - model: torchvision.models.vit_b_16 instance
        - rate: fraction in (0, 1) -> prune the bottom r neurons/heads globally
        Returns a small summary dict.
        """
        self._assert_vit_b_16(model)
        if not (0 < rate < 1):
            raise ValueError("rate must be between 0 and 1 (exclusive).")

        if self.layer_type == "linear":
            return self._prune_linear_neurons(model, rate)
        else:
            return self._prune_attention_heads(model, rate)

    # ---------- checks ----------
    def _assert_vit_b_16(self, model: nn.Module) -> None:
        if VisionTransformer is None or not isinstance(model, VisionTransformer):
            raise TypeError("This pruner only supports torchvision VisionTransformer (vit_b_16).")
        # Verify patch embedding is 16x16 (ViT-B/16)
        if not hasattr(model, "conv_proj"):
            raise TypeError("Unexpected ViT structure (missing conv_proj).")
        k = getattr(model.conv_proj, "kernel_size", None)
        s = getattr(model.conv_proj, "stride", None)
        if k != (16, 16) or s != (16, 16):
            raise TypeError("This pruner is restricted to ViT-B/16 (patch size 16x16).")
        # Sanity-check one attention block looks like ViT-B/16 (12 heads, 768 dim)
        attns = self._collect_attention_modules(model)
        if not attns:
            raise TypeError("Could not find attention modules.")
        attn0 = attns[0]
        if not hasattr(attn0, "num_heads") or not hasattr(attn0, "in_proj_weight") or not hasattr(attn0, "out_proj"):
            raise TypeError("Unexpected attention module layout.")
        embed_dim = attn0.in_proj_weight.shape[0] // 3  # qkv projects to 3*embed_dim
        if embed_dim != 768 or attn0.num_heads != 12:
            raise TypeError("Expected ViT-B/16 (embed_dim=768, num_heads=12).")

    # ---------- helpers: collectors ----------
    def _collect_linear_modules(self, model: nn.Module) -> List[nn.Module]:
        linears = []
        for m in model.modules():
            # torchvision's block MLP class is named "MLP" and exposes fc1/fc2
            if m.__class__.__name__ == "MLPBlock":
                linears.append(m)
        return linears

    def _collect_attention_modules(self, model: nn.Module) -> List[nn.Module]:
        attns = []
        for m in model.modules():
            if m.__class__.__name__ == "MultiheadAttention" and hasattr(m, 'in_proj_weight') and hasattr(m, "out_proj"):
                attns.append(m)
        return attns

    # ---------- Linear neuron pruning ----------
    @torch.no_grad()
    def _prune_linear_neurons(self, model: nn.Module, rate: float) -> Dict[str, int]:
        linears = self._collect_linear_modules(model)
        if not linears:
            raise RuntimeError("No linear modules found.")

        # 1) L2 "mask" (importance scores) for every hidden neuron (global)
        #    For hidden neuron j in a block MLP:
        #      score_j = ||fc1.weight[j, :]||^2 + ||fc2.weight[:, j]||^2 + (fc1.bias[j]^2 if bias)
        entries: List[Tuple[int, int, float]] = []  # (mlp_idx, neuron_idx, score)
        per_mlp_scores: List[torch.Tensor] = []     # store to re-use later
        for mi, mlp in enumerate(linears):
            fc1, fc2 = mlp[0], mlp[3]
            w1 = fc1.weight.data  # (hidden_dim, embed_dim)
            w2 = fc2.weight.data  # (embed_dim, hidden_dim)
            s1 = (w1 * w1).sum(dim=1)                # (hidden_dim,)
            s2 = (w2 * w2).sum(dim=0)                # (hidden_dim,)
            if fc1.bias is not None:
                sb = fc1.bias.data * mlp[0].bias.data # fc1 bias
                scores = s1 + s2 + sb
            else:
                scores = s1 + s2
            per_mlp_scores.append(scores)
            for j, val in enumerate(scores.tolist()):
                entries.append((mi, j, float(val)))

        total_neurons = sum(len(s) for s in per_mlp_scores)
        k = int(math.floor(rate * total_neurons))
        if k <= 0:
            return {"pruned": 0, "total": total_neurons, "type": "mlp", "scores": per_mlp_scores}

        # Sort ascending -> bottom-k to prune
        entries.sort(key=lambda t: t[2])
        to_prune = entries[:k]

        # 2) Binary {1,0} prune mask per MLP
        binary_masks: List[torch.Tensor] = [torch.ones_like(s, dtype=torch.float32) for s in per_mlp_scores]
        for mi, j, _ in to_prune:
            binary_masks[mi][j] = 0.0

        # 3) Apply masks by zeroing fc1 rows / bias and fc2 columns
        for mi, mlp in enumerate(linears):
            mask = binary_masks[mi]  # (hidden_dim,)
            # fc1: zero rows and biases for pruned neurons
            #   fc1.weight: (hidden_dim, embed_dim)
            #   fc1.bias:   (hidden_dim,)
            fc1 = mlp[0]
            fc1.weight.data *= mask.view(-1, 1)
            if fc1.bias is not None:
                fc1.bias.data *= mask

            # fc2: zero corresponding columns for pruned neurons
            #   fc2.weight: (embed_dim, hidden_dim)
            fc2 = mlp[3]
            fc2.weight.data *= mask.view(1, -1)
            # (fc2.bias is per-output embed unit, not per hidden neuron; no change needed)

        return {"pruned": k, "total": total_neurons, "type": "linear", "mask": binary_masks, "scores": per_mlp_scores}

    # ---------- Attention head pruning ----------
    @torch.no_grad()
    def _prune_attention_heads(self, model: nn.Module, rate: float) -> Dict[str, int]:
        attns = self._collect_attention_modules(model)
        if not attns:
            raise RuntimeError("No Attention modules found.")

        # 1) L2 scores per head, globally across all blocks.
        #    For head h: sum of squares from Q, K, V slices in qkv.weight/bias
        #    + sum of squares from the corresponding input columns in proj.weight
        #    (proj.bias is not head-specific, so we leave it)
        entries: List[Tuple[int, int, float]] = []  # (attn_idx, head_idx, score)
        per_attn_scores: List[torch.Tensor] = []
        for ai, attn in enumerate(attns):
            out_proj: nn.Linear = attn.out_proj
            num_heads = 12
            embed_dim = 768
            head_dim = 64

            w_qkv = attn.in_proj_weight.data  # (3*embed_dim)
            b_qkv = attn.in_proj_bias.data
            w_proj = out_proj.weight.data  # (embed_dim_out=embed_dim, embed_dim_in=embed_dim)

            # Build per-head score tensor
            scores = torch.zeros(num_heads, dtype=torch.float32, device=w_qkv.device)

            # row ranges in qkv for Q, K, V
            base_q = 0
            base_k = embed_dim
            base_v = 2 * embed_dim

            for h in range(num_heads):
                q_slice = slice(base_q + h * head_dim, base_q + (h + 1) * head_dim)
                k_slice = slice(base_k + h * head_dim, base_k + (h + 1) * head_dim)
                v_slice = slice(base_v + h * head_dim, base_v + (h + 1) * head_dim)

                # qkv rows (L2 over rows)
                s_q = (w_qkv[q_slice, :] ** 2).sum()
                s_k = (w_qkv[k_slice, :] ** 2).sum()
                s_v = (w_qkv[v_slice, :] ** 2).sum()

                if b_qkv is not None:
                    s_q += (b_qkv[q_slice] ** 2).sum()
                    s_k += (b_qkv[k_slice] ** 2).sum()
                    s_v += (b_qkv[v_slice] ** 2).sum()

                # proj columns corresponding to this head's output features
                #   proj.weight: (embed_dim, embed_dim); columns index input features
                s_proj_cols = (w_proj[:, v_slice] ** 2).sum()

                scores[h] = s_q + s_k + s_v + s_proj_cols

            per_attn_scores.append(scores)
            for h in range(num_heads):
                entries.append((ai, h, float(scores[h].item())))

        total_heads = sum(len(s) for s in per_attn_scores)
        k = int(math.floor(rate * total_heads))
        if k <= 0:
            return {"pruned": 0, "total": total_heads, "type": "attn", "scores": per_attn_scores}

        # Sort ascending -> bottom-k to prune
        entries.sort(key=lambda t: t[2])
        to_prune = entries[:k]

        # 2) Binary {1,0} mask per attention module (per-head)
        binary_masks: List[torch.Tensor] = [torch.ones_like(s, dtype=torch.float32) for s in per_attn_scores]
        for ai, h, _ in to_prune:
            binary_masks[ai][h] = 0.0

        # 3) Apply masks by zeroing:
        #    - qkv: zero rows (Q,K,V slices) & biases for pruned heads
        #    - proj: zero columns (input slice for that head)
        for ai, attn in enumerate(attns):
            head_mask = binary_masks[ai]  # (num_heads,)
            out_proj: nn.Linear = attn.out_proj
            num_heads = 12
            embed_dim = 768
            head_dim = 64

            w_qkv = attn.in_proj_weight.data  # (3*embed_dim)
            b_qkv = attn.in_proj_bias.data
            w_proj = out_proj.weight.data  # (embed_dim_out=embed_dim, embed_dim_in=embed_dim)

            # Build row mask for qkv (length 3*embed_dim) and col mask for proj (length embed_dim)
            row_mask_qkv = torch.ones(3 * embed_dim, dtype=torch.float32, device=head_mask.device)
            col_mask_proj = torch.ones(embed_dim, dtype=torch.float32, device=head_mask.device)

            # Fill masks per head
            base_q = 0
            base_k = embed_dim
            base_v = 2 * embed_dim
            for h in range(num_heads):
                m = head_mask[h]
                q_slice = slice(base_q + h * head_dim, base_q + (h + 1) * head_dim)
                k_slice = slice(base_k + h * head_dim, base_k + (h + 1) * head_dim)
                v_slice = slice(base_v + h * head_dim, base_v + (h + 1) * head_dim)
                row_mask_qkv[q_slice] *= m
                row_mask_qkv[k_slice] *= m
                row_mask_qkv[v_slice] *= m
                col_mask_proj[v_slice] *= m  # input features to proj come from V

            # Apply
            w_qkv *= row_mask_qkv.view(-1, 1)
            if b_qkv is not None:
                b_qkv *= row_mask_qkv
            w_proj *= col_mask_proj.view(1, -1)
            # (proj.bias left untouched)

        return {"pruned": k, "total": total_heads, "type": "attn", "mask": binary_masks, "scores": per_attn_scores}
        
