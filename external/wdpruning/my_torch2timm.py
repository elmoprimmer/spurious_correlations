import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("deep_feature_reweighting"), '../..')))
import torch
import re

_TV_LAY_RE = re.compile(r'encoder\.layers\.encoder_layer_(\d+)\.')

def _convert_torchvision_vit(state_dict: dict) -> dict:
    """
    Map TorchVision ViT keys to timm / WD-Pruning ViT keys.
    Only renames – leaves tensor data untouched.
    """
    out = {}
    for k, v in state_dict.items():

        # 1. patch-embedding conv
        if k.startswith('conv_proj.'):
            out[k.replace('conv_proj', 'patch_embed.proj')] = v
            continue

        # 2. position embed
        if k == 'encoder.pos_embedding':
            out['pos_embed'] = v
            continue

        # 3. transformer blocks
        m = _TV_LAY_RE.match(k)
        if m:
            blk = m.group(1)                          # block id 0-11
            rest = k[m.end():]                        # everything after "..._N."
            rest = (rest
                .replace('self_attention.in_proj_weight', 'attn.qkv.weight')
                .replace('self_attention.in_proj_bias',  'attn.qkv.bias')
                .replace('self_attention.out_proj',      'attn.proj')
                .replace('ln_1', 'norm1')
                .replace('ln_2', 'norm2')
                .replace('mlp.0', 'mlp.fc1')
                .replace('mlp.3', 'mlp.fc2'))
            out[f'blocks.{blk}.{rest}'] = v
            continue

        # 4. final norm
        if k.startswith('encoder.ln.'):
            out[k.replace('encoder.ln', 'norm')] = v
            continue

        # 5. classifier head
        if k.startswith('heads.head.'):
            out['head.' + k.split('.', 2)[-1]] = v
            continue

        # 6. anything else (class_token, etc.)
        if k == 'class_token':
            out['cls_token'] = v
            continue

        out[k] = v           # fallback – copy as-is

    return out


tv_ckpt  = torch.load('/home/primmere/logs/isic_logs_4/vit_isic_v2.pt', map_location='cpu')
timm_ckpt = _convert_torchvision_vit(tv_ckpt)
torch.save(timm_ckpt, '/home/primmere/logs/isic_logs_4/vit_isic_v2_timm.pt')