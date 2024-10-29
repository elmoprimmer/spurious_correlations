import torch
import torch.nn as nn
import torch.nn.functional as F


class LastEncoderLayer(nn.Module):
    def __init__(self, original_layer):
        super(LastEncoderLayer, self).__init__()

        # norms
        self.attn_weights = None
        self.ln_1 = nn.LayerNorm(normalized_shape=original_layer.ln_1.normalized_shape,
                                 eps=original_layer.ln_1.eps)
        self.ln_2 = nn.LayerNorm(normalized_shape=original_layer.ln_2.normalized_shape,
                                 eps=original_layer.ln_2.eps)

        # attn block
        self.self_attention = nn.MultiheadAttention(embed_dim=original_layer.self_attention.embed_dim,
                                                    num_heads=original_layer.self_attention.num_heads,
                                                    dropout=original_layer.self_attention.dropout,
                                                    batch_first=original_layer.self_attention.batch_first
                                                    )

        # mlp block
        self.mlp = nn.Sequential(
            nn.Linear(original_layer.mlp[0].in_features, original_layer.mlp[0].out_features),
            nn.GELU(),
            nn.Dropout(p=original_layer.mlp[2].p),
            nn.Linear(original_layer.mlp[0].out_features, original_layer.mlp[0].in_features),
            nn.Dropout(p=original_layer.mlp[4].p)
        )

        # dropout (p = 0 tho??)
        self.dropout = nn.Dropout(p=original_layer.dropout.p, inplace=False)

        #
        self._initialize_weights(original_layer)

    def _initialize_weights(self, original_layer):
        # norms
        self.ln_1.weight.data.copy_(original_layer.ln_1.weight.data)
        self.ln_1.bias.data.copy_(original_layer.ln_1.bias.data)
        self.ln_2.weight.data.copy_(original_layer.ln_2.weight.data)
        self.ln_2.bias.data.copy_(original_layer.ln_2.bias.data)

        # attn
        self.self_attention.in_proj_weight.data.copy_(original_layer.self_attention.in_proj_weight.data)
        self.self_attention.in_proj_bias.data.copy_(original_layer.self_attention.in_proj_bias.data)
        self.self_attention.out_proj.weight.data.copy_(original_layer.self_attention.out_proj.weight.data)
        self.self_attention.out_proj.bias.data.copy_(original_layer.self_attention.out_proj.bias.data)

        # mlp
        self.mlp[0].weight.data.copy_(original_layer.mlp[0].weight.data)
        self.mlp[0].bias.data.copy_(original_layer.mlp[0].bias.data)
        self.mlp[3].weight.data.copy_(original_layer.mlp[3].weight.data)
        self.mlp[3].bias.data.copy_(original_layer.mlp[3].bias.data)

    def forward(self, x, head_mask=None):
        x_residual = x
        x = self.ln_1(x)

        batch_size, seq_len, embed_dim = x.size()
        num_heads = self.self_attention.num_heads
        head_dim = embed_dim // num_heads

        # compute and reshape q,k,v
        qkv = F.linear(x, self.self_attention.in_proj_weight, self.self_attention.in_proj_bias)

        qkv = qkv.view(batch_size, seq_len, 3, embed_dim)
        qkv = qkv.permute(2, 0, 1, 3)  # Shape: (3, batch_size, seq_len, embed_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 3x (batch_size, seq_len, embed_dim)


        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # scaled dot product attention per head
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # attention outputs per head
        attn_output_per_head = torch.matmul(attn_probs, v)  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # maybe apply head mask
        if head_mask is not None:
            # head_mask shape should be (num_heads,)
            attn_output_per_head = attn_output_per_head * head_mask.view(1, -1, 1, 1)


        # concatenate heads
        attn_out = attn_output_per_head.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_out = self.self_attention.out_proj(attn_out)

        _, self.attn_weights = self.self_attention(
            x, x, x,
            need_weights=True,
            average_attn_weights=False)




        x = self.dropout(attn_out)
        x = x_residual + x

        x_residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x_residual + x

        return x

