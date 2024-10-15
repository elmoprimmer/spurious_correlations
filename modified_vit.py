import torch.nn as nn
import torch
class ModifiedViT(nn.Module):
    def __init__(self, original_model):
        super(ModifiedViT, self).__init__()
        self.model = original_model

    def forward(self, x, head_mask=None):
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.model.class_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)

        # Pass through all but the last encoder layers
        for layer in self.model.encoder.layers[:-1]:
            x = layer(x)

        # Pass through the modified last encoder layer
        x = self.model.encoder.layers[-1](x, head_mask=head_mask)
        x = self.model.encoder.ln(x)
        return x