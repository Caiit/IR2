import torch
import torch.nn as nn

class SaliencyPrediction(nn.Module):
    def __init__(self, embedding_size, device):
        super(SaliencyPrediction, self).__init__()
        # Maybe extend this further?
        self.bilinear = nn.Bilinear(embedding_size, embedding_size, 1).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, resource_embed, template_embed):
        out = self.bilinear(resource_embed, template_embed)
        return self.sigmoid(out)
