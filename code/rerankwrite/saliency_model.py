import torch
import torch.nn as nn

class SaliencyPrediction():
    def __init__(self, embedding_size):
        # Maybe extend this further? 
        self.bilinear = nn.Bilinear(embedding_size, embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(resource_embed, template_embed):
        out = self.bilinear(resource_embed, template_embed)
        return self.sigmoid(out)
