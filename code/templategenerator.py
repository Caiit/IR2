# Given resources per class, generate template sentences

import torch.nn as nn
import torch.nn.functional as F
import torch

class TemplateGenerator(nn.Module):
    def __init__(self, batch_size, vocabulary_size, embedding_dim, lstm_num_hidden=256, lstm_num_layers=2):
        super(TemplateGenerator, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, embedding_dim)

    def forward(self, x):
        # forward pass
        out = self.lstm(x)
        return self.linear(out[0])
