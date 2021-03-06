import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.3,
                 max_length=110):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, (self.max_length * 2) + 4)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.pre_att = nn.Linear(output_size, hidden_size)
        self.out_probs = nn.Linear(output_size, 9775)

    def forward(self, input, hidden, encoder_outputs):
        output = self.dropout(input)
        output = self.pre_att(output)
        attn_weights = F.softmax(
            self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((output[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])

        output = self.out_probs(output)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class CreateResponse(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p,
                 max_length, device):
        super(CreateResponse, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, device).to(device)
        self.decoder = DecoderRNN(hidden_size, output_size, device, dropout_p,
                                  max_length).to(device)
