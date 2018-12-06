import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, out_size, hidden_size, num_layers=2, drop_out=0.3,
                batch_size=1, out_size_bi=1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = drop_out,
                            bidirectional = True)

        self.bilinear =  nn.Bilinear(hidden_size, hidden_size, out_size_bi)
        self.sigmoid = nn.Sigmoid()

    def forward(self, template, resource, hidden_t, hidden_r):
        hidden_rs = torch.zeros(self.batch_size, resource.shape[0], self.hidden_size)
        hidden_ts = torch.zeros(self.batch_size, resource.shape[0], self.hidden_size)
        #output_rs = torch.zeros(self.batch_size, resource.shape[0], self.hidden_size)
        #output_ts = torch.zeros(self.batch_size, resource.shape[0], self.hidden_size)
        for i in range(resource.shape[0]):
            # Unsqueeze enzo weg als
            output_t, (hidden_t, _) = self.lstm(template[i], hidden_t)
            output_r, (hidden_r, _) = self.lstm(resource[i], hidden_r)
            hidden_rs[i] = hidden_r
            hidden_ts[i] = hidden_t
            #output_rs[i] =  output_r
            #output_ts[i] = output_t

        #enc_out = torch.cat((output_rs, output_ts), dim=2)
        h_c = torch.cat((hidden_rs, hidden_ts), dim=2)
        saliency = self.sigmoid(self.bilinear(hidden_ts, hidden_rs))

        return h_c, saliency

    def initHidden(self):
        return torch.zeros(self.batch_size, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.3, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, size):
        all_outs = torch.zeros(input.shape[0], hidden.shape[1])
        for i in range(input.shape[0]):
            embedded = self.dropout(input[i])
            attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights, encoder_outputs)

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output)

            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

            output = F.log_softmax(self.out(output[0]), dim=1)
            all_outs[i] = hidden # nog iets met topk voor de prediction?
        return all_outs, output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
