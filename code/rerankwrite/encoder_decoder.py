import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden, length):
        output, hidden = self.gru(input, hidden)
        print("output before gather", output.shape)
        output = torch.gather(output, 1, torch.tensor(length).to(self.device).expand(1, 1,
                              self.hidden_size)-1)
        print("output after gather", output.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.3, max_length=110):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        output = input
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class CreateResponse(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p, max_length, device):
        super(CreateResponse, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, device).to(device)
        self.decoder = DecoderRNN(hidden_size, output_size, device, dropout_p, max_length).to(device)
