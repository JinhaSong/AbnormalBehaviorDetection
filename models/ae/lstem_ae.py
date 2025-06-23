import torch
import torch.nn as nn


class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super(LSTMAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim)

        batch_size = x.size(0)
        seq_len = x.size(1)

        h0 = torch.zeros(self.encoder.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.encoder.num_layers, batch_size, self.hidden_dim).to(x.device)

        encoded, (hn, cn) = self.encoder(x, (h0, c0))  # (batch_size, 1, hidden_dim)
        encoded = hn[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, 1, hidden_dim)

        h0 = torch.zeros(self.decoder.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.decoder.num_layers, batch_size, self.hidden_dim).to(x.device)

        decoded_output, _ = self.decoder(encoded, (h0, c0))
        decoded_output = self.hidden_to_output(decoded_output)  # (batch_size, 1, input_dim)
        decoded_output = decoded_output.squeeze(1)  # (batch_size, input_dim)
        return decoded_output