
import torch
import torch.nn as nn

from satml.models.net import Net


class LSTM(nn.Module):
    """An LSTM sequence encoder followed by a MLP classifier."""
    def __init__(self, input_dim, hidden_dim, linear_outputs, num_lstm_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_lstm_layers, batch_first=True)

        self.net = Net(hidden_dim, linear_outputs)

    def forward(self, inp):
        """Expects a list of padded sequences."""
        padded_sequences, _ = inp

        _, (hidden, _) = self.lstm(padded_sequences)
        return self.net(hidden[-1])
