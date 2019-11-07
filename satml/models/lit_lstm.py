
import torch
import torch.nn as nn

from satml.models.net import Net


class LitLSTM(nn.Module):
    """A bidirectional LSTM encoder of literals, followed by a MLP classifier on each."""
    def __init__(self, input_dim, hidden_dim, linear_outputs, num_lstm_layers):
        super(LitLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True)

        # Times 2 because bidirectional LSTM.
        self.net = Net(hidden_dim * 2, linear_outputs)

    def forward(self, padded_sequences):
        """Expects a list of padded sequences."""
        output, _ = self.lstm(padded_sequences)
        batch_size, seq_len, _ = output.shape
    
        projected = self.net(output.reshape(batch_size * seq_len, -1))
        return projected.reshape(batch_size, seq_len)
