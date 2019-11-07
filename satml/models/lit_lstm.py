
import torch
import torch.nn as nn

from satml.models.net import Net


class LitLSTM(nn.Module):
    """A bidirectional LSTM encoder of literals, followed by a MLP classifier on each."""
    def __init__(self, input_dim, hidden_dim, linear_outputs, num_lstm_layers, max_label):
        super(LitLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_label = max_label
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True)

        # Times 2 because bidirectional LSTM.
        self.net = Net(hidden_dim * 2, linear_outputs)

    def forward(self, inp):
        """Expects a list of padded sequences."""
        padded_sequences, var_mappings = inp

        # Generate LSTM embeddings.
        output, _ = self.lstm(padded_sequences)
        batch_size, seq_len, _ = output.shape

        # Unpack the LSTM outputs. (Assuming that the variable names are normalized and ordered.)
        embedding_matrix = torch.zeros((batch_size, self.max_label, self.hidden_dim * 2))
        if torch.cuda.is_available():
            embedding_matrix = embedding_matrix.cuda()

        # For each formula, pack into embedding matrix
        for i, var_mapping in enumerate(var_mappings):
            # Read the embeddings of each variable
            for var, occurrence in var_mapping:
                # i is the batch index
                embedding = output[i, occurrence]

                # Find the embedding matrix offset
                offset = abs(var) - 1
                if var < 0:
                    offset += self.max_label / 2

                embedding_matrix[i, int(offset)] = embedding
    
        # Project each embedding to a score
        projected = self.net(embedding_matrix.reshape(batch_size * self.max_label, -1))
        return projected.reshape(batch_size, self.max_label)
