
import torch
import torch.nn as nn

class Net(nn.Module):
    """Multi-layer neural net."""
    def __init__(self, input_size, output_sizes):
        super(Net, self).__init__()
        self.l_relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else output_sizes[i - 1], output_size)
            for i, output_size in enumerate(output_sizes)
        ])

    def forward(self, inp):
        x = inp
        for lin in self.layers:
            x = self.l_relu(lin(x))

        return x
