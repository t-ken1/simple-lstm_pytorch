import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 output_dim, num_layers=1):
        super(self.__class__, self).__init__()

        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.lstm(inputs, hidden0)
        output = self.linear(output)
        return output
