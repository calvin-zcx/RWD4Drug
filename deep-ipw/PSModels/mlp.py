import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=[], dropout=0):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([])
        if (hidden_size == 0) or (hidden_size == ''):
            hidden_size = []
        elif isinstance(hidden_size, int):
            hidden_size = [hidden_size,]
        layer_dims = [input_size, ] + hidden_size + [num_classes, ]
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.layers)- 1):
            x = self.layers[i].forward(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1].forward(x)
        x = self.dropout(x)

        return x
