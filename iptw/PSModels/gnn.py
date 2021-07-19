import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GraphConv(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(GraphConv, self).__init__()
        # input and output dimension should be same in my case
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, input, G):
        # support = self.linear(input)
        # # CAUTION: Pytorch only supports sparse * dense matrix multiplication
        # # CAUTION: Pytorch does not support sparse * sparse matrix multiplication !!!
        # output = torch.sparse.mm(propagation_adj, support)
        #
        output = input
        output = torch.matmul(output, G)
        output = self.linear(output)

        return output


class RESGNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0):
        super(RESGNN, self).__init__()

        self.layers = nn.ModuleList([])

        if num_layers == 1:
            self.layers.append(GraphConv(input_size, input_size))
            self.outlayer = nn.Linear(input_size, num_classes)
        elif num_layers > 1:
            self.layers.append(GraphConv(input_size, hidden_size))
            for i in range(num_layers-2):
                self.layers.append(GraphConv(hidden_size, hidden_size))
            self.layers.append(GraphConv(hidden_size, hidden_size))
            self.outlayer = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, x, G):
        for i in range(self.num_layers-1):
            residual = x
            x = self.layers[i].forward(x, G)
            x = F.relu(x)
            x = residual + x*self.t
            x = self.dropout(x)

        residual = x
        x = self.layers[-1].forward(x, G)
        x = residual + x*self.t
        x = self.outlayer(x)
        x = self.dropout(x)

        return x


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0):
        super(GNN, self).__init__()

        self.layers = nn.ModuleList([])
        if num_layers == 1:
            self.layers.append(GraphConv(input_size, num_classes))
        elif num_layers > 1:
            self.layers.append(GraphConv(input_size, hidden_size))
            for i in range(num_layers-2):
                self.layers.append(GraphConv(hidden_size, hidden_size))
            self.layers.append(GraphConv(hidden_size, num_classes))
        else:
            raise ValueError

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x, G):

        for i in range(self.num_layers-1):
            x = self.layers[i].forward(x, G)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1].forward(x, G)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([])
        if num_layers == 1:
            self.layers.append(nn.Linear(input_size, num_classes))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_size, hidden_size))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Linear(hidden_size, num_classes))
        else:
            raise ValueError

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x, G):

        for i in range(self.num_layers - 1):
            x = self.layers[i].forward(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1].forward(x)
        x = self.dropout(x)

        return x


# class GNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, dropout=0, num_middle_layers=0):
#         super(GNN, self).__init__()
#
#         self.gc1 = GraphConv(input_size, num_classes)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, G):
#         x = self.gc1.forward(x, G)
#         # x = F.relu(x)
#         x = self.dropout(x)
#
#         return x


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, dropout=0, num_middle_layers=0):
#         super(MLP, self).__init__()
#
#         self.gc1 = nn.Linear(input_size, num_classes)
#         self.dropout = nn.Dropout(dropout)
#         self.hidden_size = hidden_size
#         self.num_middle_layers = num_middle_layers
#
#     def forward(self, x, G):
#         x = self.gc1.forward(x)  #, G)
#         # x = F.relu(x)
#         x = self.dropout(x)
#
#         return x