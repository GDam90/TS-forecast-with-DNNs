import torch.nn as nn


def set_activation_function(args):
    if args.activation == "relu":
        activation = nn.ReLU()
    elif args.activation == "tanh":
        activation = nn.Tanh()
    elif args.activation == "sigmoid":
        activation = nn.Sigmoid()
    elif args.activation == "leakyrelu":
        activation = nn.LeakyReLU()
    elif args.activation == "elu":
        activation = nn.ELU()
    elif args.activation == "selu":
        activation = nn.SELU()
    elif args.activation == "prelu":
        activation = nn.PReLU()
    elif args.activation == "rrelu":
        activation = nn.RReLU()
    elif args.activation == "celu":
        activation = nn.CELU()
    elif args.activation == "gelu":
        activation = nn.GELU()
    return activation


class DensePredictor(nn.Module):
    def __init__(self, args):
        super(DensePredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(args.history_length, args.hiddens[0]))
        self.layers.append(set_activation_function(args))
        for i in range(len(args.hiddens) - 1):
            self.layers.append(nn.Linear(args.hiddens[i], args.hiddens[i + 1]))
            self.layers.append(set_activation_function(args))
        self.layers.append(nn.Linear(args.hiddens[-1], args.pred_length))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        x = x.squeeze()
        x = self.layers(x)
        return x