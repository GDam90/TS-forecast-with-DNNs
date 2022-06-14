import torch
import torch.nn as nn
from torch.nn.utils import weight_norm



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



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, args):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = set_activation_function(args)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = set_activation_function(args)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = set_activation_function(args)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(args.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = args.input_size if i == 0 else args.num_channels[i-1]
            out_channels = args.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, args.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(args.kernel_size-1) * dilation_size, dropout=args.dropout, args=args)]
        
        self.linear = nn.Linear(args.history_length, 5)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        self.network(x)
        x = x.reshape(x.size(0), -1)
        return self.linear(x)