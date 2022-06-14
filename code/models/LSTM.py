import torch
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


class LSTM(nn.Module):

    def __init__(self, args, batch_first=True):
        super(LSTM, self).__init__()
        
        self.out_size = args.pred_length
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=batch_first)
        
        self.fc = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Propagate input through LSTM
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out