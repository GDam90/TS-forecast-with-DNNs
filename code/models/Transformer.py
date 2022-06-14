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


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.linear = nn.Linear(args.input_size, args.d_model)
        self.transformer = nn.Transformer(d_model=args.d_model, nhead=args.n_head, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                                          num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers, activation=args.activation)
        self.activation = set_activation_function(args)
        self.linearhead = nn.Linear(args.d_model, args.input_size)
    def forward(self, x, y):
        msk = self.transformer.generate_square_subsequent_mask(y.shape[1])
        x = self.linear(x)
        y = self.linear(y)
        
        x = self.activation(x)
        x = x.transpose(0,1)
        y = y.transpose(0,1)
        x = self.transformer(x, y, tgt_mask=msk)
        x = self.activation(x)
        x = self.linearhead(x)
        x = x.transpose(0,1)
        return x.squeeze()