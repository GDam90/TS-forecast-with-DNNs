import torch.nn as nn
import torch

from models.TCN import TCN
from models.Transformer import Transformer
from models.DensePredictor import DensePredictor as MLP
from models.LSTM import LSTM

from utils.args_utils import generate_exp
from utils.constants import PATH_TO_CONFIG

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

def get_model_from_args(args):
    if args.model_type == "dense":
        model = MLP(args)
    elif args.model_type == "lstm":
        model = LSTM(args)
    elif args.model_type == "tcn":
        model = TCN(args)
    elif args.model_type == "transformer":
        model = Transformer(args)
    else:
        raise ValueError("Model type not recognized")
    return model

def load_state_dict(model, args, store):
    model.load_state_dict(torch.load(args.path_to_ckpt[store]))
    return model

def get_optimizer_and_loss(args, model):
    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.criterion == "mse":
        criterion = nn.MSELoss()
    
    return optim, criterion

def get_pretrained_model(path_to_args=PATH_TO_CONFIG, store="ABLA"):
    args = generate_exp(path_to_args)
    model = get_model_from_args(args)
    if args.pretrained:
        model.load_state_dict(torch.load(args.path_to_ckpt[store]))
        print("Loaded pretrained model")
    return model