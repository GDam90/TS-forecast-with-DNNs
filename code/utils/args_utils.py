import argparse
import yaml
import torch
import os
from utils.constants import ALL_STORES

def yaml2args(yaml_path="config/config.yaml"):
    args = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    return args

def adjust_args(args):
    if args.model_type == "lstm":
        name = "{}_{}_{}_{}_{}_{}".format(args.experiment_name, args.history_length, args.pred_length, args.normalization_strategy, args.epochs, args.hidden_size)
    elif args.model_type == "dense":
        name = ("{}_{}_{}_"+"{}_"*len(args.hiddens) +"{}").format(args.experiment_name, args.history_length, args.pred_length, *args.hiddens, args.epochs)
    elif args.model_type == "tcn":
        name = ("{}_{}_{}_"+"{}_"*len(args.num_channels) +"{}").format(args.experiment_name, args.history_length, args.pred_length, *args.num_channels, args.epochs)
    elif args.model_type == "transformer":
        name = "{}_{}_{}_{}_{}_{}".format(args.experiment_name, args.history_length, args.pred_length, args.d_model, args.dim_feedforward, args.epochs)
    
    args.path_to_experiment = args.path_to_experiment.format(args.model_type) + "/{}".format(name)
    args.name = name
    args.path_to_checkpoints = {}
    args.path_to_viz = {}
    args.path_to_logs = {}
    args.path_to_ckpt = {}
    for store in args.stores_list:
        args.path_to_checkpoints[store] = args.path_to_experiment + "/{}".format(store)
        args.path_to_viz[store] = os.path.join(args.path_to_checkpoints[store], "viz")
        args.path_to_logs[store] = os.path.join(args.path_to_checkpoints[store], "logs")
        args.path_to_ckpt[store] = os.path.join(args.path_to_checkpoints[store], "ck.pt")

    if os.path.exists(args.path_to_experiment):
        args.pretrained = True
    else:
        args.pretrained = False
    
    if args.stores_list == "all":
        args.stores_list = ALL_STORES    
    # device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    # cpu_device = torch.device("cpu")
    return args
    
def create_folder_structure(args):
    if not os.path.exists(args.path_to_experiment):
        for store in args.stores_list:
            os.makedirs(args.path_to_checkpoints[store])
            os.makedirs(args.path_to_viz[store])
            os.makedirs(args.path_to_logs[store])
    if not os.path.exists(args.path_to_experiment + "/config.yaml"):
        os.system("cp config/config.yaml {}".format(args.path_to_experiment))
        print("config saved to {}".format(args.path_to_experiment + "/config.yaml"))


def generate_exp(yaml_path):
    args = yaml2args(yaml_path)
    args = adjust_args(args)
    create_folder_structure(args)
    return args


if __name__ == "__main__":
    args = generate_exp()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")