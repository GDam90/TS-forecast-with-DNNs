import sys
import os
sys.path.append('/home/zeus/guide/wp3/code')
from utils.constants import PATH_TO_CONFIG
from utils.args_utils import generate_exp
from utils.data_utils import generate_store_datasets
from utils.model_utils import get_model_from_args, get_optimizer_and_loss
from utils.viz_utils import plot_test_predictions
from utils.eval_utils import eval_epoch
from utils.train_utils import train_epoch
from utils.eval_utils import save_results


if __name__ == "__main__":
    args = generate_exp(PATH_TO_CONFIG)
    data = generate_store_datasets(args)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    print()
    for store in args.stores_list:
        model = get_model_from_args(args)
        optimizer, loss = get_optimizer_and_loss(args, model)
        print("training store: {}".format(store))
        best_loss = float("inf")
        for epoch in range(args.epochs):
            
            train_mean_loss, model, optimizer = train_epoch(model, data, store, args, epoch, loss, optimizer)
            
            if (epoch+1) % args.eval_interval == 0:
                best_loss = eval_epoch(model, data, store, args, epoch, loss, best_loss)
            
            if (epoch+1) % args.plot_interval == 0:
                plot_test_predictions(data, store, model, args, epoch)
    
        save_results(data, store, args, epoch)
        print("saved results in xlsx file")