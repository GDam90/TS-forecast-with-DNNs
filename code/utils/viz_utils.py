import matplotlib.pyplot as plt
import numpy as np


def plot_test_predictions(data, store, model, args, epoch):
    values = np.zeros([len(data[store]["test_data"])])
    preds = np.zeros([len(data[store]["test_data"])])

    for i in range(0, len(data[store]["test_dataset"]), args.pred_length):
        x, y = data[store]["test_dataset"][i]
        if args.model_type == "transformer":
            y_hat = model(x.unsqueeze(0), y.unsqueeze(0))
        else:
            y_hat = model(x.unsqueeze(0))        
        if i == 0:
            x = data[store]["test_dataset"].scaler.inverse_transform(data[store]["test_dataset"][i][0]).squeeze()
            values[:args.history_length] = x
            preds[:args.history_length] = x
        
        y = data[store]["test_dataset"].scaler.inverse_transform(data[store]["test_dataset"][i][1]).squeeze()
        values[i+args.history_length:i+args.history_length+args.pred_length] = y
        
        if args.model_type == "lstm" or args.model_type == "tcn":
            y_hat = data[store]["test_dataset"].scaler.inverse_transform(y_hat.detach().numpy()).squeeze()
        elif args.model_type == "dense" or args.model_type == "transformer":
            y_hat = data[store]["test_dataset"].scaler.inverse_transform(y_hat.detach().numpy().reshape([-1, 1])).squeeze()
        y_hat = np.round(y_hat)
        preds[i+args.history_length:i+args.history_length+args.pred_length] = y_hat

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.plot(values)
    ax.plot(preds)
    plt.savefig('{}/test_predictions_{}.png'.format(args.path_to_viz[store], epoch + 1))
    plt.show()
    plt.close()