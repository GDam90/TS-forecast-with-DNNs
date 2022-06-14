import torch
import numpy as np
import os

def train_epoch(model, data, store, args, epoch, criterion, optimizer):
    model.train()
    train_epoch_loss = np.zeros(len(data[store]["train_loader"]))
    for i, (x, y) in enumerate(data[store]["train_loader"]):
        # loss
        if args.model_type == "transformer":
            y_hat = model(x, y)
        else:
            y_hat = model(x)
        
        loss = criterion(y_hat, y.squeeze())
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        
        train_epoch_loss[i] = loss.item()
        
    train_mean_loss = np.mean(train_epoch_loss)
    
    if (epoch+1) % args.log_interval == 0:
        log_string = 'Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, train_mean_loss)
        print(log_string)
        with open(os.path.join(args.path_to_logs[store], 'train.txt'), 'a') as f:
            f.write(log_string + '\n')
            
    return train_mean_loss, model, optimizer