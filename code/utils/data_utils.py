import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
# utils.
from utils.constants import PATH_TO_DATASET, PATH_TO_CONFIG
from utils.constants import OPENING_CLOSURE as OC

from utils.args_utils import adjust_args, yaml2args


# Define the Dataset for the time series
# The dataset takes in a pandas.DataFrame and split those
# in windows of size window_size and output_size.
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, arguments, train_scaler=None):
        self.data = data[arguments.target_name].values.reshape(-1, 1)
        self.args = arguments
        self.window_size = self.args.history_length
        self.output_size = self.args.pred_length
        if train_scaler is None:
            self.got_scaler = False
            self.scaler = None
            self.scaler = self.get_scaler()
        else:
            self.got_scaler = True
            self.scaler = train_scaler
        self.segs_np = self.gen_dataset()
    
    def get_scaler(self):
        if self.scaler is None:
            if self.args.normalization_strategy == "standard":
                self.scaler = StandardScaler()
            elif self.args.normalization_strategy == "minmax":
                self.scaler = MinMaxScaler()
        return self.scaler
    
    def scale_data(self):
        data = self.data.copy()
        data = data.reshape(-1, 1)
        
        if not self.got_scaler:
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)
        return data
    
    def windows_split(self, data):
        segs_X, segs_Y = [], []
        for i in range(len(data) - self.window_size - self.output_size):
            segs_X.append(data[i: i + self.window_size])
            segs_Y.append(data[i + self.window_size: i + self.window_size + self.output_size])
        
        segs_X = np.stack(segs_X, axis=0)
        segs_Y = np.stack(segs_Y, axis=0)
        return segs_X, segs_Y
        
    def gen_dataset(self):
        '''
        Function to split data in windows with their labels
        Data have to be scaled leveraging the scaler.
        '''
        data = self.scale_data()
        segs_x, segs_y = self.windows_split(data)
        
        return segs_x, segs_y
    
    def __len__(self):
        return len(self.segs_np[0])
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.segs_np[0][idx]).float()
        y = torch.from_numpy(self.segs_np[1][idx]).float()
        
        if self.args.model_type == "tcn":
            x = x.permute(1, 0)
            y = y.permute(1, 0)
            
        return x, y

def group_store_1h(data, s : str, opening : int, closing : int):     
  store = data[data.STORE==s]  #["STORE"]
  store = store.rename(columns={'DATE TIME ERP INSERT' : 'Get'})

  first_last = store['Get'].agg(['min','max']).dt.normalize().copy()
  first_last['min'] = first_last['min'] + pd.DateOffset(hours=opening)
  first_last['max'] = first_last['max'] + pd.DateOffset(hours=closing)
  store = store.append(first_last.to_frame(), ignore_index=True) 
  
  store.set_index('Get',inplace=True)
  store = store.groupby(pd.Grouper(freq="1h")).size().to_frame('n_events')

  to_drop = store.between_time(datetime.time(closing), datetime.time(opening-1)).index
  store.drop(to_drop, inplace=True)

  store.reset_index(inplace=True)
  store.at[0,'n_events'] -=1

  return store

def generate_store_datasets(args):
    data = pd.read_excel(PATH_TO_DATASET, engine='openpyxl')
    stores_dict = {}
    year, month, day = args.split_date.split('-')
    split_date = datetime.datetime(int(year), int(month), int(day))
    for store in args.stores_list:
        stores_dict[store] = {}
        stores_dict[store]["data"] = group_store_1h(data, store, OC[store][0], OC[store][1])
        stores_dict[store]["train_data"] = stores_dict[store]["data"][stores_dict[store]["data"].Get < split_date]
        stores_dict[store]["test_data"] = stores_dict[store]["data"][stores_dict[store]["data"].Get >= split_date]
        stores_dict[store]["train_dataset"] = TimeSeriesDataset(stores_dict[store]["train_data"], args)
        stores_dict[store]["test_dataset"] = TimeSeriesDataset(stores_dict[store]["test_data"], args, stores_dict[store]["train_dataset"].scaler)
        stores_dict[store]["train_loader"] = torch.utils.data.DataLoader(stores_dict[store]["train_dataset"] , batch_size=args.batchsize, shuffle=True)
        stores_dict[store]["test_loader"] = torch.utils.data.DataLoader(stores_dict[store]["test_dataset"] , batch_size=args.batchsize, shuffle=False)
    
    return stores_dict


if __name__ == "__main__":
    args = yaml2args(PATH_TO_CONFIG)
    args = adjust_args(args)
    
    all_stores = generate_store_datasets(args)
    
    print()
    
    
