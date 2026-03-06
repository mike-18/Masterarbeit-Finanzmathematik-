import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.io import arff
from scipy import stats
from copy import deepcopy
from torch.utils.data import Dataset
from Utils.masking_utils import noise_mask
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from sklearn.preprocessing import MinMaxScaler

class CustomDatasetGuided(Dataset):
    def __init__(
        self,
        data_root,
        end, 
        window=10, 
        save2npy=True, 
        neg_one_to_one=True,
        period='train',
        output_dir='./OUTPUT',
        
    ):
        super(CustomDatasetGuided, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'

        self.auto_norm, self.save2npy = neg_one_to_one, save2npy
        self.data_0, self.data_1, self.scaler = self.read_data(data_root,window,end)
        self.labels = np.zeros(self.data_0.shape[0] + self.data_1.shape[0]).astype(np.int64)
        self.labels[self.data_0.shape[0]:] = 1
        self.rawdata = np.vstack([self.data_0, self.data_1])
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]

        self.samples = self.normalize(self.rawdata)


        self.sample_num = self.samples.shape[0]
        print('fertig')

    

    def read_data(self, filepath, length,end):
        csv_path = filepath
        data = pd.read_csv(csv_path)
        data = data.values
        data_0, data_1 = self.__Classify__(data,end ,window=length)
        data_0 = self.__Sequence_slicer__(data_0, length)
        data_1 = self.__Sequence_slicer__(data_1, length)

        print(f"Class 0: {data_0.shape}, Class 1: {data_1.shape}")

        data = np.vstack([data_0.reshape(-1, data_0.shape[-1]), data_1.reshape(-1, data_1.shape[-1])])

        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        
        return data_0, data_1, scaler

    @staticmethod
    def __Sequence_slicer__(data, window):
      num_variables = data.shape[1]
      seq_length = window
      overlap_size = window-1
      num_points = data.shape[0]
      step_size = seq_length - overlap_size
      num_sequences = (num_points - seq_length) // step_size + 1
      trimmed_data = np.array([data[i:i + seq_length] for i in range(0, num_points - seq_length + 1, step_size)])

      return trimmed_data.reshape(num_sequences, seq_length, num_variables)

    @staticmethod
    def __Classify__(data, end, window):
      print(end)
      all_seq  = data[:end,:]
      last_seq = data[end-window:end,:]
      
      return all_seq, last_seq


    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            y = self.labels[ind]  # (1,) int
            return torch.from_numpy(x).float(), torch.tensor(y)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    def shift_period(self, period):
        assert period in ['train', 'test'], 'period must be train or test.'
        self.period = period