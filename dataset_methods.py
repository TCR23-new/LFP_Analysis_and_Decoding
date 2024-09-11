import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split,RepeatedKFold,RepeatedStratifiedKFold
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import os
import pickle
import h5py
from ax.service.managed_loop import optimize
from collections import OrderedDict


class OFC_Dataset_Ensemble_TimeFB2(Dataset):
    def __init__(self,data,value_data,scalers=None):
        self.data = data
        self.valuedata = value_data
        self.scalers = scalers
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        if self.scalers != None:
          out['trialdata'] = torch.from_numpy(np.zeros_like(self.data[idx]))
          for iy,scaler in enumerate(self.scalers):
            out['trialdata'][iy,:] = torch.from_numpy(scaler.transform(self.data[idx,iy,:].reshape(1,-1)))
        else:
          out['trialdata'] = torch.from_numpy(self.data[idx,:,:])
        out['value'] = self.valuedata[idx]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.stack([x['trialdata'] for x in data]).float()
        values = torch.tensor([x['value'] for x in data]).long().reshape(-1,1) #float().reshape(-1,1)
        return (trials,values)
