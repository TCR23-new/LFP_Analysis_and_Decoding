from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold,cross_validate
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import copy
from torch import nn
from sklearn.model_selection import train_test_split,RepeatedKFold,RepeatedStratifiedKFold
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,r2_score
import os
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
import pickle
from sklearn.decomposition import PCA
import h5py
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

def Value_CS_ShiraData(stimIDs):
  val_cs = np.zeros((len(stimIDs),2))
  for i in range(val_cs.shape[0]):
    val_cs[i,0] = int(str(int(stimIDs[i,0]))[1])
    val_cs[i,1] = int(str(int(stimIDs[i,0]))[2])
  return val_cs

def BetterFeatureEncoding(val_cs):
    y = np.zeros((val_cs.shape[0],))
    for i in range(val_cs.shape[0]):
        if val_cs[i,0] > val_cs[i,1]:
            y[i] = 1
    return y

def HighMidLow_Encoding(val):
    new_enc = np.zeros((val.shape[0],))
    for i in range(val.shape[0]):
        if val[i] == 1 or val[i] == 2:
            new_enc[i] = 0
        elif val[i] == 3:
            new_enc[i] = 1
        else:
            new_enc[i] = 2
    return new_enc

def CollectBands_AmpPhase2(data,bandList,timebin_Ref,tbin):
    tmp_amp = []
    tmp_phase = []
    for b in bandList:
        b1 = 'amp'+b
        d1 = np.transpose(data[b1][:],(3,2,1,0))[tbin,:,:,timebin_Ref]
        b2 = 'ph'+b
        # print(b2)
        d2 = np.sin(np.transpose(data[b2][:],(3,2,1,0))[tbin,:,:,timebin_Ref])
        d3 = np.cos(np.transpose(data[b2][:],(3,2,1,0))[tbin,:,:,timebin_Ref])
        tmp_amp.append(d1)
        tmp_phase.append(d2)
        tmp_phase.append(d3)
    spks = np.transpose(data['spk'][:],(3,2,1,0))[tbin,:,:,timebin_Ref]
    return np.stack(tmp_amp,axis=1),np.stack(tmp_phase,axis=1),spks.reshape(spks.shape[0],1,spks.shape[1])


def FrqBand_Scaling(data):
    scalers = []
    for ii in range(data.shape[1]):
        scaler = MinMaxScaler()
        scaler.fit(data[:,ii,:])
        scalers.append(scaler)
    return scalers

def CollectBands(data,bandList,timebin,spkflag):
    tmp = []
    for b in bandList:
        d1 = np.transpose(data[b][:],(2,1,0))[:,:,timebin]
        tmp.append(d1)
    if spkflag==True:
        tmp.append(np.transpose(data['spk'][:],(2,1,0))[:,:,timebin])
    return np.stack(tmp,axis=1)

def CollectBands_AmpPhase(data,bandList,timebin_Ref,tbin):
    tmp_amp = []
    tmp_phase = []
    for b in bandList:
        b1 = 'amp'+b
        d1 = np.transpose(data[b1][:],(3,2,1,0))[tbin,:,:,timebin_Ref]
        b2 = 'ph'+b
        # print(b2)
        d2 = np.sin(np.transpose(data[b2][:],(3,2,1,0))[tbin,:,:,timebin_Ref])
        d3 = np.cos(np.transpose(data[b2][:],(3,2,1,0))[tbin,:,:,timebin_Ref])
        tmp_amp.append(d1)
        tmp_phase.append(d2)
        tmp_phase.append(d3)
    spks = np.transpose(data['spk'][:],(3,2,1,0))[tbin,:,:,timebin_Ref]
    return np.stack(tmp_amp,axis=1),np.stack(tmp_phase,axis=1),spks.reshape(spks.shape[0],1,spks.shape[1])


def three_vals(v):
    nv = np.zeros_like(v)
    for i in range(v.shape[0]):
        if v[i] == 1 or v[i]==2:
            nv[i] = 0
        elif v[i] == 3:
            nv[i] = 1
        else:
            nv[i] = 2
    return nv

def extract_identifier(f):
    parts = f.split('_')
    return parts[1]

def GetIndices(bandNames,namesOI):
    bandNames2 = bandNames.copy()
    bandNames2.append('spks')
    indices = []
    for n in namesOI:
        indices.append(bandNames2.index(n))
    return indices

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model%2 != 0:
          d_model += 1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.shape[0],:,:x.shape[2]] #self.pe[:x.size(0)] x.shape[2]
        return self.dropout(x)
