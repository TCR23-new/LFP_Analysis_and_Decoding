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

def test_batch(input,model,device):
    model.eval()
    with torch.no_grad():
      preds = model(input[0].float().to(device))
    loss_val = nn.CrossEntropyLoss()(preds,input[1].to(device).reshape(-1))
    return loss_val,preds

def TestModel(tsX,tsY,scalers,model,device):
    model.eval()
    outputs = []
    # preprocess the data
    test_data = OFC_Dataset_Ensemble_TimeFB2(tsX,tsY,scalers=scalers)
    tst_dl = DataLoader(test_data,batch_size=tsX.shape[0],collate_fn=test_data.collate_fn,shuffle=True)
    for ix,batch in enumerate(iter(tst_dl)):
        # run the data through the model
        with torch.no_grad():
          preds = model(batch[0].float().to(device))
        ypreds = torch.argmax(preds,dim=1)
        # compute metrics
        # accuracy
        acc1 = accuracy_score(batch[1].cpu().long().numpy(),ypreds.cpu().numpy())
        print('accuracy score {}'.format(acc1))
        outputs.append(acc1)
        # confusion matrix
        outputs.append(confusion_matrix(batch[1].cpu().long().numpy(),ypreds.cpu().numpy()))
    return outputs

def test_batch_ensemble(input,model,device):
    model.eval()
    with torch.no_grad():
      # preds = model(input[0].float().to(device))
      ypred1,ypred2,ypred3,ypred4 = model.forward(input[0].float().to(device))
      loss_val1 = nn.CrossEntropyLoss()(ypred1,input[1].to(device).reshape(-1))
      loss_val2 = nn.CrossEntropyLoss()(ypred2,input[1].to(device).reshape(-1))
      loss_val3 = nn.CrossEntropyLoss()(ypred3,input[1].to(device).reshape(-1))
      loss_val4 = nn.CrossEntropyLoss()(ypred4,input[1].to(device).reshape(-1))
    loss_val = loss_val1 + loss_val2 + loss_val3 + loss_val4
    # loss_val = nn.CrossEntropyLoss()(preds,input[1].to(device).reshape(-1))
    return loss_val
