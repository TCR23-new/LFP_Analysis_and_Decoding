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

def Decodes(X,y):
    y2 = np.zeros((y.shape[0],))
    kf = KFold(n_splits = 5,shuffle=True)
    for ix,(train_index,test_index) in enumerate(kf.split(X)):
        trX = X[train_index,:]
        trY = y[train_index]
        tsX = X[test_index,:]
        tsY = y[test_index]
        model = LinearRegression()
        scaler = StandardScaler(with_mean=True,with_std=True) #MinMaxScaler(feature_range=(-1,1))
        scaler.fit(trX)
        model.fit(scaler.transform(trX),trY)
        y2[test_index] = model.predict(scaler.transform(tsX))
    return y2

def Decoding(all_amps,v):
    mp_ = np.zeros(all_amps[0].shape[2],)
    for i in range(all_amps[0].shape[2]):
        tmp_data = []
        for q in range(len(all_amps)):
            tmp_data.append(PCA(n_components = 0.9).fit_transform(StandardScaler().fit_transform(all_amps[q][:,:,i])))
        if len(all_amps) == 1:
            X = tmp_data[0]
            # print('data shape {}'.format(X.shape))
        else:
            X = np.concatenate((tmp_data),axis=1)
        cv = cross_validate(LinearDiscriminantAnalysis(),X,v,cv=KFold(n_splits=5,shuffle=True),scoring=['accuracy'])
        mp_[i] = np.mean(cv['test_accuracy'])
    return mp_

def MB_Decoding(main_dir,spike_files):
    # get the band directories
    band_dirs = sorted(os.listdir(main_dir))
    # get the session identifiers
    session_Ids = [extract_identifier(f) for f in os.listdir(main_dir + band_dirs[0])]
    # looping over the session identifiers....
    all_mps = []
    for s in session_Ids:
        try:
            tmp_list = []
            # tmp_vars = []
            # get the corresponding file from each band
            # print(s)
            spk_fileOI = [fs1 for fs1 in spike_files if s in fs1][0]
            data_spks = h5py.File(spk_fileOI,'r')
            spks = np.squeeze(np.transpose(data_spks['spk'][:],(3,2,1,0))[:,:,:,1])
            for ij,b in enumerate(band_dirs):
                tmp_f = main_dir +b+'/aligned_{}_{}.mat'.format(s,b)
                # load the data
                a1 =h5py.File(tmp_f,'r')
                ampBin = a1['ampBin']
                info = a1['info']
                if ij == 0:
                    val1 = three_vals(info['val']['val1'][:].reshape(-1))
                    val2 = three_vals(info['val']['val2'][:].reshape(-1))
                    fix1chosen = info['val']['firstChosen'][:].reshape(-1)
                    removeTrials = info['removeTr'][:].reshape(-1)
                    idx11 = np.nonzero(removeTrials == 0)[0]
                    tmp_vars=[val1,val2,fix1chosen]
                amp = np.transpose(ampBin['fix2'][:],(1,0,2))
                # store the data
                tmp_list.append(amp)
            # combine the data
            spks = spks[:,idx11,:]
            spks = np.transpose(spks,(1,2,0))
            tmp_list.append(spks)
            # try:
            tmp_mp = []
            for v_ in tmp_vars:
                tmp_mp.append(Decoding(tmp_list,v_))
            all_mps.append(tmp_mp)
        except:
            print('failure on session id {}'.format(s))
    return all_mps

def MB_DecodingCombs(main_dir,spike_files):
    # get the band directories
    # bandCombs = [['spks'],['band30to50Hz','band50to100Hz','spks'],['band30to50Hz','band50to100Hz','band100to200Hz','spks']]
    bandCombs = [['spks'],\
    ['band50to100Hz','band100to200Hz','spks'],\
    ['band30to50Hz','spks'],\
    ['band30to50Hz','band50to100Hz','band100to200Hz','spks']]
    band_dirs0 = sorted(os.listdir(main_dir))
    all_mps_combs = []
    for band_dirs in bandCombs:
        # get the session identifiers
        session_Ids = [extract_identifier(f) for f in os.listdir(main_dir + band_dirs0[0])]
        # looping over the session identifiers....
        all_mps = []
        for s in session_Ids:
            try:
                tmp_list = []
                spk_fileOI = [fs1 for fs1 in spike_files if s in fs1][0]
                data_spks = h5py.File(spk_fileOI,'r')
                spks = np.squeeze(np.transpose(data_spks['spk'][:],(3,2,1,0))[:,:,:,1])
                for ij,b in enumerate(band_dirs0):
                    tmp_f = main_dir +b+'/aligned_{}_{}.mat'.format(s,b)
                    # load the data
                    a1 =h5py.File(tmp_f,'r')
                    ampBin = a1['ampBin']
                    info = a1['info']
                    if ij == 0:
                        val1 = three_vals(info['val']['val1'][:].reshape(-1))
                        val2 = three_vals(info['val']['val2'][:].reshape(-1))
                        fix1chosen = info['val']['firstChosen'][:].reshape(-1)
                        removeTrials = info['removeTr'][:].reshape(-1)
                        idx11 = np.nonzero(removeTrials == 0)[0]
                        tmp_vars=[val1,val2,fix1chosen]
                    amp = np.transpose(ampBin['fix2'][:],(1,0,2))
                    # store the data
                    tmp_list.append(amp)
                # combine the data
                spks = spks[:,idx11,:]
                spks = np.transpose(spks,(1,2,0))
                tmp_list.append(spks)
                indices = GetIndices(band_dirs0,band_dirs)
                tmp_list22 = [tmp_list[xy] for xy in indices]
                tmp_mp = []
                for v_ in tmp_vars:
                    # print(v_.shape)
                    tmp_mp.append(Decoding(tmp_list22,v_))
                all_mps.append(tmp_mp)
            except:
                print('failure on session id {}'.format(s))
        all_mps_combs.append(all_mps)
    return all_mps_combs


def CCA_Corr(ampdata_s,y,n_splits,n_repeats):
    mat1 = np.zeros((ampdata_s.shape[1],ampdata_s.shape[1]))
    mat2 = np.zeros((2,ampdata_s.shape[1],ampdata_s.shape[1]))
    factors = []
    for i in range(ampdata_s.shape[1]):
        # fa1= FactorAnalysis(n_components = 10).fit_transform(StandardScaler().fit_transform(ampdata_s[:,i,:]))
        fa1= PCA(n_components=0.95).fit_transform(StandardScaler().fit_transform(ampdata_s[:,i,:]))
        factors.append(fa1)
    for i in range(ampdata_s.shape[1]):
        for j in range(ampdata_s.shape[1]):
            if i != j:
                shape_i,shape_j = factors[i].shape[1],factors[j].shape[1]
                min_shape = np.min([shape_i,shape_j]).astype('int')
                a,b = CCA(n_components = min_shape,max_iter=2000).fit_transform(factors[i],factors[j])
                ab = np.concatenate((a[:,0].reshape(-1,1),b[:,0].reshape(-1,1)),axis=1)
                mat1[i,j] = np.corrcoef(ab.T)[0,1]
                mat2[0,i,j] = np.mean(CV_Calc(LinearDiscriminantAnalysis(),a,y,n_splits,n_repeats))
                mat2[1,i,j] = np.mean(CV_Calc(LinearDiscriminantAnalysis(),b,y,n_splits,n_repeats))
    return mat1,mat2

def CV_Calc(model,X,y,n_splits,n_repeats):
    #(LinearDiscriminantAnalysis(),X_Amp_Spk[:,i,:],val1,cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
    rkf = RepeatedKFold(n_splits = n_splits,n_repeats = n_repeats)
    out1 = []
    for i,(train_index,test_index) in enumerate(rkf.split(X)):
        model1 = copy.deepcopy(model)
        trX,tsX = X[train_index,:],X[test_index,:]
        trY,tsY = y[train_index],y[test_index]
        scaler = StandardScaler()
        scaler.fit(trX)
        model1.fit(scaler.transform(trX),trY)
        ypred = model1.predict(scaler.transform(tsX))
        out1.append(accuracy_score(tsY,ypred))
    return out1

def BayesOpt_Trainer(parameters): #lfpdata,modelOI,val,training_Data,device):
    if modelOI == 'RNN':
        #(self,inputDim,hiddenLayerSize,nLayers,num_classes,drpOut,NumLinLayers,ActFnctFlag)
        model = RNNNet(inputDims[1],parameters.get('hls',100),parameters.get('nLayers',3),\
        np.unique(val_data).shape[0],parameters.get('drpOut',0),parameters.get('NLinLayers',1),parameters.get('ActFn','Tanh'))
    else:
        #inputDims,num_classes,NumLinLayers,ActFnctFlag,MxPl_Size)
        model = ConvNet(inputDims,np.unique(val_data).shape[0],parameters.get('NLinLayers',1),\
        parameters.get('ActFn','Tanh'),parameters.get('MxPl',4))
    #
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    NumEpochs = 1800
    lr = parameters.get("lr",1e-4) #1e-4 #3 #5e-6 #2e-5
    # labels = val-1
    lfpdata0 = lfpdata
    if lfpdata0.shape[1] > lfpdata0.shape[2]:
        lfpdata0 = np.transpose(lfpdata0,(0,2,1))
    train_X0,test_X0,train_y,test_y= training_Data
    train_X = lfpdata0[train_X0[:,0],:,:]
    test_X = lfpdata0[test_X0[:,0],:,:]
    #
    sflag = True
    scalers = FrqBand_Scaling(train_X)
    model = model.float()
    model = model.to(device)
    # define the optimizer Adam
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    #data,value_data,scalers
    train_data = OFC_Dataset_Ensemble_TimeFB2(train_X,train_y,scalers=scalers)
    test_data = OFC_Dataset_Ensemble_TimeFB2(test_X,test_y,scalers=scalers)
    trn_dl = DataLoader(train_data,batch_size=batch_size,collate_fn=train_data.collate_fn,shuffle=True)
    tst_dl = DataLoader(test_data,batch_size=batch_size,collate_fn=test_data.collate_fn,shuffle=True)
    #  Begin training....
    train_loss_values,test_loss_values,ctr,acc_values,acc_tst_values = [],[],0,[],[]
    epochs_no_improve,n_epochs_stop,min_tst_loss = 0,20,1e7 # 1
    warmup_period = 0
    for i in range(NumEpochs):
        # print('epoch '+str(i+1))
        tmp_train_loss,ctr = 0,0
        for ix,batch in enumerate(iter(trn_dl)):
            trials = batch
            loss = train_batch(trials,model,opt,device)
            tmp_train_loss+=loss.item()
            ctr+=1
        # print(' avg. train loss '+str(tmp_train_loss/ctr))
        train_loss_values.append(tmp_train_loss/ctr)
        tmp_test_loss,tst_ctr = 0,0
        for ix,tst_batch in enumerate(iter(tst_dl)):
          trials_t = tst_batch
          tst_loss,preds = test_batch(trials_t,model,device)
          tmp_test_loss += tst_loss.item()
          tst_ctr += 1
        test_loss_values.append(tmp_test_loss/tst_ctr)
        # print('avg. test loss '+str(tmp_test_loss/tst_ctr))
        if (tmp_test_loss/tst_ctr) < min_tst_loss:
          min_tst_loss = (tmp_test_loss/tst_ctr)
          epochs_no_improve = 0
        else:
          epochs_no_improve += 1
        if epochs_no_improve > n_epochs_stop and i >150: # 10 150
          break
        if i < warmup_period:
          for g in opt.param_groups:
            g['lr'] = lr*((i+1)/warmup_period)
    # evaluate
    tmp_out = TestModel(test_X,test_y,scalers,model,device)
    acc_sum = tmp_out[0]
    return acc_sum


def train_batch_ensemble(input,model,optimizer,device):
    model.train()
    ypred1,ypred2,ypred3,ypred4 = model.forward(input[0].float().to(device))
    loss_val1 = nn.CrossEntropyLoss()(ypred1,input[1].to(device).reshape(-1))
    loss_val2 = nn.CrossEntropyLoss()(ypred2,input[1].to(device).reshape(-1))
    loss_val3 = nn.CrossEntropyLoss()(ypred3,input[1].to(device).reshape(-1))
    loss_val4 = nn.CrossEntropyLoss()(ypred4,input[1].to(device).reshape(-1))
    loss_val = loss_val1 + loss_val2 + loss_val3 + loss_val4
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    return loss_val

def train_batch(input,model,optimizer,device):
    model.train()
    ypred = model.forward(input[0].float().to(device))
    loss_val = nn.CrossEntropyLoss()(ypred,input[1].to(device).reshape(-1))
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    return loss_val

def Trainer(lfpdata,modelOI,val,training_Data,device,lr):
    batch_size = 100
    NumEpochs = 1800
    # lr = 1e-4 #3 #5e-6 #2e-5
    # labels = val-1
    if lfpdata.shape[1] > lfpdata.shape[2]:
        lfpdata = np.transpose(lfpdata,(0,2,1))
    train_X0,test_X0,train_y,test_y= training_Data
    train_X = lfpdata[train_X0[:,0],:,:]
    test_X = lfpdata[test_X0[:,0],:,:]
    #
    scalers = FrqBand_Scaling(train_X)
    model = modelOI.float()
    model = model.to(device)
    # define the optimizer Adam
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    #data,value_data,scalers
    train_data = OFC_Dataset_Ensemble_TimeFB2(train_X,train_y,scalers=scalers)
    test_data = OFC_Dataset_Ensemble_TimeFB2(test_X,test_y,scalers=scalers)
    trn_dl = DataLoader(train_data,batch_size=batch_size,collate_fn=train_data.collate_fn,shuffle=True)
    tst_dl = DataLoader(test_data,batch_size=batch_size,collate_fn=test_data.collate_fn,shuffle=True)
    #  Begin training....
    train_loss_values,test_loss_values,ctr,acc_values,acc_tst_values = [],[],0,[],[]
    epochs_no_improve,n_epochs_stop,min_tst_loss = 0,20,1e7 # 1
    warmup_period = 0
    for i in range(NumEpochs):
        print('epoch '+str(i+1))
        tmp_train_loss,ctr = 0,0
        for ix,batch in enumerate(iter(trn_dl)):
            trials = batch
            loss = train_batch(trials,model,opt,device)
            tmp_train_loss+=loss.item()
            ctr+=1
        print(' avg. train loss '+str(tmp_train_loss/ctr))
        train_loss_values.append(tmp_train_loss/ctr)
        tmp_test_loss,tst_ctr = 0,0
        for ix,tst_batch in enumerate(iter(tst_dl)):
          trials_t = tst_batch
          tst_loss,preds = test_batch(trials_t,model,device)
          tmp_test_loss += tst_loss.item()
          tst_ctr += 1
        test_loss_values.append(tmp_test_loss/tst_ctr)
        print('avg. test loss '+str(tmp_test_loss/tst_ctr))
        if (tmp_test_loss/tst_ctr) < min_tst_loss:
          min_tst_loss = (tmp_test_loss/tst_ctr)
          epochs_no_improve = 0
        else:
          epochs_no_improve += 1
        if epochs_no_improve > n_epochs_stop and i >200: # 10 150
          break
        if i < warmup_period:
          for g in opt.param_groups:
            g['lr'] = lr*((i+1)/warmup_period)
    return scalers,model,[train_X,test_X,train_y,test_y,train_loss_values,test_loss_values,train_X0,test_X0]

def Trainer_Ensemble(lfpdata,modelOI,val,training_Data,device,lr):
    batch_size = 100
    NumEpochs = 1800
    train_X0,test_X0,train_y,test_y= training_Data
    train_X = lfpdata[train_X0[:,0],:,:]
    test_X = lfpdata[test_X0[:,0],:,:]
    #
    scalers = FrqBand_Scaling(train_X)
    model = modelOI.float()
    model = model.to(device)
    # define the optimizer Adam
    opt = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-3)
    #data,value_data,scalers
    train_data = OFC_Dataset_Ensemble_TimeFB2(train_X,train_y,scalers=scalers)
    test_data = OFC_Dataset_Ensemble_TimeFB2(test_X,test_y,scalers=scalers)
    trn_dl = DataLoader(train_data,batch_size=batch_size,collate_fn=train_data.collate_fn,shuffle=True)
    tst_dl = DataLoader(test_data,batch_size=batch_size,collate_fn=test_data.collate_fn,shuffle=True)
    #  Begin training....
    train_loss_values,test_loss_values,ctr,acc_values,acc_tst_values = [],[],0,[],[]
    epochs_no_improve,n_epochs_stop,min_tst_loss = 0,20,1e7
    warmup_period = 0
    for i in range(NumEpochs):
        # print('epoch '+str(i+1))
        tmp_train_loss,ctr = 0,0
        for ix,batch in enumerate(iter(trn_dl)):
            trials = batch
            loss = train_batch_ensemble(trials,model,opt,device)
            tmp_train_loss+=loss.item()
            ctr+=1
        train_loss_values.append(tmp_train_loss/ctr)
        tmp_test_loss,tst_ctr = 0,0
        for ix,tst_batch in enumerate(iter(tst_dl)):
          trials_t = tst_batch
          tst_loss = test_batch_ensemble(trials_t,model,device)
          tmp_test_loss += tst_loss.item()
          tst_ctr += 1
        test_loss_values.append(tmp_test_loss/tst_ctr)
        #
        if (tmp_test_loss/tst_ctr) < min_tst_loss:
          min_tst_loss = (tmp_test_loss/tst_ctr)
          epochs_no_improve = 0
        else:
          epochs_no_improve += 1
        if epochs_no_improve > n_epochs_stop and i >200: # 10 150
          break
        if i < warmup_period:
          for g in opt.param_groups:
            g['lr'] = lr*((i+1)/warmup_period)
    return scalers,model,[train_X,test_X,train_y,test_y,train_loss_values,test_loss_values,train_X0,test_X0]
