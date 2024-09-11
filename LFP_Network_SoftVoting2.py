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
import math
from torch import Tensor
# from ax.service.managed_loop import optimize
from collections import OrderedDict
from preproc_methods import CollectBands_AmpPhase,HighMidLow_Encoding
from neural_nets import Spk_LFP_Net3
from training_methods import Trainer
from eval_methods import TestModel

if __name__ == '__main__':
    global lfpdata,modelOI,val,training_Data,inputDims,scalers
    scalers = []
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobid = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    # print('--- Current jobid {} ---'.format(jobid))
    spkflag = True
    if  jobid >= 83:
        valflag = 2
        timebin = 1
        val_name = 'val2'
    elif jobid > 41 and jobid <= 82:
        valflag = 1
        timebin = 1
        val_name = 'val1'
    else:
        valflag = 0
        timebin = 0
        val_name = 'val1'
    savedir = '/projectsp/vm430_1/Mlab/TevinFolder/LFP_Net_Outputs/'
    other_dir = '/home/tr561/ShiraStuff/LFP_Net_Outputs/'
    dir1 = '/projectsp/vm430_1/Mlab/TevinFolder/NewData_PhaseAmp/'
    allfiles1 = sorted(os.listdir(dir1))
    allfiles1 = [f for f in allfiles1 if '_post_' in f]
    alldata = h5py.File(dir1 + allfiles1[(jobid-1)%41],'r')
    bandRanges_newdata = ['4to8Hz','8to14Hz','15to30Hz','30to50Hz','50to100Hz','100to200Hz']
    ampdata,phdata,spks = CollectBands_AmpPhase(alldata,bandRanges_newdata,timebin,13)
    val_data = alldata[val_name][:].reshape(-1)
    # 3 class case

    val_data = HighMidLow_Encoding(val_data)
    val_data = val_data.astype('long')
    #
    ampdata_spks = np.concatenate((ampdata,spks),axis=1)
    numBands = ampdata.shape[1]
    #
    n_splits = 5
    n_repeats=1
    lr = 1e-4
    model_perf = np.zeros((n_splits*n_repeats,3))
    ypreds_out = []
    pedata_all_outs = []
    ctr = 0
    rkf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats)
    split_testMetrics = []
    best_params_List = []
    #
    val = val_data
    for i, (train_index, test_index) in enumerate(rkf.split(np.array(range(ampdata_spks.shape[0])).reshape(-1,1),val_data)):
      print('fold {}'.format(i+1))
      if i!=0 and i % n_splits == 0:
        ctr +=1
      train_X0 = train_index.reshape(-1,1)
      test_X0 = test_index.reshape(-1,1)
      test_X = ampdata_spks[test_X0[:,0],:,:]
      train_X = ampdata_spks[train_X0[:,0],:,:]
      train_y = val_data[train_X0[:,0]].astype('long')
      test_y = val_data[test_X0[:,0]].astype('long')
      training_Data = [train_X0,test_X0,train_y,test_y]
      #
      # model0 = Spk_LFP_Net3(numBands,train_X.shape[2],100,3)
      model0 = Spk_LFP_Net3(numBands+1)
      scalers,model,training_Data = Trainer(ampdata_spks,model0,val_data,training_Data,device,lr)
      train_X0,test_X0,train_y0,test_y0,trn_lss,tst_lss,train_X0_ind,test_X0_ind = training_Data
      #
      tmp_test_outs = TestModel(test_X0,test_y0,scalers,model,device)
      split_testMetrics.append(tmp_test_outs)
    # save stuff
    pickle.dump([split_testMetrics],open('{}LFP_ModelType_NewArch_Training_job_{}_vflag_{}_wSpks_SoftVotingTRF.pickle'.format(savedir,jobid,valflag),'wb'))
