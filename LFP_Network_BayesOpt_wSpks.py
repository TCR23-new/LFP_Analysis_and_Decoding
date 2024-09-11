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
from preproc_methods import CollectBands,HighMidLow_Encoding
from neural_nets import RNNNet,ConvNet
from eval_methods import TestModel
from training_methods import Trainer,BayesOpt_Trainer

def session_params(jobid):
    spkflag = False
    if jobid in list(range(1,42)) + list(range(124,165)):
        valflag = 0;timebin = 0;val_name = 'val1'
        if jobid in list(range(124,165)):
            spkflag = True
    elif jobid in list(range(42,83)) + list(range(165,206)):
        valflag = 1;timebin = 1;val_name = 'val1'
        if jobid in list(range(165,206)):
            spkflag = True
    else:
        valflag = 2;timebin = 1;val_name = 'val2'
        if jobid in list(range(206,247)):
            spkflag = True
    return valflag,timebin,val_name,spkflag

#
if __name__ == '__main__':
    global lfpdata,modelOI,val,training_Data,inputDims,scalers
    scalers = []
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobid = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    valflag,timebin,val_name,spkflag = session_params(jobid)
    modelflag = 'CNN'#'CNN' #
    # device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = '/projectsp/vm430_1/Mlab/TevinFolder/LFP_Net_Outputs/'
    dir1 = '/projectsp/vm430_1/Mlab/TevinFolder/singleBinData/'
    allfiles1 = sorted(os.listdir(dir1))
    allfiles1 = [f for f in allfiles1 if '_post_' in f]
    alldata = h5py.File(dir1 + allfiles1[(jobid-1)%41],'r')
    # bandRanges = ['band15to30Hz','band30to50Hz','band50to100Hz','band100to200Hz']#'band4to8Hz','band8to14Hz',
    bandRanges = ['band4to8Hz','band8to14Hz','band15to30Hz','band30to50Hz','band50to100Hz','band100to200Hz']#
    lfpdata_0 = CollectBands(alldata,bandRanges,timebin,spkflag)
    val_data = alldata[val_name][:].reshape(-1)
    # 3 class case
    val_data = HighMidLow_Encoding(val_data)
    val_data = val_data.astype('long')
    #
    lfpdata = lfpdata_0
    if lfpdata.shape[1] > lfpdata.shape[2]:
        inputDims = [lfpdata.shape[2],lfpdata.shape[1]]
    else:
        inputDims = [lfpdata.shape[1],lfpdata.shape[2]]
    #
    n_splits = 5
    n_repeats=1
    model_perf = np.zeros((n_splits*n_repeats,3))
    ypreds_out = []
    pedata_all_outs = []
    ctr = 0
    rkf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats)
    if modelflag == 'CNN':
        BOp_Params = [{"name":"lr","type":"range","bounds":[1e-5,1e-4],"value_type":"float"},\
          {"name":"NLinLayers","type":"range","bounds":[1,4],"value_type":"int"},\
          {"name":"ActFn","type":"choice","values":['ReLU','Tanh'],"value_type":"str"},\
          {"name":"MxPl","type":"range","bounds":[2,5],"value_type":"int"}]
    else:
        BOp_Params = [{"name":"lr","type":"range","bounds":[1e-5,1e-4],"value_type":"float"},\
          {"name":"NLinLayers","type":"range","bounds":[1,4],"value_type":"int"},\
          {"name":"ActFn","type":"choice","values":['ReLU','Tanh'],"value_type":"str"},\
          {"name":"hls","type":"range","bounds":[10,100],"value_type":"int"},\
          {"name":"drpOut","type":"range","bounds":[0,0.5],"value_type":"float"},\
          {"name":"nLayers","type":"range","bounds":[2,6],"value_type":"int"}]
    split_testMetrics = []
    best_params_List = []
    #
    val = val_data
    modelOI = modelflag
    for i, (train_index, test_index) in enumerate(rkf.split(np.array(range(lfpdata_0.shape[0])).reshape(-1,1),val_data)):
      if i!=0 and i % n_splits == 0:
        ctr +=1
      train_X0 = train_index.reshape(-1,1)
      test_X0 = test_index.reshape(-1,1)
      test_X = lfpdata[test_X0[:,0],:,:]
      train_X = lfpdata[train_X0[:,0],:,:]
      train_y = val_data[train_X0[:,0]].astype('long')
      test_y = val_data[test_X0[:,0]].astype('long')
      training_Data = [train_X0,test_X0,train_y,test_y]
      #
      best_params,values,experiment,model = optimize(parameters = BOp_Params,\
      evaluation_function = BayesOpt_Trainer,objective_name='Acc',total_trials=100)
      best_params_List.append(best_params)
      if modelOI == 'RNN':
          model0 = RNNNet(inputDims[1],best_params['hls'],best_params['nLayers'],\
          np.unique(val_data).shape[0],best_params['drpOut'],best_params['NLinLayers'],best_params['ActFn'])
      else:
          model0 = ConvNet(inputDims,np.unique(val_data).shape[0],best_params['NLinLayers'],\
          best_params['ActFn'],best_params['MxPl'])
      scalers,model,training_Data = Trainer(lfpdata,model0,val_data,training_Data,device,best_params['lr'])
      train_X0,test_X0,train_y0,test_y0,trn_lss,tst_lss,train_X0_ind,test_X0_ind = training_Data
      #
      tmp_test_outs = TestModel(test_X0,test_y0,scalers,model,device)
      # tmp_test_outs = TestModel(test_X,test_y,tmp_scalers,model,device)
      split_testMetrics.append(tmp_test_outs)
    # save stuff
    pickle.dump([split_testMetrics,best_params_List],open('{}LFP_ModelType_{}_Training_job_{}_vflag_{}_wSpks_{}_6Bands.pickle'.format(savedir,modelflag,jobid,valflag,spkflag),'wb'))
