from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import CCA
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.preprocessing import StandardScaler
import os
from scipy.io import loadmat
from sklearn.decomposition import PCA
import h5py
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from preproc_methods import CollectBands_AmpPhase2,Value_CS_ShiraData,HighMidLow_Encoding
from preproc_methods import BetterFeatureEncoding
from training_methods import CV_Calc,CCA_Corr

if __name__ == '__main__':
    savedir = '/projects/vm430_1/Mlab/TevinFolder/LFP_Net_Outputs/'
    dir1 = '/projects/vm430_1/Mlab/TevinFolder/NewData_PhaseAmp/'
    dir_stim_variables = '/projects/vm430_1/Mlab/TevinFolder/behVars/'
    allfiles_cs_ = sorted(os.listdir(dir_stim_variables))
    mflag = 'c' #c k
    allfiles_cs = [f for f in allfiles_cs_ if '_post_' in f and '_{}'.format(mflag) in f]
    allfiles1_ = sorted(os.listdir(dir1))
    allfiles1 = [f for f in allfiles1_ if '_{}'.format(mflag) in f]
    ns,nr = 5,2 #20
    valflag = 1 # 0,1,2
    timebin = 1 #1 #0
    file_count = len(allfiles1)
    bandRanges_newdata = ['4to8Hz','8to14Hz','15to30Hz','30to50Hz','50to100Hz','100to200Hz']
    numBands_ = 3*len(bandRanges_newdata) + 1
    jbdidx = int(os.environ.get('SLURM_ARRAY_TASK_ID'))-1
    for jobid in [jbdidx]: #range(file_count):
        print("Current job: {}".format(jobid))
        alldata = h5py.File(dir1 + allfiles1[jobid],'r')#41
        cs_value_data = loadmat(dir_stim_variables + allfiles_cs[jobid])
        mp_base,mp_dimRed,dim_t = np.zeros((24,numBands_)),np.zeros((24,numBands_)),np.zeros((24,numBands_))
        recon_error,mp_cbetter = np.zeros((24,numBands_)),np.zeros((24,numBands_))
        mp_base2,mp_dimRed2,dim_t2 = np.zeros((24,numBands_)),np.zeros((24,numBands_)),np.zeros((24,numBands_))
        recon_error2,mp_cbetter2 = np.zeros((24,numBands_)),np.zeros((24,numBands_))
        #
        cca_fulldim =np.zeros((24,numBands_,numBands_))
        cca_decoding1_fulldim = np.zeros((24,2,numBands_,numBands_))
        cca_decoding2_fulldim = np.zeros((24,2,numBands_,numBands_))
        #
        mp_base_avg,mp_base2_avg = np.zeros((24,numBands_)),np.zeros((24,numBands_))
        for time in range(24):
            ampdata,phdata,spks = CollectBands_AmpPhase2(alldata,bandRanges_newdata,timebin,time)
            stim_ids1 = cs_value_data['firstId'][:]
            stim_ids2 = cs_value_data['secondId'][:]
            val_cs1 = Value_CS_ShiraData(stim_ids1)
            val1 = HighMidLow_Encoding(np.sum(val_cs1,axis=1))
            val_cs2 = Value_CS_ShiraData(stim_ids2)
            val2 = HighMidLow_Encoding(np.sum(val_cs2,axis=1))
            idx1_ = np.nonzero(np.sum(val_cs1,axis=1) == 3)[0]
            idx2_ = np.nonzero(np.sum(val_cs2,axis=1) == 3)[0]
            cbetter1 = BetterFeatureEncoding(val_cs1)
            cbetter2 = BetterFeatureEncoding(val_cs2)
            X_amp_only = ampdata
            X_spk_only = spks
            # X_Amp_Spk = np.concatenate((ampdata,spks),axis=1)
            X_Amp_Spk = np.concatenate((ampdata,spks,phdata),axis=1)
            #(ampdata_s,y,n_splits,n_repeats)
            cca_fulldim[time,:,:],cca_decoding1_fulldim[time,:,:,:] = CCA_Corr(X_Amp_Spk,val1,ns,nr)
            for i in range(X_Amp_Spk.shape[1]):
                # compute the dimensionality for all bands and spikes
                pipeline = Pipeline([('scaler',StandardScaler()),('pca',PCA(n_components = 0.95))]).fit(X_Amp_Spk[:,i,:])
                dim_t[time,i] = pipeline[1].n_components_
                pcs = pipeline.transform(X_Amp_Spk[:,i,:])
                # decode using base activity
                # cv = cross_validate(LinearDiscriminantAnalysis(),X_Amp_Spk[:,i,:],val1,cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
                cv =  CV_Calc(LinearDiscriminantAnalysis(),X_Amp_Spk[:,i,:],val1,ns,nr)
                mp_base[time,i] = np.mean(cv)
                #
                #if i != 7:
                cv_avg = CV_Calc(LinearDiscriminantAnalysis(),np.mean(X_Amp_Spk[:,i,:],axis=1).reshape(-1,1),val1,ns,nr)
                mp_base_avg[time,i] = np.mean(cv_avg)
                # decode using dim reduce activity
                # cv = cross_validate(LinearDiscriminantAnalysis(),pcs,val1,cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
                cv =  CV_Calc(LinearDiscriminantAnalysis(),pcs,val1,ns,nr)
                mp_dimRed[time,i] = np.mean(cv)
                # reconstruction error
                mse = mean_squared_error(X_Amp_Spk[:,i,:],pipeline.inverse_transform(pcs))
                recon_error[time,i] = mse
                # color better decoding
                # cv1 = cross_validate(LinearDiscriminantAnalysis(),X_Amp_Spk[idx1_,i,:],cbetter1[idx1_],cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
                cv1 = CV_Calc(LinearDiscriminantAnalysis(),X_Amp_Spk[idx1_,i,:],cbetter1[idx1_],ns,nr)
                mp_cbetter[time,i] = np.mean(cv1) #cv1['test_accuracy'])
                if timebin == 1:
                    if i == 0:
                        _,cca_decoding2_fulldim[time,:,:,:] = CCA_Corr(X_Amp_Spk,val2,ns,nr)
                    #
                    cv = CV_Calc(LinearDiscriminantAnalysis(),X_Amp_Spk[:,i,:],val2,ns,nr)
                    # cv = cross_validate(LinearDiscriminantAnalysis(),X_Amp_Spk[:,i,:],val2,cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
                    mp_base2[time,i] =np.mean(cv) # np.mean(cv['test_accuracy'])
                    # decode using dim reduce activity
                    #if i != 7:
                    cv_avg = CV_Calc(LinearDiscriminantAnalysis(),np.mean(X_Amp_Spk[:,i,:],axis=1).reshape(-1,1),val2,ns,nr)
                    mp_base2_avg[time,i] = np.mean(cv_avg)
                    # cv = cross_validate(LinearDiscriminantAnalysis(),pcs,val2,cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
                    cv = CV_Calc(LinearDiscriminantAnalysis(),pcs,val2,ns,nr)
                    mp_dimRed2[time,i] = np.mean(cv) #np.mean(cv['test_accuracy'])
                    # color better decoding
                    # cv1 = cross_validate(LinearDiscriminantAnalysis(),X_Amp_Spk[idx2_,i,:],cbetter2[idx2_],cv=RepeatedKFold(n_splits=ns,n_repeats=nr),scoring=['accuracy'])
                    cv1 = CV_Calc(LinearDiscriminantAnalysis(),X_Amp_Spk[idx2_,i,:],cbetter2[idx2_],ns,nr)
                    mp_cbetter2[time,i] = np.mean(cv1) #np.mean(cv1['test_accuracy'])
        #save outputs
        outputs = [cca_fulldim,mp_base,mp_dimRed,dim_t,recon_error,mp_cbetter,mp_base2,mp_dimRed2,dim_t2,
        recon_error2,mp_cbetter2,cca_decoding1_fulldim,cca_decoding2_fulldim,mp_base_avg,mp_base2_avg]
        savefilename = '{}FileID_{}_Monkey_{}_ValFlag_{}_timebin_{}.pickle'.format(savedir,jobid,mflag,valflag,timebin)
        pickle.dump(outputs,open(savefilename,'wb'))
