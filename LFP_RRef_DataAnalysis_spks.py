import numpy as np
from scipy.io import loadmat
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.model_selection import cross_validate,KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import os
import copy
from training_methods import MB_DecodingCombs


if __name__ == '__main__':
    jobid = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    # lfp_data_dir = '/home/tr561/ShiraStuff/LFP_Data/'
    spk_data_dir = '/projects/vm430_1/Mlab/TevinFolder/NewData_PhaseAmp/'
    spk_files0 = os.listdir(spk_data_dir)
    spk_files = [spk_data_dir + f for f in spk_files0]
    main_data_dir = '/projects/vm430_1/Mlab/TevinFolder/onlyBins/'
    main_dirs = sorted(os.listdir(main_data_dir))
    # all_session_Ids = [extract_identifier(f) for f in os.listdir(main_data_dir + main_dirs[0])]
    savedir = '/projects/vm430_1/Mlab/TevinFolder/RRef_Analysis_Results/'
    current_dir = main_data_dir + main_dirs[jobid-1]
    current_files = sorted(os.listdir(current_dir))
    if jobid == 1:
        # all_mp_ = MB_Decoding(main_data_dir,spk_files)
        # pickle.dump([all_mp_],open('{}Model_perf_MCMB_PCA_wSpks.pickle'.format(savedir),'wb'))
        # MB_DecodingCombs()
        all_mp_ = MB_DecodingCombs(main_data_dir,spk_files)
        pickle.dump([all_mp_],open('{}Model_perf_MCMB_PCA_wSpksCombinations.pickle'.format(savedir),'wb'))
    # for ix,f in enumerate(current_files):
    #     try:
    #         current_sess_id = extract_identifier(f)
    #         spk_fileOI = [fs1 for fs1 in spk_files if current_sess_id in fs1][0]
    #         data_spks = h5py.File(spk_fileOI,'r')
    #         spks = np.squeeze(np.transpose(data_spks['spk'][:],(3,2,1,0))[:,:,:,1]) # FIX THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         a1 =h5py.File(current_dir+os.sep+f,'r')
    #         ampBin = a1['ampBin']
    #         info = a1['info']
    #         removeTrials = info['removeTr'][:].reshape(-1)
    #         idx11 = np.nonzero(removeTrials == 0)[0]
    #         spks = spks[:,idx11,:]
    #         spks = np.transpose(spks,(1,2,0))
    #         val1 = three_vals(info['val']['val1'][:].reshape(-1))
    #         val2 = three_vals(info['val']['val2'][:].reshape(-1))
    #         # chval = three_vals(info['val']['chosenVal'][:].reshape(-1))
    #         fix1chosen = info['val']['firstChosen'][:].reshape(-1)
    #         amp = np.transpose(ampBin['fix2'][:],(1,0,2))
    #         # amp = np.concatenate((amp,spks),axis=1)
    #         #
    #         mp_= np.zeros((amp.shape[2],3))
    #         mp_spks = np.zeros_like(mp_)
    #         # try:
    #         for iy,v in enumerate([val1,val2,fix1chosen]):
    #             # looping over time ....
    #             for i in range(amp.shape[2]):
    #                 X = amp[:,:,i]
    #                 X_spks = spks[:,:,i]
    #                 X = PCA(n_components = 0.9).fit_transform(StandardScaler().fit_transform(X))
    #                 X_spks = PCA(n_components = 0.9).fit_transform(StandardScaler().fit_transform(X_spks))
    #                 cv = cross_validate(LinearDiscriminantAnalysis(),X,v,cv=KFold(n_splits=5,shuffle=True),scoring=['accuracy'])
    #                 mp_[i,iy] = np.mean(cv['test_accuracy'])
    #                 #
    #                 cv_s = cross_validate(LinearDiscriminantAnalysis(),X_spks,v,cv=KFold(n_splits=5,shuffle=True),scoring=['accuracy'])
    #                 mp_spks[i,iy] = np.mean(cv_s['test_accuracy'])
    #         pickle.dump([mp_,mp_spks,X.shape[1]],open('{}Model_perf_PCA_{}_File{}_spks.pickle'.format(savedir,main_dirs[jobid-1],ix+1),'wb'))
    #     except:
    #         print('failure on file number {} in directory {}'.format(ix+1,main_dirs[jobid-1]))
