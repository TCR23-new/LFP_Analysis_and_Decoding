import numpy as np
import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self,inputDims,num_classes,NumLinLayers,ActFnctFlag,MxPl_Size):
        super(ConvNet,self).__init__()
        self.inputDims = inputDims
        self.conv1 =nn.Conv1d(self.inputDims[0],100,kernel_size=2,stride=1)#16
        self.norm1 = nn.BatchNorm1d(100)#Identity() #B
        self.conv2 = nn.Conv1d(100,50,kernel_size=2,stride=1)#2
        self.norm2 = nn.BatchNorm1d(50)
        self.conv3 = nn.Conv1d(50,25,kernel_size=2,stride=1)#2
        self.norm3 = nn.BatchNorm1d(25)
        self.pool = nn.MaxPool1d(MxPl_Size) #16 4
        input_shape = self.flatten_shape()
        ta1 = []
        for i in range(NumLinLayers):
            ta1.append(('lin{}'.format(i+1),nn.Linear(input_shape,input_shape)))
            if ActFnctFlag == 'Tanh':
                ta1.append(('act{}'.format(i+1),nn.Tanh()))
            else:
                ta1.append(('act{}'.format(i+1),nn.ReLU()))
        ta1.append(('out_layer',nn.Linear(input_shape,num_classes)))
        self.lin = nn.Sequential(
         OrderedDict(ta1)
        )
        if ActFnctFlag == 'Tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()
        self.act2 = nn.Softmax(dim=1)
    def flatten_shape(self):
        input = torch.zeros((1,self.inputDims[0],self.inputDims[1]))
        out = self.pool(self.conv3(self.conv2(self.conv1(input))))
        return out.reshape(-1).shape[0]
    def forward(self,x):
        out1 = self.act(self.norm1(self.conv1(x)))
        out2 = self.act(self.norm2(self.conv2(out1)))
        out3 = self.act(self.norm3(self.conv3(out2)))
        pool_out = self.act(self.pool(out3))
        final_out = self.act2(self.lin(pool_out.reshape(x.shape[0],-1)))
        return final_out


class RNNNet(nn.Module):
    def __init__(self,inputDim,hiddenLayerSize,nLayers,num_classes,drpOut,NumLinLayers,ActFnctFlag):
        super(RNNNet,self).__init__()
        self.inputDim = inputDim
        self.hls = hiddenLayerSize
        self.nLayers = nLayers
        self.rnn = nn.LSTM(self.inputDim,self.hls,self.nLayers,dropout=drpOut)
        ta1 = []
        for i in range(NumLinLayers):
            ta1.append(('lin{}'.format(i+1),nn.Linear(self.hls,self.hls)))
            if ActFnctFlag == 'Tanh':
                ta1.append(('act{}'.format(i+1),nn.Tanh()))
            else:
                ta1.append(('act{}'.format(i+1),nn.ReLU()))
        ta1.append(('out_layer',nn.Linear(self.hls,num_classes)))
        self.outlayer = nn.Sequential(
         OrderedDict(ta1)
        )
        self.act = nn.Softmax(dim=1)
    def forward(self,x):
        out1,_ = self.rnn(x)
        out1=out1[:,-1,:]
        out2 = self.outlayer(out1)
        return self.act(out2)


class LFPNet(nn.Module):
    def __init__(self,numBands,output_dim):
        super(LFPNet,self).__init__()
        # self.drpRate = 0.3#0.4
        self.output_dim = output_dim
        self.net = nn.Sequential(
           nn.Conv1d(numBands,20,kernel_size=2,stride = 1),
           nn.BatchNorm1d(20),
           nn.SELU(),#ReLU(),
           nn.Conv1d(20,40,kernel_size=2,stride = 1),
           nn.BatchNorm1d(40),
           nn.SELU(),#ReLU(),
           nn.Conv1d(40,60,kernel_size=2,stride = 1),
           nn.BatchNorm1d(60),
           nn.SELU(),#ReLU(),
           nn.AdaptiveMaxPool1d(1),
           nn.SELU(),#ReLU(),
        )
        self.pool2 = nn.AdaptiveMaxPool1d(self.output_dim)
        self.act = nn.Softmax(dim=1)
    def forward(self,x1):
        out1 = self.net(x1).permute(0,2,1)
        out2 = self.act(self.pool2(out1).squeeze(1))
        return out2


class Spk_LFP_Net2(nn.Module):
    def __init__(self,numBands,numCells,hidden_dim,output_dim):
        super(Spk_LFP_Net2,self).__init__()
        self.numBands = numBands
        self.numCells = numCells
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Spike component
        self.spk_comp = nn.Sequential(
        nn.Linear(self.numCells,self.hidden_dim),
        nn.BatchNorm1d(self.hidden_dim),
        nn.SELU(),
        nn.Linear(self.hidden_dim,self.hidden_dim),
        nn.BatchNorm1d(self.hidden_dim),
        nn.SELU(),
        nn.Linear(self.hidden_dim,self.output_dim),
        nn.Softmax(dim=1)
        )
        # LFP component - low freq
        # self.lfp_comp_low = LFPNet(self.numBands//2,self.output_dim)
        self.lfp_comp_low = LFPNet(2,self.output_dim)
        # LFp component - high freq
        # self.lfp_comp_high=LFPNet(self.numBands//2,self.output_dim)
        self.lfp_comp_high=LFPNet(4,self.output_dim)
        #
        self.phase_comp = LFPNet(12,self.output_dim)
    def forward(self,x):
        # lfp_part2 = x[:,self.numBands//2:self.numBands,:]
        # lfp_part1 = x[:,:self.numBands//2,:]
        lfp_part1 = x[:,:2,:]
        lfp_part2 = x[:,2:self.numBands,:]
        spk_part = x[:,self.numBands,:]
        ph_part = x[:,self.numBands+1:,:]
        # compute the spk par
        out1 = self.spk_comp(spk_part)
        # compute the lfp part
        out2 = self.lfp_comp_low(lfp_part1)
        out3 = self.lfp_comp_high(lfp_part2)
        #
        out4 = self.phase_comp(ph_part)
        # compute the average
        # final_out = (out1 +out2 +out3+out4)/4#3
        return out1,out2,out3,out4
    def combined_pred(self,x):
        lfp_part1 = x[:,:2,:]
        lfp_part2 = x[:,2:self.numBands,:]
        spk_part = x[:,self.numBands,:]
        ph_part = x[:,self.numBands+1:,:]
        # compute the spk par
        out1 = self.spk_comp(spk_part)
        # compute the lfp part
        out2 = self.lfp_comp_low(lfp_part1)
        out3 = self.lfp_comp_high(lfp_part2)
        #
        out4 = self.phase_comp(ph_part)
        # compute the average
        final_out = (out1 +out2 +out3+out4)/4
        return final_out


class Spk_LFP_Net3(nn.Module):
    def __init__(self,numBands):
        super(Spk_LFP_Net3,self).__init__()
        self.net = nn.Sequential(
           nn.Conv1d(numBands,20,kernel_size=2,stride = 1),
           nn.SELU(),
           nn.Conv1d(20,40,kernel_size=2,stride = 1),
           nn.SELU(),
           nn.Conv1d(40,60,kernel_size=2,stride = 1),
           nn.SELU(),
        )
        self.lin = nn.Linear(60,3)
        self.trf = Trf_Rep2(60)
    def forward(self,x1):
        # x1 = x1.permute(0,2,1)
        out1 = self.net(x1).permute(0,2,1)
        out2 = self.lin(self.trf(out1))
        return out2


class Trf_Rep2(nn.Module):
  def __init__(self,d_mod,device=torch.device('cpu')):
    super(Trf_Rep2,self).__init__()
    self.d_mod = d_mod
    self.device = device
    self.drpRate= 0.5 #0.4
    self.dimff = d_mod
    self.nlayers = 2
    self.nheads = 1
    self.rnn = nn.Identity()
    self.encoder_layerT2 = nn.TransformerEncoderLayer(d_model=self.d_mod, nhead=self.nheads,dim_feedforward=self.dimff,dropout=self.drpRate,activation='gelu',batch_first=True)#0.2 relu
    self.transformer_encoderT2 = nn.TransformerEncoder(self.encoder_layerT2, num_layers=self.nlayers,enable_nested_tensor=False)#2
    self.act = nn.ReLU()
    self.clf_head_time = nn.Parameter(torch.rand(1,1,self.d_mod),requires_grad = True)
    self.apply(self._init_weights)
  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_uniform_(module.weight)
  def forward(self,x):
    x_Tpath= x
    clf_headT = self.clf_head_time.repeat(x.shape[0],1,1)
    new_input_T = torch.cat((clf_headT,x_Tpath),dim=1)
    new_input_T_pe = new_input_T
    trf_out_T = self.transformer_encoderT2(new_input_T_pe)
    lin_out = trf_out_T[:,0,:]
    return lin_out
