import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision
import resnet as resnet
import pdb
import numpy as np
'''
Feature_Extractor
'''
class WH_ACE(nn.Module):
    def __init__(self):
        super(WH_ACE, self).__init__()
    def forward(self, dense_pred, target):
        dense_prob = torch.softmax(dense_pred, dim = 2)
        nB = target.size(0)
        nClass = dense_prob.size(-1)
        hollow_params = []
        target_wasserstein = torch.zeros(nB, nClass).type_as(dense_prob.data).float()
        for i in range(0, nB):
            hollow_param = torch.ones(nClass).type_as(dense_prob.data).float()
            hollow_param[0] = 0
            tt = target[i].tolist()
            for j in range(0, len(tt)):
                if tt[j] == 0:
                    break
                hollow_param[tt[j]] = 0
            hollow_params.append(hollow_param)
        hollow_params = torch.stack(hollow_params, dim = 0)
        dense_prob = dense_prob.sum(0)
        # pdb.set_trace()
        whace_loss = (dense_prob * hollow_params).mean()
        return whace_loss

class CTC(nn.Module):
    def __init__(self):
        super(CTC, self).__init__()
        self.criterion = nn.CTCLoss(reduction='none', zero_infinity=True)
    def forward(self, input, label):
        batch_size, total_len = label.size()
        label_len = np.zeros(batch_size)
        label_seq = []
        label_total = 0;
        for bn in range(batch_size):
          for tn in range(total_len):
            if label[bn][tn] != -1:
              label_len[bn] = label_len[bn]+1
              label_total += 1
              label_seq.append(int(label[bn][tn])+1)
        label_seq = np.array(label_seq)
        label_len = Variable(torch.from_numpy(label_len).type(torch.IntTensor), requires_grad=False)
        label = Variable(torch.from_numpy(label_seq).type(torch.IntTensor), requires_grad=False)
        probs_sizes = Variable(torch.IntTensor([input.size(0)]*batch_size), requires_grad=False)
        torch.backends.cudnn.enabled = False
        loss = self.criterion(input.log_softmax(2), label.cuda(), probs_sizes, label_len).mean()/total_len
        torch.backends.cudnn.enabled = True
        return loss

class SqueezeNet(nn.Module):
    # Also known as Column Attention or Weighted Collapse or whatever else.
    # The core idea is emphasizing the character features on each column with softmax function.
    # Concentrate loss: making the attention map closer to one-hot.
    def __init__(self, in_size):
        super(SqueezeNet, self).__init__()
        self.f2map = nn.Conv2d(in_size, 1, 3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, input, temperature=1):
        squeezemap = F.relu(self.bn(self.f2map(input)))
        concentrate_loss = 1 / squeezemap.var(2).mean()
        squeezemap = F.softmax(squeezemap/temperature, 2)
        output = torch.sum(input*squeezemap, 2)
        return output, squeezemap, concentrate_loss

class ExCTC(nn.Module):
    def __init__(self, nClass):
        super(ExCTC, self).__init__()
        self.F = resnet.resnet18()
        self.S = SqueezeNet(512)
        self.C = nn.Linear(512, nClass)
        self.WH_ACE = WH_ACE()
        self.CTC = CTC()
    def forward(self, input, label, temperature=1, IFA_inference=False):
        # IFA_inference: set as True for full-page inference
        if not IFA_inference: 
            features = self.F(input)
            features = F.dropout(features, 0.5, training = self.training)

            features_1D, squeezemap, concentrate_loss = self.S(features, temperature)
            features_1D = features_1D.permute(2, 0, 1)
            output = self.C(features_1D)
            ctc_loss = self.CTC(output, label)

            nB, nC, nH, nW = features.shape
            outputdense = self.C(features.view(nB, nC, -1).permute(2, 0, 1))
            whace_loss = self.WH_ACE(outputdense, label+1)
            return output, squeezemap, ctc_loss, whace_loss, concentrate_loss
        else:
            features = self.F(input)
            features = features.permute(2,3,0,1)
            output = self.C(features)
            return output    





