import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.ae.MemAE.modules import *

# 필요한 기본 모듈 정의


class up(nn.Module):
    def __init__(self, in_ch, out_ch, op):
        super(up, self).__init__()
        if op == "concat":
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.conv = double_conv(in_ch, out_ch)
        elif op == "none":
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.conv = double_conv(out_ch, out_ch)
        else:
            raise NotImplementedError

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class Memory(nn.Module):
    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))  # M,C
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M]
        att_weight = F.softmax(att_weight, dim=1)  # NxM

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            att_weight = F.normalize(att_weight, p=1, dim=1)  # [N,M]

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)


class ML_MemAE_SC(nn.Module):
    def __init__(self, num_in_ch, seq_len, feature_dim, features_root,
                 num_slots, shrink_thres,
                 mem_usage, skip_ops):
        super(ML_MemAE_SC, self).__init__()
        self.num_in_ch = num_in_ch
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.mem_usage = mem_usage
        self.num_mem = sum(mem_usage)
        self.skip_ops = skip_ops

        self.fc_in = nn.Linear(self.feature_dim, features_root)
        self.fc_out = nn.Linear(features_root, self.feature_dim)

        self.mem1 = Memory(num_slots=self.num_slots, slot_dim=features_root,
                           shrink_thres=self.shrink_thres) if self.mem_usage[1] else None
        self.mem2 = Memory(num_slots=self.num_slots, slot_dim=features_root,
                           shrink_thres=self.shrink_thres) if self.mem_usage[2] else None
        self.mem3 = Memory(num_slots=self.num_slots, slot_dim=features_root,
                           shrink_thres=self.shrink_thres) if self.mem_usage[3] else None

    def forward(self, x):
        """
        :param x: size [bs, seq_len, 1, feature_dim]
        :return:
        """
        bs, seq_len, _, _ = x.shape
        x = x.view(bs * seq_len, -1)

        x = self.fc_in(x)

        if self.mem_usage[3]:
            mem3_out = self.mem3(x)
            x = mem3_out["out"]
            att_weight3 = mem3_out["att_weight"]

        if self.mem_usage[2]:
            mem2_out = self.mem2(x)
            x = mem2_out["out"]
            att_weight2 = mem2_out["att_weight"]

        if self.mem_usage[1]:
            mem1_out = self.mem1(x)
            x = mem1_out["out"]
            att_weight1 = mem1_out["att_weight"]

        x = self.fc_out(x)
        x = x.view(bs, seq_len, 1, -1)

        if self.num_mem == 3:
            outs = dict(recon=x, att_weight3=att_weight3, att_weight2=att_weight2, att_weight1=att_weight1)
        elif self.num_mem == 2:
            outs = dict(recon=x, att_weight3=att_weight3, att_weight2=att_weight2,
                        att_weight1=torch.zeros_like(att_weight3))  # dummy attention weights
        elif self.num_mem == 1:
            outs = dict(recon=x, att_weight3=att_weight3,
                        att_weight2=torch.zeros_like(att_weight3),
                        att_weight1=torch.zeros_like(att_weight3))  # dummy attention weights
        return outs