# -*- coding: utf-8 -*-
#""""""""""""""""""""""""""""""""""""" Powered By WHU MVP Lab """"""""""""""""""""""""""""""""""""""""""""""""
# @DESC:
#    - CAN: Correspondence Attention Transformer
#
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import torch
import torch.nn as nn
from loss import batch_episym
from modules import Architecture

class CAN(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = Architecture(config.net_channels, 4, depth_each_stage)
        self.weights_iter = [Architecture(config.net_channels, 6, depth_each_stage) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        res_logits, res_e_hat, res_mean = [], [], []
        logits, e_hat = self.weights_init(data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        weight = torch.relu(torch.tanh(logits)).unsqueeze(1).unsqueeze(-1).repeat(1,1,1,2).detach()
        for i in range(self.iter_num):
            logits, e_hat = self.weights_iter[i](
                torch.cat([data['xs'], weight], dim=3)
                )
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat

