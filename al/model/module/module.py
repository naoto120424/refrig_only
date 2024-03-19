import torch
from torch import nn


class LossPred_Module(nn.Module):
    def __init__(self, args, cfg):
        super(LossPred_Module, self).__init__()
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.out_len = args.out_len
        self.d_model = args.d_model

        self.width = args.width

        if args.model == "dot" or args.model == "dol":
            self.ll_pred = nn.Sequential(nn.LayerNorm(self.out_len * self.num_pred_features * self.width * 2), nn.Linear(self.out_len * self.num_pred_features * self.width * 2, 1))
        elif args.model == "lstm" or args.model == "bt":
            self.ll_pred = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))

    def forward(self, x):
        return self.ll_pred(x)
