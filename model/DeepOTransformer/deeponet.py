import torch
from torch import nn
from einops import rearrange, repeat


class DeepONet(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.in_len = args.in_len
        self.out_len = args.out_len

        self.branch_layers = args.branch_layers
        self.trunk_layers = args.trunk_layers
        self.width = args.width

        self.inp_dense = nn.Linear(self.num_all_features, args.d_model)
        self.spec_dense = nn.Linear(self.num_control_features, args.d_model)

        self.branch = nn.ModuleDict()
        self.branch["LinM1"] = nn.Linear((self.in_len + self.out_len) * args.d_model, self.out_len * self.num_pred_features * self.width)
        self.branch["NonM1"] = nn.ReLU()
        for i in range(2, self.branch_layers):
            self.branch["LinM{}".format(i)] = nn.Linear(self.out_len * self.num_pred_features * self.width, self.out_len * self.num_pred_features * self.width)
            self.branch["NonM{}".format(i)] = nn.ReLU()
        self.branch["LinMout"] = nn.Linear(self.out_len * self.num_pred_features * self.width, self.out_len * self.num_pred_features * self.width)

        self.trunk = nn.ModuleDict()
        self.trunk["LinM1"] = nn.Linear(self.num_all_features, self.num_pred_features * self.width)
        self.trunk["NonM1"] = nn.ReLU()
        for i in range(2, self.trunk_layers + 1):
            self.trunk["LinM{}".format(i)] = nn.Linear(self.num_pred_features * self.width, self.num_pred_features * self.width)
            self.trunk["NonM{}".format(i)] = nn.ReLU()

        self.params = nn.ParameterDict()
        self.params["bias"] = nn.Parameter(torch.zeros([self.num_pred_features]))

    def forward(self, inp, spec, h=None):
        inp_d = self.inp_dense(inp)
        spec_d = self.spec_dense(spec)

        x = torch.cat([inp_d, spec_d], dim=1)
        x = rearrange(x, "bs l d -> bs (l d)")

        if self.branch_layers == 1:
            LinM = self.branch["LinM1"]
            x = LinM(x)
            x = rearrange(x, "bs (out_len n d) -> bs out_len n d", out_len=self.out_len, n=self.num_pred_features)
        else:
            for i in range(1, self.branch_layers):
                LinM = self.branch["LinM{}".format(i)]
                NonM = self.branch["NonM{}".format(i)]
                x = NonM(LinM(x))
            x = self.branch["LinMout"](x)
            x = rearrange(x, "bs (out_len n d) -> bs out_len n d", out_len=self.out_len, n=self.num_pred_features)

        y = inp[:, -1:, :]

        for t in range(1, self.out_len + 1):
            y_out = y
            for i in range(1, self.trunk_layers + 1):
                LinM = self.trunk["LinM{}".format(i)]
                NonM = self.trunk["NonM{}".format(i)]
                y_out = NonM(LinM(y_out))

            y_out = rearrange(y_out, "bs l (n d) -> bs l n d", n=self.num_pred_features)
            y_out = torch.nansum(x[:, :t] * y_out, dim=-1, keepdim=False) + self.params["bias"]

            y_out = torch.cat((spec[:, :t], y_out), dim=-1)
            y = torch.cat((y, y_out[:, -1:]), dim=1)

        return y[:, 1:, self.num_control_features :], None
