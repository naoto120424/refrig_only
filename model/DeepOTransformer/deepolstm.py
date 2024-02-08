import torch
from torch import nn
from einops import rearrange, repeat


class DeepOLSTM(nn.Module):
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

        self.lstm = nn.LSTM(
            input_size=cfg.NUM_ALL_FEATURES,
            hidden_size=args.d_model,
            num_layers=args.e_layers,
            dropout=args.dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(cfg.NUM_CONTROL_FEATURES, args.d_model)

        self.branch_before = nn.Linear((self.out_len + 1) * args.d_model, args.d_model)
        self.branch = nn.ModuleDict()
        self.branch["LinM1"] = nn.Linear((self.out_len + 1) * args.d_model, self.out_len * self.num_pred_features * self.width)
        self.branch["NonM1"] = nn.ReLU()
        for i in range(2, self.branch_layers):
            self.branch["LinM{}".format(i)] = nn.Linear(self.out_len * self.num_pred_features * self.width, self.out_len * self.num_pred_features * self.width)
            self.branch["NonM{}".format(i)] = nn.ReLU()
        self.branch["LinMout"] = nn.Linear(self.width, self.width)

        self.trunk = nn.ModuleDict()
        for n in range(1, self.out_len + 1):
            self.trunk["L{}_LinM1".format(n)] = nn.Linear(n * self.num_all_features, n * self.num_pred_features * self.width)
            self.trunk["L{}_NonM1".format(n)] = nn.ReLU()
            for i in range(2, self.trunk_layers + 1):
                self.trunk["L{}_LinM{}".format(n, i)] = nn.Linear(n * self.num_pred_features * self.width, n * self.num_pred_features * self.width)
                self.trunk["L{}_NonM{}".format(n, i)] = nn.ReLU()

        self.params = nn.ParameterDict()
        for i in range(1, self.out_len + 1):
            self.params["L{}_bias".format(i)] = nn.Parameter(torch.zeros([i, self.num_pred_features]))

    def forward(self, inp, spec, h=None):
        hidden1, _ = self.lstm(inp, h)
        hidden2 = self.spec_dense(spec)

        x = torch.cat([hidden1[:, -1:, :], hidden2], dim=1)
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
            x = rearrange(x, "bs (out_len n d) -> bs out_len n d", out_len=self.out_len, n=self.num_pred_features)
            x = self.branch["LinMout"](x)

        y = inp[:, -1:, :]

        for t in range(1, self.out_len + 1):
            y_out = y
            y_out = rearrange(y_out, "bs len n -> bs (len n)")
            for i in range(1, self.trunk_layers + 1):
                LinM = self.trunk["L{}_LinM{}".format(t, i)]
                NonM = self.trunk["L{}_NonM{}".format(t, i)]
                y_out = NonM(LinM(y_out))

            y_out = rearrange(y_out, "bs (out_len n d) -> bs out_len n d", out_len=t, n=self.num_pred_features)
            y_out = torch.nansum(x[:, :t] * y_out, dim=-1, keepdim=False) + self.params["L{}_bias".format(t)]

            y_out = torch.cat((spec[:, :t], y_out), dim=-1)
            y = torch.cat((y, y_out[:, -1:]), dim=1)

        return y[:, 1:, self.num_control_features :], None
