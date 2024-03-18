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
        self.LN1 = nn.LayerNorm(normalized_shape=(1, args.d_model))
        self.LN2 = nn.LayerNorm(normalized_shape=(self.out_len, args.d_model))

        self.branch = nn.ModuleDict()
        if self.branch_layers == 1:
            self.branch["LinM1"] = nn.Linear(args.d_model * 2, self.num_pred_features * self.width)
        else:
            self.branch["LinM1"] = nn.Linear(args.d_model * 2, self.num_pred_features * args.d_model)
            self.branch["NonM1"] = nn.ReLU()
            for i in range(2, self.branch_layers):
                self.branch["LinM{}".format(i)] = nn.Linear(self.num_pred_features * args.d_model, self.num_pred_features * args.d_model)
                self.branch["NonM{}".format(i)] = nn.ReLU()
            self.branch["LinMout"] = nn.Linear(self.num_pred_features * args.d_model, self.num_pred_features * self.width)

        self.trunk = nn.ModuleDict()
        if self.trunk_layers == 1:
            self.trunk["LinM1"] = nn.Linear(self.num_all_features, self.num_pred_features * self.width)
            self.trunk["NonM1"] = nn.ReLU()
        else:
            self.trunk["LinM1"] = nn.Linear(self.num_all_features, self.num_pred_features * args.d_model)
            self.trunk["NonM1"] = nn.ReLU()
            for i in range(2, self.trunk_layers):
                self.trunk["LinM{}".format(i)] = nn.Linear(self.num_pred_features * args.d_model, self.num_pred_features * args.d_model)
                self.trunk["NonM{}".format(i)] = nn.ReLU()
            self.trunk["LinM{}".format(self.trunk_layers)] = nn.Linear(self.num_pred_features * args.d_model, self.num_pred_features * self.width)
            self.trunk["NonM{}".format(self.trunk_layers)] = nn.ReLU()

        self.params = nn.ParameterDict()
        self.params["bias"] = nn.Parameter(torch.zeros([self.num_pred_features]))

    def forward(self, inp, spec, h=None):
        hidden1, _ = self.lstm(inp, h)
        hidden2 = self.spec_dense(spec)
        hidden1 = self.LN1(hidden1[:, -1:, :])
        hidden2 = self.LN2(hidden2)

        hidden1 = repeat(hidden1, "bs l d -> bs (repeat l) d", repeat=self.out_len)
        x = torch.cat([hidden1, hidden2], dim=-1)

        if self.branch_layers == 1:
            LinM = self.branch["LinM1"]
            x = LinM(x)
            x = rearrange(x, "bs l (n d) -> bs l n d", n=self.num_pred_features)
        else:
            for i in range(1, self.branch_layers):
                LinM = self.branch["LinM{}".format(i)]
                NonM = self.branch["NonM{}".format(i)]
                x = NonM(LinM(x))
            x = self.branch["LinMout"](x)
            x = rearrange(x, "bs l (n d) -> bs l n d", n=self.num_pred_features)

        y = inp[:, -1:, :]

        for t in range(1, self.out_len + 1):
            y_out = y
            for i in range(1, self.trunk_layers + 1):
                LinM = self.trunk["LinM{}".format(i)]
                NonM = self.trunk["NonM{}".format(i)]
                y_out = NonM(LinM(y_out))

            y_out = rearrange(y_out, "bs l (n d) -> bs l n d", n=self.num_pred_features)
            y_ll_pred = y_out
            y_out = torch.sum(x[:, :t] * y_out, dim=-1, keepdim=False) + self.params["bias"]

            y_out = torch.cat((spec[:, :t], y_out), dim=-1)
            y = torch.cat((y, y_out[:, -1:]), dim=1)

        ll_pred = torch.cat((x, y_ll_pred), dim=1)
        ll_pred = rearrange(ll_pred, "bs l n d -> bs (l n d)")

        return y[:, 1:, self.num_control_features :], ll_pred
