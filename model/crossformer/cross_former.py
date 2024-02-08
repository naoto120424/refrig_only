import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from copy import deepcopy

from model.crossformer.cross_encoder import Encoder
from model.crossformer.cross_encoder import Encoder
from model.crossformer.cross_decoder import Decoder
from model.crossformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from model.crossformer.cross_embed import DSW_embedding

from model.BaseTransformer.spec_embed import SpecEmbedding

from math import ceil


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class Crossformer(nn.Module):
    def __init__(self, cfg, args):
        super(Crossformer, self).__init__()
        self.data_dim = cfg.NUM_ALL_FEATURES
        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.dim = args.d_model
        self.depth = args.e_layers
        self.win_size = args.win_size
        self.heads = args.n_heads
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.fc_dim = args.d_ff
        self.seg_len = args.seg_len
        self.merge_win = args.win_size
        self.dropout = args.dropout
        self.factor = args.factor

        # self.baseline = baseline

        # self.device = device

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, self.dim)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), self.dim))
        self.pre_norm = nn.LayerNorm(self.dim)

        self.spec_embedding = DSW_embedding(self.seg_len, self.dim)

        # Encoder
        self.encoder = Encoder(self.depth, self.win_size, self.dim, self.heads, self.fc_dim, block_depth=1, dropout=self.dropout, in_seg_num=(self.pad_in_len // self.seg_len), factor=self.factor)

        # Decoder
        self.dec_value_embedding = DSW_embedding(self.seg_len, self.dim)
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), self.dim))
        self.decoder = Decoder(self.seg_len, self.depth + 1, self.dim, self.heads, self.fc_dim, self.dropout, out_seg_num=(self.pad_out_len // self.seg_len), factor=self.factor)

    def forward(self, x, spec, tgt):
        # if (self.baseline):
        #     base = x_seq.mean(dim = 1, keepdim = True)
        # else:
        #     base = 0

        # print(x.shape, spec.shape, tgt.shape)
        inp = x
        batch_size = x.shape[0]
        if self.in_len_add != 0:
            x = torch.cat((x[:, :1, :].expand(-1, self.in_len_add, -1), x), dim=1)

        x = self.enc_value_embedding(x)
        # print("x embed: ", x.shape)
        # x_spec = self.spec_embedding(spec)
        # print("spec embed: ", x_spec.shape)
        # x = torch.cat((x, x_spec), dim=1)
        # print(x.shape, self.enc_pos_embedding.shape)
        x += self.enc_pos_embedding
        x = self.pre_norm(x)

        enc_out = self.encoder(x)
        # print("enc_out.shape: ", len(enc_out), enc_out[0].shape)

        tgt = torch.cat((spec, tgt), dim=2)
        y = torch.cat((inp[:, -1:, :], tgt), dim=1)[:, :-1, :]

        dec_in = self.dec_value_embedding(y)
        dec_in += repeat(self.dec_pos_embedding, "b ts_d l d -> (repeat b) ts_d l d", repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        # print("predict shape: ", predict_y.shape)

        return predict_y[:, : self.out_len, self.num_control_features :], None

    def predict_func(self, x, spec):
        batch_size = x.shape[0]
        dec_in = x[:, -1:, :]

        x = self.enc_value_embedding(x)
        x += self.enc_pos_embedding
        x = self.pre_norm(x)

        enc_out = self.encoder(x)

        dec_in = self.dec_value_embedding(dec_in)
        dec_in += repeat(self.dec_pos_embedding, "b ts_d l d -> (repeat b) ts_d l d", repeat=batch_size)[dec_in.shape]
        predict_y = self.decoder(dec_in, enc_out)

        return predict_y[:, : self.out_len, self.num_control_features :], None
