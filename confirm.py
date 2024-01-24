import torch, argparse

from utils.utils import *
from utils.dataloader import *

from sklearn.model_selection import train_test_split

from model.BaseTransformer.base_transformer import BaseTransformer
from model.crossformer.cross_former import Crossformer
from model.linear.dlinear import Model
from model.s4.s4 import S4Block
from model.s4.s4d import S4D
from model.DeepOTransformer.deepotransformer import DeepOTransformer
from model.DeepOTransformer.deepolstm import DeepOLSTM

parser = argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project")

parser.add_argument("--e_name", type=str, default="Mazda Refrigerant Circuit", help="experiment name")
parser.add_argument("--model", type=str, default="BaseTransformer", help="model name")
parser.add_argument("--debug", type=bool, default=False, help="debug")
parser.add_argument("--seed", type=int, default=42, help="seed")

parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--train_epochs", type=int, default=20, help="train epochs")
parser.add_argument("--criterion", type=str, default="MSE", help="criterion name. MSE / L1")

parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--delta", type=float, default=0.0, help="early stopping delta")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="optimizer initial learning rate")  # add

parser.add_argument("--in_len", type=int, default=12, help="input MTS length (T)")  # change look_back -> in_len
parser.add_argument("--out_len", type=int, default=12, help="output MTS length (\tau)")  # add out_len

parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers (N)")  # change depth -> e_layers
parser.add_argument("--d_model", type=int, default=256, help="dimension of hidden states (d_model)")  # change dim -> d_model
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads of multi-head Attention")  # change heads -> n_heads
parser.add_argument("--d_ff", type=int, default=512, help="dimension of MLP in transformer")  # change fc_dim -> d_ff

parser.add_argument("--seg_len", type=int, default=3, help="(CrossFormer) segment length (L_seg)")
parser.add_argument("--win_size", type=int, default=2, help="(CrossFormer) window size for segment merge")
parser.add_argument("--factor", type=int, default=10, help="(CrossFormer) num of routers in Cross-Dimension Stage of TSA (c)")

parser.add_argument("--trunk_layers", type=int, default=3, help="(DeepOx) num of Trunk Net Layers")
parser.add_argument("--trunk_d_model", type=int, default=256, help="(DeepOx) Trunk Net dimension")

args = parser.parse_args()

seed_everything(seed=args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# model = Crossformer(CFG, args)
model = BaseTransformer(CFG, args)
# model = Model(CFG, args)
# model = S4Block(50)
# model = S4D(args.d_model)
# model = DeepOTransformer(CFG, args)
# model = DeepOLSTM(CFG, args)

criterion = criterion_list[args.criterion]

model.to(device)
model.train()
x_inp = torch.rand(args.bs, args.in_len, CFG.NUM_ALL_FEATURES).to(device)
spec = torch.rand(args.bs, args.out_len, CFG.NUM_CONTROL_FEATURES).to(device)
timedata = torch.rand(args.bs, args.out_len).to(device)
gt = torch.rand(args.bs, args.out_len, CFG.NUM_PRED_FEATURES).to(device)
# pred, _ = model(x_inp, spec, timedata)
pred, _ = model(x_inp, spec)
print(f"pred: {pred.shape}")
loss = criterion(pred, gt)


"""
for model_name in model_list:
    args.model = model_name
    print(f"{model_name} check start")
    model = modelDecision(args, CFG)
    model.to(device)
    model.eval()
    inp = torch.rand(args.bs, args.in_len, CFG.NUM_ALL_FEATURES).to(device)
    spec = torch.rand(args.bs, CFG.NUM_CONTROL_FEATURES).to(device)
    gt = torch.rand(args.bs, CFG.NUM_PRED_FEATURES).to(device)
    if "Transformer" == args.model:
        pred = model(inp, spec, gt)
    else:
        pred, attn = model(inp, spec)
    print(f"{model_name} check complete\n")
"""
