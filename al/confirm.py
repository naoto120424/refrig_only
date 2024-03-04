import torch, argparse

from utils.utils import CFG, criterion_list, predict_time_list, deviceDecision, seed_everything, modelDecision
from utils.dataloader import load_data, create_dataset

from sklearn.model_selection import train_test_split

"""parser"""
parser = argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project Step3")

""" exp setting """
parser.add_argument("--e_name", type=str, default="Refrigerant Only", help="experiment name")
parser.add_argument("--dataset", type=str, default="refrig_only", help="dataset. refrig_only / teacher")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--debug", type=bool, default=False, help="debug")

parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="optimizer initial learning rate")
parser.add_argument("--criterion", type=str, default="mse", help="criterion name. MSE / L1")

""" early stopping """
parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
parser.add_argument("--delta", type=float, default=0.0, help="early stopping delta")

""" model """
parser.add_argument("--model", type=str, default="BaseTransformer", help="model name")
parser.add_argument("--in_len", type=int, default=10, help="input length")
parser.add_argument("--out_len", type=int, default=1, help="output length")

""" for all models """
parser.add_argument("--d_model", type=int, default=256, help="dimension of hidden states (d_model)")
parser.add_argument("--e_layers", type=int, default=3, help="num main module layers (N)")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

""" Transformer """
parser.add_argument("--n_heads", type=int, default=8, help="num of heads of multi-head Attention")
parser.add_argument("--d_ff", type=int, default=512, help="dimension of MLP in transformer")

""" Crossformer """
parser.add_argument("--seg_len", type=int, default=3, help="(CrossFormer) segment length (L_seg)")
parser.add_argument("--win_size", type=int, default=2, help="(CrossFormer) window size for segment merge")
parser.add_argument("--factor", type=int, default=10, help="(CrossFormer) num of routers in Cross-Dimension Stage of TSA (c)")

""" DeepOx """
parser.add_argument("--trunk_layers", type=int, default=3, help="(DeepOx) num of Trunk Net Layers")
parser.add_argument("--branch_layers", type=int, default=1, help="(DeepOx) num of Branch Net Layers")
parser.add_argument("--width", type=int, default=128, help="(DeepOx) Trunk Net and Branch Net dimension")

args = parser.parse_args()


seed_everything(seed=args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG(args)
model = modelDecision(args, cfg)

criterion = criterion_list[args.criterion]

model.to(device)
model.train()
x_inp = torch.rand(args.bs, args.in_len, cfg.NUM_ALL_FEATURES).to(device)
spec = torch.rand(args.bs, args.out_len, cfg.NUM_CONTROL_FEATURES).to(device)
gt = torch.rand(args.bs, args.out_len, cfg.NUM_PRED_FEATURES).to(device)

if ("decoder" in args.model) or ("Crossformer" in args.model):
    pred, _ = model(x_inp, spec, gt)
else:
    pred, _ = model(x_inp, spec)
print(f"pred: {pred.shape}")
loss = criterion(pred, gt)

if ("decoder" in args.model) or ("Crossformer" in args.model):
    out, _ = model.predict_func(x_inp[-1:], spec[-1:])
    print(f"out: {out.shape}")

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
