import torch
from torch.utils.data import DataLoader
import numpy as np
import time, datetime, os
import mlflow, shutil, argparse
import hydra
from omegaconf import DictConfig

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.utils import CFG, criterion_list, predict_time_list, deviceDecision, seed_everything, modelDecision
from utils.dataloader import load_data, create_dataset
from utils.visualization import Eva, print_model_summary, mlflow_summary
from utils.earlystopping import EarlyStopping


# @hydra.main(config_path="hydra/config.yaml")
def main():
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
    parser.add_argument("--model", type=str, default="bt", help="model name")
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

    cfg = CFG(args)
    if os.path.isdir(cfg.RESULT_PATH):
        shutil.rmtree(cfg.RESULT_PATH)

    """Prepare"""
    seed_everything(seed=args.seed)
    data, csv_files = load_data(cfg, args.dataset, in_len=args.in_len, out_len=args.out_len, debug=args.debug)
    data_len = len(data["inp"])
    eva = Eva(data)
    device = deviceDecision()

    train_index_list, test_index_list = train_test_split(np.arange(data_len), test_size=0.2)
    train_index_list, val_index_list = train_test_split(train_index_list, test_size=0.1)

    train_dataset, mean_list, std_list = create_dataset(cfg, args.dataset, data, train_index_list, csv_files, is_train=True)
    val_dataset, _, _ = create_dataset(cfg, args.dataset, data, val_index_list, csv_files, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=os.cpu_count())

    model = modelDecision(args, cfg)
    model.to(device)

    criterion = criterion_list[args.criterion]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(path=cfg.RESULT_PATH, patience=args.patience, delta=args.delta, verbose=True)
    epoch_num = args.train_epochs if not args.debug else 3

    """Train"""
    print_model_summary(args, device, len(train_index_list), len(val_index_list))
    mlflow_summary(cfg, args)
    train_start_time = time.perf_counter()
    for epoch in range(1, epoch_num + 1):
        print("----------------------------------------------")
        print(f"[Epoch {epoch}]")
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inp = batch["inp"].to(device)
            spec = batch["spec"].to(device)
            gt = batch["gt"].to(device)

            # train pred here
            if ("decoder" in args.model) or ("Crossformer" in args.model):
                pred = model(inp, spec, gt)
            else:
                pred = model(inp, spec)

            loss = criterion(pred, gt)
            loss.backward()

            epoch_loss += loss.item() * inp.size(0)
            optimizer.step()

        epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Train Loss: {epoch_loss}")

        with torch.no_grad():
            model.eval()
            epoch_test_error = 0
            for batch in tqdm(val_dataloader):
                inp = batch["inp"].to(device)
                spec = batch["spec"].to(device)
                gt = batch["gt"].to(device)

                # validation pred here
                if ("decoder" in args.model) or ("Crossformer" in args.model):
                    pred = model(inp, spec, gt)
                else:
                    pred = model(inp, spec)

                test_error = torch.mean(torch.abs(gt - pred))
                epoch_test_error += test_error.item() * inp.size(0)

            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f"Val Loss: {epoch_test_error}")

        mlflow.log_metric(f"train loss", epoch_loss, step=epoch)
        mlflow.log_metric(f"val loss", epoch_test_error, step=epoch)

        os.makedirs(cfg.RESULT_PATH, exist_ok=True)
        early_stopping(epoch_test_error, model, epoch)
        if early_stopping.early_stop:
            break

    train_end_time = time.perf_counter()
    train_time = datetime.timedelta(seconds=(train_end_time - train_start_time))
    mlflow.log_metric(f"train time", train_end_time - train_start_time)

    print("----------------------------------------------")
    print(f"Train Time: {train_time}")

    """Test"""
    with torch.no_grad():
        print(f"\n\nTest Start. Case Num: {len(test_index_list)}")
        print("----------------------------------------------")
        model.load_state_dict(torch.load(early_stopping.path))
        model.eval()

        for test_index in tqdm(test_index_list):
            case_name = f"case{str(test_index+1).zfill(4)}"

            inp_data = data["inp"][test_index]  # shape(3299-(in_len+out_len), in_len, num_all_features)
            spec_data = data["spec"][test_index]  # shape(3299-(in_len+out_len), out_len, num_control_features)
            gt_data = data["gt"][test_index]  # shape(3299-(in_len+out_len), out_len, num_pred_features)
            gt_output_data = data["gt_data"][test_index]

            scaling_input_data = inp_data[0].copy()  # shape(in_len, num_all_features)
            scaling_spec_data = spec_data.copy()
            scaling_gt_data = gt_data.copy()
            for i in range(scaling_input_data.shape[1]):  # input scaling
                scaling_input_data[:, i] = (scaling_input_data[:, i] - mean_list[i + 1]) / std_list[i + 1]
            for i in range(scaling_spec_data.shape[2]):  # spec scaling
                scaling_spec_data[:, :, i] = (scaling_spec_data[:, :, i] - mean_list[i + 1]) / std_list[i + 1]
            for i in range(scaling_gt_data.shape[2]):  # ground truth scaling
                scaling_gt_data[:, :, i] = (scaling_gt_data[:, :, i] - mean_list[i + cfg.NUM_CONTROL_FEATURES + 1]) / std_list[i + cfg.NUM_CONTROL_FEATURES + 1]

            start_time = time.perf_counter()

            for i in range((gt_output_data.shape[0] - args.in_len) // args.out_len):
                input = torch.from_numpy(scaling_input_data[-args.in_len :].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i * args.out_len].astype(np.float32)).clone().unsqueeze(0).to(device)

                if ("decoder" in args.model) or ("Crossformer" == args.model):
                    scaling_pred_data = model.predict_func(input, spec)
                else:
                    scaling_pred_data = model(input, spec)  # test pred here
                scaling_pred_data = scaling_pred_data.detach().to("cpu").numpy().copy()[0]
                new_scaling_input_data = np.concatenate([scaling_spec_data[i * args.out_len], scaling_pred_data], axis=1)
                scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)

            end_time = time.perf_counter()
            predict_time_list.append(end_time - start_time)

            pred_output_data = scaling_input_data.copy()
            for i in range(pred_output_data.shape[1]):  # undo scaling
                pred_output_data[:, i] = pred_output_data[:, i] * std_list[i + 1] + mean_list[i + 1]
            pred_data = pred_output_data[:, cfg.NUM_CONTROL_FEATURES :]

            gt_output_data = gt_output_data[: pred_data.shape[0]]

            eva.evaluation(args.in_len, gt_output_data, pred_data)
            eva.visualization(gt_output_data, pred_data, case_name, cfg.RESULT_PATH)

    eva.save_evaluation()
    mlflow.log_metric(f"predict time mean", np.mean(predict_time_list))
    mlflow.log_artifacts(local_dir=cfg.RESULT_PATH, artifact_path="result")
    shutil.rmtree(cfg.RESULT_PATH)
    mlflow.end_run()
    print("----------------------------------------------")
    print(f"Predict Time: {np.mean(predict_time_list)} [s]\n")
    print("Experiment End")


if __name__ == "__main__":
    main()
