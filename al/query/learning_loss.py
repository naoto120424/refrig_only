import torch
import numpy as np
from tqdm import tqdm

from model.module.module import LossPred_Module
from utils.utils import CFG, criterion_list, predict_time_list, deviceDecision, seed_everything, modelDecision


""" クエリ戦略のために、trainに未使用のデータに対して推論を行う """


def learning_loss(
    args,
    cfg,
    device,
    model_path,
    module_path,
    data,
    mean_list,
    std_list,
    labeled_indices,
    unlabeled_indices,
):
    print("\n\nLearning Loss Algorithm Start")
    print("----------------------------------------------")
    with torch.no_grad():
        n_add = int((len(labeled_indices) + len(unlabeled_indices)) * 0.1)
        model = modelDecision(args, cfg)
        module = LossPred_Module(args, cfg)

        model.load_state_dict(torch.load(model_path))
        module.load_state_dict(torch.load(module_path))

        model.to(device)
        module.to(device)

        model.eval()
        module.eval()

        loss_pred_index = []

        for unlabeled_index in tqdm(unlabeled_indices):
            inp_data = data["inp"][unlabeled_index]  # shape(3299-(in_len+out_len), in_len, num_all_features)
            spec_data = data["spec"][unlabeled_index]  # shape(3299-(in_len+out_len), out_len, num_control_features)

            scaling_input_data = inp_data[0].copy()  # shape(in_len, num_all_features)
            scaling_spec_data = spec_data.copy()
            for i in range(scaling_input_data.shape[1]):  # input scaling
                scaling_input_data[:, i] = (scaling_input_data[:, i] - mean_list[i + 1]) / std_list[i + 1]
            for i in range(scaling_spec_data.shape[2]):  # spec scaling
                scaling_spec_data[:, :, i] = (scaling_spec_data[:, :, i] - mean_list[i + 1]) / std_list[i + 1]

            loss_pred_list = []
            for i in range(inp_data.shape[0] // args.out_len):
                input = torch.from_numpy(scaling_input_data[-args.in_len :].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i * args.out_len].astype(np.float32)).clone().unsqueeze(0).to(device)

                scaling_pred_data, middle_pred_data = model(input, spec)  # test pred here
                scaling_pred_data = scaling_pred_data.detach().to("cpu").numpy().copy()[0]
                new_scaling_input_data = np.concatenate([scaling_spec_data[i * args.out_len], scaling_pred_data], axis=1)
                scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)

                loss_pred = module(middle_pred_data)
                loss_pred = loss_pred.mean(dim=0)
                loss_pred = loss_pred.detach().to("cpu").numpy().copy()
                loss_pred_list.append(loss_pred)

            loss_pred_index.append(np.mean(np.array(loss_pred_list)))

        loss_pred_index = np.argsort(loss_pred_index)
        loss_pred_index = loss_pred_index[-n_add:]

        selection = []
        for select in loss_pred_index:
            selection.append(unlabeled_indices[select])

        labeled_indices = np.concatenate([labeled_indices, selection], axis=0)
        unlabeled_indices = [i for i in unlabeled_indices if i not in selection]

    print("----------------------------------------------")
    print("Learning Loss Algorithm End")

    return labeled_indices, unlabeled_indices
