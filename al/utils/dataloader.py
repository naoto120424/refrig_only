import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import glob

from tqdm import tqdm
from einops import rearrange


# Data decimate
def decimate(data):
    dt = 1
    new_data = []
    pick_time = dt
    for i in range(len(data)):
        if data[i, 0] > pick_time:
            x = data[i, 0], data[i - 1, 0]
            y = data[i, :], data[i - 1, :]
            a, b = np.polyfit(x, y, 1)  # Find the slope a and intercept b of the linear approximation equation for n seconds and n+0.1 seconds
            new_data.append([a * pick_time + b])  # n seconds of data is obtained from an approximate formula
            pick_time += dt
    new_data = np.array(new_data)
    new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[2]))
    return new_data


# Load All Data
def load_data(cfg, dataset, in_len=20, out_len=20, debug=False):
    print("Load Data")
    print("----------------------------------------------")
    csv_files = glob.glob(os.path.join(cfg.DATA_PATH, "*.csv"))
    csv_files = csv_files[:30] if debug else csv_files
    csv_files.sort()
    del_files = []

    data = {}
    Xdata = []
    Specdata = []
    Ydata = []
    GTData = []

    for file in tqdm(csv_files):
        csv_data = pd.read_csv(os.path.join(cfg.DATA_PATH, file), skiprows=1).values
        single_data = decimate(csv_data)[:, 1:]  # shape(3299, 62)
        # single_data = csv_data[:, 1:]  # No decimate function
        spec_data = single_data[:, : cfg.NUM_CONTROL_FEATURES]
        output_data = single_data[:, cfg.NUM_CONTROL_FEATURES :]
        input_time_list = []
        spec_list = []
        gt_list = []

        if dataset == "refrig_only" and single_data.shape[0] != 3299:
            del_files.append(file)
        else:
            for t in range(single_data.shape[0] - in_len - out_len + 1):
                input_time_list.append(single_data[t : t + in_len])
                spec_list.append(spec_data[t + in_len: t + in_len + out_len])
                gt_list.append(output_data[t + in_len : t + in_len + out_len])
            Xdata.append(np.array(input_time_list))
            Specdata.append(np.array(spec_list))
            Ydata.append(np.array(gt_list))
            GTData.append(single_data[:, -cfg.NUM_PRED_FEATURES:])
            # print(np.array(input_time_list).shape)
            # print(np.array(spec_list).shape)
            # print(np.array(gt_list).shape)
            # print(np.array(t_list).shape)

    data["inp"] = np.array(Xdata)           # shape(case_num, 3299-(in_len+out_len), in_len, num_all_features)
    data["spec"] = np.array(Specdata)       # shape(case_num, 3299-(in_len+out_len), out_len, num_control_features)
    data["gt"] = np.array(Ydata)  # shape(case_num, 3299-(in_len+out_len), out_len, num_pred_features)
    data["gt_data"] = np.array(GTData)      # shape(case_num, 3299, num_all_features)
    # print('data["inp"]', data["inp"].shape)
    # print('data["spec"]', data["spec"].shape)
    # print('data["gt"]', data["gt"].shape)
    # print('data["gt_data"]', data["gt_data"].shape)

    data["feature_name"] = list(map(lambda x: x.split(".")[0], pd.read_csv(os.path.join(cfg.DATA_PATH, csv_files[0]), skiprows=0, dtype=str).columns))
    data["feature_unit"] = list(map(lambda x: x.split(".")[0], pd.read_csv(os.path.join(cfg.DATA_PATH, csv_files[0]), skiprows=1, dtype=str).columns))
    data["feature_name"][7] = "ChillerInT_spec"
    data["feature_name"][38] = "ChillerInT" if dataset == "refrig_only" else data["feature_name"][38]

    data["pred_name"] = data["feature_name"][-cfg.NUM_PRED_FEATURES:]
    data["pred_unit"] = data["feature_unit"][-cfg.NUM_PRED_FEATURES:]
    data["target_name"] = data["feature_name"][-cfg.NUM_TARGET_FEATURES:]
    data["target_unit"] = data["feature_unit"][-cfg.NUM_TARGET_FEATURES:]

    print("----------------------------------------------")

    for file in del_files:
        csv_files.remove(file)

    return data, csv_files


# Calculate mean & std for scaling from train data
def find_meanstd(cfg, dataset, train_index_list, csv_files):
    train_csv_files = []
    for index in train_index_list:
        train_csv_files.append(csv_files[index])
    input_data = []
    for file in train_csv_files:
        single_data = pd.read_csv(os.path.join(cfg.DATA_PATH, file), skiprows=1).values
        if dataset == "refrig_only":
            if single_data.shape[0] == 33000:
                input_data.append(decimate(single_data))
        else:
            input_data.append(decimate(single_data))
    input_array = np.array(input_data)
    mean_list = []
    std_list = []
    for i in range(input_array.shape[2]):
        mean_list.append(input_array[:, :, i].mean())
        std_list.append(input_array[:, :, i].std())
    # print("Mean: ", len(mean_list), mean_list)
    # print("Std : ", len(std_list), std_list)
    return mean_list, std_list


class MazdaDataset(Dataset):
    def __init__(self, data):
        super(MazdaDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data["inp"])

    def __getitem__(self, index):
        return {
            "inp": torch.Tensor(self.data["inp"][index]),
            "spec": torch.Tensor(self.data["spec"][index]),
            "gt": torch.Tensor(self.data["gt"][index]),
        }


# データセットを作成する関数
def create_dataset(cfg, dataset, original_data, index_list, csv_files, is_train, mean_list=[], std_list=[]):
    num_control_features = cfg.NUM_CONTROL_FEATURES
    data = {}
    input_data_list = []
    spec_data_list = []
    gt_data_list = []

    for index in index_list:
        input_data_list.append(original_data["inp"][index])
        spec_data_list.append(original_data["spec"][index])
        gt_data_list.append(original_data["gt"][index])
    input_array = np.array(input_data_list)
    spec_array = np.array(spec_data_list)
    gt_array = np.array(gt_data_list)
    # print(input_array.shape, spec_array.shape, gt_array.shape)

    mean_list, std_list = find_meanstd(cfg, dataset, index_list, csv_files) if is_train else (mean_list, std_list)
    print("\n\nTrain Dataset Normalization") if is_train else print("\n\nValidation Dataset Normalization")
    print("----------------------------------------------")

    # scaling input data
    print("[input]")
    for i in tqdm(range(input_array.shape[3])):
        input_array[:, :, :, i] = (input_array[:, :, :, i] - mean_list[i+1]) / std_list[i+1]

    # scaling spec data
    print("\n[spec]")
    for i in tqdm(range(spec_array.shape[3])):
        spec_array[:, :, :, i] = (spec_array[:, :, :, i] - mean_list[i+1]) / std_list[i+1]

    # scaling ground truth data
    print("\n[ground truth]")
    for i in tqdm(range(gt_array.shape[3])):
        gt_array[:, :, :, i] = (gt_array[:, :, :, i] - mean_list[i+num_control_features+1]) / std_list[i+num_control_features+1]

    data["inp"] = rearrange(input_array, "a b c d -> (a b) c d")
    data["spec"] = rearrange(spec_array, "a b c d -> (a b) c d")
    data["gt"] = rearrange(gt_array, "a b c d -> (a b) c d")

    print("----------------------------------------------")

    return MazdaDataset(data), mean_list, std_list
