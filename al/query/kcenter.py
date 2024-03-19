import numpy as np
import torch
from tqdm import tqdm


def kcenter(data, labeled_indices, unlabeled_indices):
    print("\n\nKCenter Algorithm Start")
    print("----------------------------------------------")
    labeled_dataset = torch.Tensor().cuda()
    unlabeled_dataset = torch.Tensor().cuda()

    selection = list()

    for labeled_index in labeled_indices:
        temp_inp = torch.flatten(torch.from_numpy(data["inp"][labeled_index].astype(np.float32)).clone(), start_dim=1)[:1]
        temp_spec = torch.flatten(torch.from_numpy(data["spec"][labeled_index].astype(np.float32)).clone(), start_dim=1)[:1]
        temp_dataset = torch.cat([temp_inp, temp_spec], dim=1).cuda()

        labeled_dataset = torch.cat([labeled_dataset, temp_dataset], dim=0)  # (unlabeled dataset size, 12*40*40*20+7)

    for unlabeled_index in unlabeled_indices:
        temp_inp = torch.flatten(torch.from_numpy(data["inp"][unlabeled_index].astype(np.float32)).clone(), start_dim=1)[:1]
        temp_spec = torch.flatten(torch.from_numpy(data["spec"][unlabeled_index].astype(np.float32)).clone(), start_dim=1)[:1]
        temp_dataset = torch.cat([temp_inp, temp_spec], dim=1).cuda()

        unlabeled_dataset = torch.cat([unlabeled_dataset, temp_dataset], dim=0)  # (unlabeled dataset size, 12*40*40*20+7)

    temp_dataset = torch.cat([unlabeled_dataset, labeled_dataset], dim=0)  # (unlabeled dataset size + labeled dataset size, 12*40*40*20+7)
    distance_mat = np.array(torch.matmul(temp_dataset, temp_dataset.transpose(0, -1)).cpu())
    diagonal = np.array(distance_mat.diagonal().reshape((1, -1)))

    distance_mat *= -2
    distance_mat += diagonal
    distance_mat += diagonal.transpose()

    temp_mat = distance_mat[: len(unlabeled_indices), len(unlabeled_indices) :]  # (unlabeled dataset size, labeled dataset size)

    n_add = int((len(labeled_indices) + len(unlabeled_indices)) * 0.1)
    for _ in tqdm(range(n_add)):
        temp_selection = np.argmax(temp_mat.min(axis=1))

        temp_mat = np.concatenate([temp_mat, distance_mat[: len(unlabeled_indices), temp_selection].reshape((-1, 1))], axis=1)

        selection.append(temp_selection)

    labeled_indices = np.concatenate([labeled_indices, selection], axis=0)
    unlabeled_indices = [i for i in unlabeled_indices if i not in selection]

    print("----------------------------------------------")
    print("KCenter Algorithm End")

    return labeled_indices, unlabeled_indices
