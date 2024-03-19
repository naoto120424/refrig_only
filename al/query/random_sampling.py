import numpy as np
import random
from tqdm import tqdm


def random_sampling(labeled_indices, unlabeled_indices, n_add):
    print("\n\nRandom Sampling Algorithm Start")
    print("----------------------------------------------")
    selection = random.sample(unlabeled_indices, n_add)
    labeled_indices += selection
    unlabeled_indices = [i for i in tqdm(unlabeled_indices) if i not in selection]

    print("----------------------------------------------")
    print("Random Sampling Algorithm End")

    return labeled_indices, unlabeled_indices
