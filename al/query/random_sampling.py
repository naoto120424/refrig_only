import numpy as np
import random


def random_sampling(labeled_indices, unlabeled_indices, n_add):
    selection = random.sample(unlabeled_indices, n_add)
    labeled_indices += selection
    unlabeled_indices = list(np.delete(unlabeled_indices, selection))

    return labeled_indices, unlabeled_indices
