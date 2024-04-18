from collections import defaultdict
import numpy as np


def by_pred_set(triplets: np.ndarray):
    set_dict = defaultdict(set)
    for r in triplets.tolist():
        set_dict[r[2]].add(tuple(r[:2]))
    return set_dict
