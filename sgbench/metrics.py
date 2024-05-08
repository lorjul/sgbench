from typing import Union
import numpy as np
from collections import defaultdict

# the functions here are single-image only


def _by_pred_set(triplets: np.ndarray):
    set_dict = defaultdict(set)
    for r in triplets.tolist():
        set_dict[r[2]].add(tuple(r[:2]))
    return set_dict


def recall(
    k: Union[int, float],
    mean: bool,
    gt_triplets: np.ndarray,
    matched_triplets: np.ndarray,
    num_predicates: int,
):
    """
    Calculates recall@k, mean recall@k for a given image.

    :param k:
    :param mean: Wether mean recall or plain recall should be used.
    :param gt_triplets: Ground truth relation triplets.
    :param matched_triplets: Matched output relation triplets, calculated with `remap_triplets()`.
    """

    if mean:
        gt_sets = _by_pred_set(gt_triplets)
        pred_sets = _by_pred_set(matched_triplets[:k])
        rs = np.full((num_predicates,), fill_value=np.nan, dtype=np.float64)
        for p, gt in gt_sets.items():
            rs[p] = len(gt.intersection(pred_sets[p])) / len(gt)
        return rs
    else:
        gt = set([tuple(x) for x in gt_triplets.tolist()])
        pred = set([tuple(x) for x in matched_triplets[:k].tolist()])
        return len(gt.intersection(pred)) / len(gt)


def instance_recall(num_gt_instances: int, inst_pred2gt: np.ndarray):
    """
    Calculates instance recall (InstR) for a given image.

    :param num_gt_instances: Number of ground truth instances in the image.
    :param inst_pred2gt: Instance mapping, generated by `match_instances()`.
    """
    return len(set(inst_pred2gt[inst_pred2gt >= 0])) / num_gt_instances


def predicate_rank(
    gt_triplets: np.ndarray, matched_triplets: np.ndarray, num_predicates: int
):
    """
    Calculates predicate rank (PRank) for a given image.

    :param gt_triplets: Ground truth relation triplets.
    :param matched_triplets: Matched output relation triplets, calculated with `remap_triplets()`.
    """

    # predicate rank
    # convert matched to rank lookup
    # WARN: it makes only sense if there are multiple predictions per sbj-obj pair
    rank_lookup = {}
    pair_counter = defaultdict(lambda: 0)
    for m in matched_triplets:
        if m[0] == -1 or m[1] == -1:
            continue
        m = tuple(m.tolist())
        if m not in rank_lookup:
            rank_lookup[m] = pair_counter[m[:2]]
            pair_counter[m[:2]] += 1

    gt_by_pred = _by_pred_set(gt_triplets)

    ranks = []
    for p in range(num_predicates):
        if len(gt_by_pred[p]) == 0:
            ranks.append(float("nan"))
        else:
            r = []
            for pair in gt_by_pred[p]:
                tpl = (*pair, p)
                if tpl in rank_lookup:
                    r.append(rank_lookup[tpl])
            if r:
                ranks.append(np.mean(r))
            else:
                ranks.append(float("nan"))

    return ranks


def pair_recall(k: int, gt_triplets: np.ndarray, matched_triplets: np.ndarray):
    # ignore predicate
    # triplets are stored as subject, object, predicate
    matched = set([tuple(t) for t in matched_triplets[:k, :2].tolist()])
    gt = set([tuple(t) for t in gt_triplets[:k, :2].tolist()])
    return len(gt.intersection(matched)) / len(gt)


# TODO: class wise outputs