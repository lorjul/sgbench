import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import time

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


import multiprocessing as mp
from .pred import parse_pred, PredImg
from .match import load_layered_seg_mask, load_rgb_seg_mask, match_masks, match_boxes
from .utils import by_pred_set


def assign_gt_masks(gt_img: dict, p: PredImg, gt_seg_dir: Path, pred_seg_dir: Path):
    # load ground truth segmentation
    seg_ids = []
    seg_cats = []
    for s in gt_img["segments_info"]:
        seg_ids.append(s["id"])
        seg_cats.append(s["category_id"])
    seg_ids = np.array(seg_ids)
    seg_cats = np.array(seg_cats)
    gt_seg = load_rgb_seg_mask(
        path=gt_seg_dir / gt_img["pan_seg_file_name"], seg_ids=seg_ids
    )

    # load predicted segmentation mask
    pred_seg = load_layered_seg_mask(path=pred_seg_dir / p.seg_filename)

    # match segmentation mask to ground truth
    assigned_gt = match_masks(
        gt_masks=gt_seg,
        pred_masks=pred_seg,
        gt_labels=seg_cats,
        pred_labels=p.categories,
    )

    return assigned_gt


def assign_gt_boxes(img: dict, p: PredImg):
    gt_labels = []
    gt_boxes = []
    for a in img["annotations"]:
        gt_labels.append(a["category_id"])
        gt_boxes.append(a["bbox"])
    gt_labels = np.array(gt_labels)
    gt_boxes = np.array(gt_boxes)

    assigned_gt = match_boxes(
        gt_boxes=gt_boxes,
        pred_boxes=p.bboxes,
        gt_labels=gt_labels,
        pred_labels=p.categories,
    )

    return assigned_gt


def get_gt_pair_masks(gt_dict: dict, gt_seg_dir):
    # load ground truth segmentation
    seg_ids = []
    seg_cats = []
    for s in gt_dict["segments_info"]:
        seg_ids.append(s["id"])
        seg_cats.append(s["category_id"])
    seg_ids = np.array(seg_ids)
    seg_cats = np.array(seg_cats)
    gt_seg = load_rgb_seg_mask(
        path=gt_seg_dir / gt_dict["pan_seg_file_name"], seg_ids=seg_ids
    )

    output = []
    # load all the ground truth masks
    for sbj, obj, rel in gt_dict["relations"]:
        gt_union = gt_seg[sbj] | gt_seg[obj]
        gt_labels = (int(seg_cats[sbj]), int(seg_cats[obj]), int(rel))
        output.append((gt_union, gt_labels))
    return output


def single_evaluate(
    gt_dict: dict,
    gt_triplets: np.ndarray,
    p: PredImg,
    use_masks: bool,
    num_predicates: int,
    gt_seg_dir=None,
    pred_seg_dir=None,
):
    # convert to pairs by predicate
    gt_by_pred = by_pred_set(gt_triplets)

    if use_masks:
        assigned_gt = assign_gt_masks(gt_dict, p, gt_seg_dir, pred_seg_dir)
    else:
        assigned_gt = assign_gt_boxes(gt_dict, p)

    # the following check is only for Predicate Classification
    # assert (assigned_gt == np.arange(len(assigned_gt))).all()

    # replace subject/object in PredImg triplets
    matched = np.stack(
        (
            assigned_gt[p.triplets[:, 0]],
            assigned_gt[p.triplets[:, 1]],
            p.triplets[:, 2],
        ),
        axis=1,
    )
    ng_matched = np.stack(
        (
            assigned_gt[p.ng_triplets[:, 0]],
            assigned_gt[p.ng_triplets[:, 1]],
            p.ng_triplets[:, 2],
        ),
        axis=1,
    )
    # masks that couldn't be matched now have a -1
    # it is not required to remove those triplets because they won't match with any ground truth triplet

    outputs = {}

    # mask recall
    tmp = matched[:, :2].flatten()
    found_gt_masks = set(tmp[tmp != -1].tolist())
    assert len(found_gt_masks) <= len(gt_dict["segments_info"])
    outputs["mask_recall"] = len(found_gt_masks) / len(gt_dict["segments_info"])

    # recall@inf
    found_gt_masks = np.array(list(found_gt_masks))
    inf_hit = []
    for p in range(num_predicates):
        g = np.array(list(gt_by_pred[p]))
        if len(g) == 0:
            inf_hit.append(0)
        else:
            sbj = (g[:, 0][:, None] == found_gt_masks[None]).any(1)
            obj = (g[:, 1][:, None] == found_gt_masks[None]).any(1)
            inf_hit.append(int((sbj & obj).sum()))

    outputs["r_inf"] = inf_hit

    # predicate rank
    # convert matched to rank lookup
    rank_lookup = {}
    pair_counter = defaultdict(lambda: 0)
    for m in ng_matched:
        if m[0] == -1 or m[1] == -1:
            continue
        m = tuple(m.tolist())
        if m not in rank_lookup:
            rank_lookup[m] = pair_counter[m[:2]]
            pair_counter[m[:2]] += 1

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

    outputs["predicate_ranks"] = ranks

    abs_ks = [(f"rabs{k}", k) for k in [20, 50, 100]]
    rel_ks = [(f"rrel{k}", round(k * len(gt_triplets))) for k in [1.0, 2.0, 10.0]]
    # R@k and mR@k
    for name, k in abs_ks + rel_ks:
        # collect data for mean recall@k
        predset = by_pred_set(matched[:k])
        h = [len(gt_by_pred[p].intersection(predset[p])) for p in range(num_predicates)]
        outputs[name] = h

    # ngR@k and mNgR@k
    for name, k in abs_ks + rel_ks:
        predset = by_pred_set(ng_matched[:k])
        h = [len(gt_by_pred[p].intersection(predset[p])) for p in range(num_predicates)]
        outputs[f"ng_{name}"] = h

    # pair recall
    # strip the predicate information
    gt_pairrecall = {tuple(p) for p in gt_triplets[:, :2].tolist()}

    abs_ks = [(f"pr@abs{k}", k) for k in [20, 50, 100]]
    rel_ks = [(f"pr@rel{k}", round(k * len(gt_pairrecall))) for k in [1.0, 2.0, 10.0]]
    ks = abs_ks + rel_ks
    for name, k in ks:
        out_pairrecall = {tuple(p) for p in matched[:k, :2].tolist()}
        outputs[name] = len(gt_pairrecall.intersection(out_pairrecall)) / len(
            gt_pairrecall
        )

    return outputs


def worker_func(args):
    return single_evaluate(**args)


def evaluate(
    gt_path, pred_path, use_masks: bool, gt_seg_dir=None, num_workers=0, chunksize=8
):
    if use_masks:
        assert gt_seg_dir is not None
        gt_seg_dir = Path(gt_seg_dir)
        pred_seg_dir = Path(pred_path).parent
    else:
        gt_seg_dir = None
        pred_seg_dir = None

    # load predictions
    preds = parse_pred(pred_path)

    # load ground truth
    with open(gt_path) as f:
        gt = json.load(f)

    t0 = time.time()

    num_predicates = len(gt["predicate_classes"])

    gt_counts = []

    # only process the test images with relation annotations
    gt_test_data = [
        d
        for d in gt["data"]
        if d["image_id"] in gt["test_image_ids"] and len(d["relations"]) > 0
    ]

    worker_data = []

    for img in gt_test_data:
        img_id = img["image_id"]

        # load ground truth triplets
        gt_triplets = np.array(img["relations"])
        gt_triplets = np.unique(gt_triplets, axis=0)

        gtc = np.zeros((num_predicates,), dtype=int)
        for i in gt_triplets[:, 2]:
            gtc[i] += 1
        gt_counts.append(gtc)

        if img_id in preds:
            worker_data.append(
                dict(
                    gt_dict=img,
                    gt_triplets=gt_triplets,
                    p=preds[img_id],
                    use_masks=use_masks,
                    num_predicates=num_predicates,
                    gt_seg_dir=gt_seg_dir,
                    pred_seg_dir=pred_seg_dir,
                )
            )
        else:
            raise NotImplementedError()

    if num_workers > 0:
        with mp.Pool(processes=num_workers) as pool:
            the_results = list(
                tqdm(
                    pool.imap(worker_func, worker_data, chunksize=chunksize),
                    total=len(worker_data),
                    dynamic_ncols=True,
                    desc=pred_seg_dir.name,
                )
            )
    else:
        the_results = [
            worker_func(d)
            for d in tqdm(worker_data, dynamic_ncols=True, desc=pred_seg_dir.name)
        ]

    gt_counts = np.stack(gt_counts)
    outputs = {}
    for mode, k in [
        ("abs", 20),
        ("abs", 50),
        ("abs", 100),
        ("rel", 1.0),
        ("rel", 2.0),
        ("rel", 10.0),
    ]:
        rh2 = np.array([x[f"r{mode}{k}"] for x in the_results])
        with np.errstate(invalid="ignore"):
            mR50 = np.nanmean(rh2 / gt_counts, axis=0).mean()
        r50 = (rh2.sum(axis=1) / gt_counts.sum(axis=1)).mean()

        outputs[f"mR@{mode}{k}"] = mR50
        outputs[f"R@{mode}{k}"] = r50

        rh2 = np.array([x[f"ng_r{mode}{k}"] for x in the_results])
        with np.errstate(invalid="ignore"):
            mR50 = np.nanmean(rh2 / gt_counts, axis=0).mean()
        r50 = (rh2.sum(axis=1) / gt_counts.sum(axis=1)).mean()

        outputs[f"mNgR@{mode}{k}"] = mR50
        outputs[f"ngR@{mode}{k}"] = r50

        outputs[f"pr@{mode}{k}"] = float(
            np.mean([x[f"pr@{mode}{k}"] for x in the_results])
        )

    outputs["mask_recall"] = float(np.mean([x["mask_recall"] for x in the_results]))
    r_inf = np.array([x["r_inf"] for x in the_results])
    with np.errstate(invalid="ignore"):
        outputs["mR@inf"] = np.nanmean(r_inf / gt_counts, axis=0).mean()
    outputs["R@inf"] = r_inf.sum() / gt_counts.sum()

    predi_ranks = np.array([x["predicate_ranks"] for x in the_results])
    with np.errstate(invalid="ignore"):
        outputs["predicate_ranks"] = np.nanmean(predi_ranks)

    t1 = time.time()
    print("Time taken:", t1 - t0)

    return outputs
