# functions here load files from disk
# in all other modules, data is expected to be in memory
from typing import Union
import multiprocessing as mp
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from collections import defaultdict
import numpy as np
from .pred import parse_pred
from .metrics import instance_recall, predicate_rank, recall
from .io import load_layered_seg_mask, load_rgb_seg_mask, load_sgbench
from .match import match_instances, remap_triplets
from .version import FILE_VERSION


def evaluate(anno_path, pred_path, gt_seg_dir=None, workers: int = 0):
    with TemporaryDirectory(prefix="sgbench-") as tmp_dir:
        is_panoptic, triplets = load_sgbench(pred_path=pred_path, extract_dir=tmp_dir)

        if is_panoptic:
            return _evaluate(
                anno_path=anno_path,
                pred_data=triplets,
                gt_seg_dir=Path(gt_seg_dir),
                pred_seg_dir=Path(tmp_dir),
                workers=workers,
            )
        else:
            return _evaluate(
                anno_path=anno_path,
                pred_data=triplets,
                gt_seg_dir=None,
                pred_seg_dir=None,
                workers=workers,
            )


def _single_evaluate(
    gt_instances: np.ndarray,
    gt_labels: np.ndarray,
    gt_triplets: np.ndarray,
    pred_instances: np.ndarray,
    pred_labels: np.ndarray,
    pred_triplets: np.ndarray,
    pred_triplets_ng: np.ndarray,
    num_predicates: int,
):
    inst_pred2gt = match_instances(
        gt_instances=gt_instances,
        gt_labels=gt_labels,
        pred_instances=pred_instances,
        pred_labels=pred_labels,
    )
    matched_triplets = remap_triplets(
        inst_pred2gt=inst_pred2gt, pred_triplets=pred_triplets
    )
    matched_triplets_ng = remap_triplets(
        inst_pred2gt=inst_pred2gt, pred_triplets=pred_triplets_ng
    )

    metrics = {}
    for k in (20, 50, 100):
        metrics[f"mR@{k}"] = recall(
            k=k,
            mean=True,
            gt_triplets=gt_triplets,
            matched_triplets=matched_triplets,
            num_predicates=num_predicates,
        )
        metrics[f"R@{k}"] = recall(
            k=k,
            mean=False,
            gt_triplets=gt_triplets,
            matched_triplets=matched_triplets,
            num_predicates=num_predicates,
        )

    metrics["InstR"] = instance_recall(
        num_gt_instances=len(gt_instances), inst_pred2gt=inst_pred2gt
    )
    metrics["PRank"] = predicate_rank(
        gt_triplets=gt_triplets,
        matched_triplets=matched_triplets_ng,
        num_predicates=num_predicates,
    )

    return metrics


def _missing_values(gt_triplets: np.ndarray, num_predicates: int):
    metrics = {"InstR": 0.0}

    mR = np.full((num_predicates,), fill_value=np.nan, dtype=np.float64)
    # TODO: no need for for-loop here, there must be some scatter_add or something
    for p in gt_triplets[:, 2]:
        mR[p] += 1

    for k in (20, 50, 100):
        metrics[f"mR@{k}"] = mR
        metrics[f"R@{k}"] = 0.0
    return metrics


def _work(args):
    gt_data, p, gt_seg_dir, pred_seg_dir, num_predicates = args

    # load ground truth segmentation
    gt_triplets = np.array(gt_data["relations"])
    gt_triplets = np.unique(gt_triplets, axis=0)

    if gt_seg_dir is None:
        # use bounding boxes
        gt_labels = [box["category_id"] for box in gt_data["annotations"]]
        gt_instances = np.array([box["bbox"] for box in gt_data["annotations"]])
        pred_instances = p.bboxes
    else:
        # load gt/predicted segmentation mask
        seg_ids = []
        gt_labels = []
        for s in gt_data["segments_info"]:
            seg_ids.append(s["id"])
            gt_labels.append(s["category_id"])
        seg_ids = np.array(seg_ids)
        gt_instances = load_rgb_seg_mask(
            path=gt_seg_dir / gt_data["pan_seg_file_name"], seg_ids=seg_ids
        )
        pred_instances = load_layered_seg_mask(path=pred_seg_dir / p.seg_filename)

    return _single_evaluate(
        gt_instances=gt_instances,
        gt_labels=np.array(gt_labels),
        gt_triplets=gt_triplets,
        pred_instances=pred_instances,
        pred_labels=p.categories,
        pred_triplets=p.triplets,
        pred_triplets_ng=p.ng_triplets,
        num_predicates=num_predicates,
    )


def _evaluate(
    anno_path: Union[str, Path],
    pred_data,
    gt_seg_dir: Union[Path, None],
    pred_seg_dir: Union[Path, None],
    workers: int,
):
    preds = parse_pred(pred_data)

    with open(anno_path) as f:
        anno = json.load(f)

    num_predicates = len(anno["predicate_classes"])

    payload = []
    all_res = []

    for gt_data in anno["data"]:
        if (
            gt_data["image_id"] not in anno["test_image_ids"]
            or len(gt_data["relations"]) == 0
        ):
            continue
        # load all required data

        if gt_data["image_id"] in preds:
            p = preds[gt_data["image_id"]]
            payload.append((gt_data, p, gt_seg_dir, pred_seg_dir, num_predicates))
        else:
            # constant values to use for missing images
            res = _missing_values(
                gt_triplets=np.array(gt_data["relations"]),
                num_predicates=num_predicates,
            )
            all_res.append(res)

    if workers == 0:
        for args in payload:
            all_res.append(_work(args))
    else:
        with mp.Pool(processes=workers) as pool:
            all_res.extend(pool.map(_work, payload))

    single_results = defaultdict(list)
    for res in all_res:
        for key, value in res.items():
            single_results[key].append(value)

    # aggregate single results
    output = {}
    for key, values in single_results.items():
        if key == "PRank":
            prank = np.array(single_results[key])
            output[key] = np.nanmean(prank)
        elif key.startswith("mR@"):
            output[key] = np.nanmean(np.stack(values, axis=0), axis=0).mean()
        else:
            output[key] = np.mean(values)

    return output
