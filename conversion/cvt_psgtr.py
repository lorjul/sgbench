# PSGTr and PSGFormer
import torch
import numpy as np
import tifffile


def convert_item(
    out_dir,
    seg_filename,
    orig_item,
    img_id,
    compression=tifffile.COMPRESSION.DEFLATE,
    skip_img=False,
):
    assert (
        orig_item.refine_bboxes is not None
        and orig_item.labels is not None
        and orig_item.rel_dists is not None
        and orig_item.rel_pair_idxes is not None
        and orig_item.rel_labels is not None
    )

    if not skip_img:
        tifffile.imwrite(
            out_dir / seg_filename,
            orig_item.masks.astype(np.uint8),
            compression=compression,
        )
    # has to be retrieved from annotation file
    # orig_item.bboxes appears to be the same as orig_item.refine_bboxes
    # however, PSGTR only has refine_bboxes
    bboxes: np.ndarray = orig_item.refine_bboxes
    box_labels: np.ndarray = orig_item.labels - 1
    assert (box_labels >= 0).all()
    bboxes = bboxes[:, :4].tolist()
    categories = box_labels.tolist()

    # keep the ordering for standard triplets
    triplets = np.concatenate(
        (orig_item.rel_pair_idxes, orig_item.rel_dists[:, 1:].argmax(1)[:, None]),
        axis=1,
    )
    # triplets = np.concatenate(
    #     (orig_item.rel_pair_idxes, orig_item.rel_labels[:, None] - 1), axis=1
    # )
    pred_scores = orig_item.rel_dists[:, 1:]
    num_predicates = pred_scores.shape[1]
    assert (triplets[:, 2] >= 0).all()
    assert (triplets[:, 2] < num_predicates).all()

    tpl_order = pred_scores.flatten().argsort()[::-1]
    pair_sel = tpl_order // num_predicates
    predicates = tpl_order % num_predicates
    ng_triplets = np.concatenate(
        (orig_item.rel_pair_idxes[pair_sel], predicates[:, None]), axis=1
    )

    return {
        "id": img_id,
        "seg_filename": str(seg_filename),
        "bboxes": bboxes,
        "categories": categories,
        "triplets": triplets.tolist(),
        "ng_triplets": ng_triplets.tolist(),
    }
