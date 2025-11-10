# conversion for all two-stage methods from the OpenPSG paper
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
        and orig_item.refine_scores is not None
        and orig_item.rel_pair_idxes is not None
        and orig_item.rels is not None
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
    bboxes: np.ndarray = orig_item.refine_bboxes[:, :4]
    box_labels: np.ndarray = orig_item.labels - 1
    assert (box_labels >= 0).all()
    bboxes = bboxes.tolist()
    categories = box_labels.tolist()

    triplets = orig_item.rels.copy()
    triplets[:, 2] -= 1

    pred_scores = orig_item.rel_dists[:, 1:]
    num_predicates = pred_scores.shape[1]

    node_scores = torch.tensor(orig_item.refine_scores).softmax(1).numpy()[:, 1:].max(1)
    pair_scores = node_scores[orig_item.rel_pair_idxes].prod(1)
    tpl_scores = pred_scores * pair_scores[:, None]
    tpl_order = tpl_scores.flatten().argsort()[::-1]
    pair_sel = tpl_order // num_predicates
    predicates = tpl_order % num_predicates
    ng_triplets = np.concatenate(
        (orig_item.rel_pair_idxes[pair_sel], predicates[:, None]), axis=1
    )

    assert (triplets[:, 2] >= 0).all() and (triplets[:, 2] < num_predicates).all()
    assert (ng_triplets[:, 2] >= 0).all() and (ng_triplets[:, 2] < num_predicates).all()

    return {
        "id": img_id,
        "seg_filename": str(seg_filename),
        "bboxes": bboxes,
        "categories": categories,
        "triplets": triplets.tolist(),
        "ng_triplets": ng_triplets.tolist(),
    }
