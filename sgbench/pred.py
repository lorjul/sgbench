import json
from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PredImg:
    id: str
    triplets: np.ndarray
    ng_triplets: np.ndarray
    # not used at the moment
    seg_ids: Optional[np.ndarray]
    categories: Optional[np.ndarray]
    bboxes: Optional[np.ndarray]
    seg_filename: Optional[str]


def deduplicate(triplets: np.ndarray):
    assert triplets.shape[1] == 3
    # remove duplicate entries
    # np.unique sorts the triplets. We don't want that! Keep the ordering the same
    _, idxes = np.unique(triplets, axis=0, return_index=True)
    return triplets[np.sort(idxes)]


def parse_pred(path) -> Dict[str, PredImg]:
    with open(path) as f:
        pred_data = json.load(f)

    imgs = {}
    for img in pred_data["images"]:
        img_id = img["id"]

        if len(img["triplets"]) == 0:
            triplets = np.empty((0, 3), dtype=int)
        else:
            triplets = deduplicate(np.array(img["triplets"]))
        # TODO: remove these hard-coded assertions
        assert (triplets[:, 2] >= 0).all() and (triplets[:, 2] < 56).all()

        if (
            "ng_triplets" not in img
            or img["ng_triplets"] is None
            or len(img["ng_triplets"]) == 0
        ):
            ng_triplets = np.empty((0, 3), dtype=int)
        else:
            ng_triplets = deduplicate(np.array(img["ng_triplets"]))
        # TODO: remove these hard-coded assertions
        assert (ng_triplets[:, 2] >= 0).all() and (ng_triplets[:, 2] < 56).all()

        seg_ids = []
        cats = []
        bboxes = []

        for s in img["annotation"]:
            if "seg_id" in s:
                seg_ids.append(s["seg_id"])
            cats.append(s["category"])
            bboxes.append(s["bbox"])

        if seg_ids:
            seg_ids = np.array(seg_ids)
        else:
            seg_ids = None
        cats = np.array(cats)
        # TODO: remove these hard-coded assertions
        assert (cats >= 0).all()
        assert (cats < 133).all()
        bboxes = np.array(bboxes)

        imgs[img_id] = PredImg(
            id=img_id,
            triplets=triplets,
            ng_triplets=ng_triplets,
            seg_ids=seg_ids,
            categories=cats,
            bboxes=bboxes,
            seg_filename=img.get("seg_filename"),
        )

    return imgs
