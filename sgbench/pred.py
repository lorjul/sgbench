from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PredImg:
    id: str
    triplets: np.ndarray
    ng_triplets: np.ndarray
    categories: Optional[np.ndarray]
    bboxes: Optional[np.ndarray]
    seg_filename: Optional[str]


def deduplicate(triplets: np.ndarray):
    assert triplets.shape[1] == 3
    # remove duplicate entries
    # np.unique sorts the triplets. We don't want that! Keep the ordering the same
    _, idxes = np.unique(triplets, axis=0, return_index=True)
    return triplets[np.sort(idxes)]


def parse_pred(pred_data) -> Dict[str, PredImg]:
    imgs = {}
    for img in pred_data["images"]:
        img_id = img["id"]

        if len(img["triplets"]) == 0:
            triplets = np.empty((0, 3), dtype=int)
        else:
            triplets = deduplicate(np.array(img["triplets"]))
        assert (triplets[:, 2] >= 0).all()

        if (
            "ng_triplets" not in img
            or img["ng_triplets"] is None
            or len(img["ng_triplets"]) == 0
        ):
            ng_triplets = np.empty((0, 3), dtype=int)
        else:
            ng_triplets = deduplicate(np.array(img["ng_triplets"]))
        assert (ng_triplets[:, 2] >= 0).all()

        cats = []
        bboxes = []

        for s in img["annotation"]:
            cats.append(s["category"])
            bboxes.append(s["bbox"][:4])

        cats = np.array(cats)
        assert (cats >= 0).all()
        bboxes = np.array(bboxes)

        imgs[img_id] = PredImg(
            id=img_id,
            triplets=triplets,
            ng_triplets=ng_triplets,
            categories=cats,
            bboxes=bboxes,
            seg_filename=img.get("seg_filename"),
        )

    return imgs
