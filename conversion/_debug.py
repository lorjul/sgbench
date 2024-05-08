from dataclasses import dataclass
from pathlib import Path
import numpy as np


class ResultContainer:
    def __init__(self, indir):
        indir = Path(indir)
        self.paths = sorted(indir.glob("*.npz"), key=lambda p: int(p.stem))

    def __getitem__(self, idx: int):
        result = np.load(self.paths[idx])
        return Result(
            masks=result["masks"],
            refine_bboxes=result["refine_bboxes"],
            labels=result["labels"],
            rel_pair_idxes=result["rel_pair_idxes"],
            rel_labels=result["rel_labels"],
            rel_dists=result["rel_dists"],
        )

    def __len__(self):
        return len(self.paths)


@dataclass
class Result:
    masks: np.ndarray
    refine_bboxes: np.ndarray
    labels: np.ndarray
    rel_pair_idxes: np.ndarray
    rel_labels: np.ndarray
    rel_dists: np.ndarray
