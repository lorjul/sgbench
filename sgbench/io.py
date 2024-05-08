try:
    from PIL import Image
    import tifffile
except ImportError:
    raise RuntimeError(
        "PIL or tifffile dependency is missing. Most likely, SGBench was installed in light dependency mode. For the full install, run: pip install -U 'sgbench[all]'"
    )
import numpy as np
from pathlib import Path
from zipfile import ZipFile
import json
from .version import FILE_VERSION


def rgb2id(color: np.ndarray):
    """Converts a given color to the internal segmentation id
    Adapted from https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L73
    """
    if color.dtype == np.uint8:
        color = color.astype(np.int32)
    return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]


def load_rgb_seg_mask(path, seg_ids: np.ndarray):
    with Image.open(path) as img:
        ids_arr = rgb2id(np.asarray(img))
    # convert to NxHxW shape
    return ids_arr[None] == seg_ids[:, None, None]


def load_layered_seg_mask(path):
    return tifffile.imread(path).astype(bool)


def write_layered_seg_mask(
    seg_mask: np.ndarray, path, compression=tifffile.COMPRESSION.DEFLATE
):
    assert seg_mask.dtype == bool
    tifffile.imwrite(path, seg_mask.astype(np.uint8), compression=compression)


def load_sgbench(pred_path, extract_dir=None):
    # if the prediction file can be opened as a JSON file, it's SGG
    # if the prediction file can be opened as a ZIP file, it's PSGG
    try:
        with open(pred_path) as f:
            triplets = json.load(f)
            return False, triplets
    except UnicodeDecodeError:
        # not a JSON file, move on to ZIP mode
        pass

    assert extract_dir is not None
    extract_dir = Path(extract_dir)
    with ZipFile(pred_path) as archive:
        archive.extract("triplets.json", path=extract_dir)

        img_members = []
        with (extract_dir / "triplets.json").open() as f:
            triplets = json.load(f)

            # check version
            assert triplets.get("version") == FILE_VERSION

            for img in triplets["images"]:
                assert img["seg_filename"].endswith(".tiff")
                img_members.append(img["seg_filename"])

        archive.extractall(members=img_members, path=extract_dir)

        return True, triplets
