# conversion for all two-stage methods from the OpenPSG paper
from argparse import ArgumentParser
import pickle
from pathlib import Path
import json
from zipfile import ZipFile
from tempfile import TemporaryDirectory

from _psg_ids import get_psg_ids
from _debug import ResultContainer
from cvt_psgtr import convert_item as cvt_psgtr
from cvt_pairnet import convert_item as cvt_pairnet
from cvt_openpsg_2stage import convert_item as cvt_2stage


def cli():
    parser = ArgumentParser()
    parser.add_argument("psg")
    parser.add_argument("original")
    parser.add_argument("output")
    parser.add_argument(
        "--method",
        choices=("psgtr", "psgformer", "pairnet", "hilo", "2stage"),
        required=True,
    )
    parser.add_argument("--skip-img", default=False, action="store_true")
    args = parser.parse_args()

    if args.method in ("psgtr", "psgformer", "hilo"):
        convert_item = cvt_psgtr
    elif args.method == "pairnet":
        print(
            "Note: If Pair-Net processing fails, try modifying PYTHONPATH to include the Pair-Net source code folder"
        )
        convert_item = cvt_pairnet
    elif args.method == "2stage":
        convert_item = cvt_2stage
    else:
        raise ValueError()

    img_ids = get_psg_ids(args.psg)

    if Path(args.original).is_dir():
        print("Using numpy workaround")
        original = ResultContainer(args.original)
    else:
        with open(args.original, "rb") as f:
            original = pickle.load(f)

    assert len(original) == len(img_ids), "Annotation file mismatch"

    with TemporaryDirectory(prefix="cvt-sgbench-") as out_dir:
        out_dir = Path(out_dir)
        with ZipFile(args.output, "w") as archive:
            converted = []
            for i, (x, img_id) in enumerate(zip(original, img_ids)):
                converted.append(
                    convert_item(
                        out_dir=out_dir,
                        seg_filename=f"{i}.tiff",
                        orig_item=x,
                        img_id=img_id,
                        skip_img=args.skip_img,
                    )
                )
                archive.write(out_dir / f"{i}.tiff", arcname=f"{i}.tiff")

            with open(out_dir / "triplets.json", "w") as f:
                json.dump(
                    obj={"version": 2, "images": converted}, fp=f, separators=(",", ":")
                )
            archive.write(out_dir / "triplets.json", arcname="triplets.json")


if __name__ == "__main__":
    cli()
