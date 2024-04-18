from argparse import ArgumentParser
import json
from .sg_eval import evaluate


def cli():
    parser = ArgumentParser()
    parser.add_argument("gt", help="Ground truth annotation file")
    parser.add_argument("pred", nargs="+", help="Prediction triplet files")
    parser.add_argument("--masks", default=False, action="store_true")
    parser.add_argument(
        "--gt_seg_dir", default=None, help="Ground truth segmentation masks directory"
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="Number of workers for multiprocessing. Set to 0 to disable multiprocessing",
    )
    parser.add_argument(
        "--chunk", default=8, type=int, help="Chunk size for multiprocessing"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file. If not specified, output will be written to stdout.",
    )
    args = parser.parse_args()

    results = []
    for pred_path in args.pred:
        results.append(
            evaluate(
                gt_path=args.gt,
                pred_path=pred_path,
                use_masks=args.masks,
                gt_seg_dir=args.gt_seg_dir,
                num_workers=args.workers,
                chunksize=args.chunk,
            )
        )

    if args.out is None:
        print()
        max_len = max(len(k) for k in results[0])
        for pred_path, result in zip(args.pred, results):
            print(pred_path)
            for k, v in result.items():
                print(k.ljust(max_len), v, sep=" : ")
    else:
        to_write = []
        for pred_path, result in zip(args.pred, results):
            to_write.append(dict(path=pred_path, **result))
        with open(args.out, "w") as f:
            json.dump(to_write, f, indent=4)


if __name__ == "__main__":
    cli()
