from argparse import ArgumentParser

try:
    from .file import evaluate
except ImportError:
    raise RuntimeError(
        """Could not import required dependencies for CLI usage. Most likely, SGBench was installed in light dependency mode.
If you want to use the CLI, make sure that SGBench is correctly installed: pip install -U 'sgbench[all]'"""
    )


def cli():
    parser = ArgumentParser()
    parser.add_argument("annotation", help="JSON ground truth annotation file")
    parser.add_argument("prediction", help="Prediction file in SGBench format")
    parser.add_argument("--gt-masks", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers to use for calculating the metrics",
    )
    args = parser.parse_args()

    metrics = evaluate(
        anno_path=args.annotation,
        pred_path=args.prediction,
        gt_seg_dir=args.gt_masks,
        workers=args.workers,
    )

    for key, value in metrics.items():
        print(key, value, sep=": ")


if __name__ == "__main__":
    cli()
