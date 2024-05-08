# SGBench: A Review and Efficient Implementation of Scene Graph Generation Metrics

Published at [CVPR 2024, Scene Graphs and Graph Representation Learning Workshop](https://sites.google.com/view/sg2rl/index).

For more information, visit our [project page](https://lorjul.github.io/sgbench/).

## Installation

``` bash
# for the full package (includes dependencies to image loading libraries)
pip install sgbench[all]
# for the light package (depends only on NumPy, but supports no IO)
pip install sgbench
```

We strive to make this package compatible with any scene graph generation codebase out there. If you should have any issues or version conflicts when installing SGBench, please file an issue on GitHub and we will look into it.

## Dependencies

- [NumPy](https://numpy.org/) - For faster array operations
- [Pillow](https://pillow.readthedocs.io/en/stable/index.html) - To load ground truth PNG files
- [tifffile](https://github.com/cgohlke/tifffile) - To open TIFF files
- [imagecodecs](https://github.com/cgohlke/imagecodecs) - To support compression of TIFF files

If you choose to install SGBench without `[all]`, only NumPy is required.

## Usage

You can use SGBench from the command line or as a Python package.

### Python

``` python
import sgbench
# for a single image
mapping = sgbench.match_instances(gt_masks, gt_labels, pred_masks, pred_labels)
matched_triplets = sgbench.remap_triplets(mapping, pred_triplets)
mR50 = sgbench.recall(
    k=50,
    mean=True,
    gt_triplets=gt_triplets,
    matched_triplets=matched_triplets,
    num_predicates=56,
)
pr = sgbench.pair_recall(
    k=20,
    gt_triplets=gt_triplets,
    matched_triplets=matched_triplets,
    num_predicates=56,
)
```

### Command Line Interface

The CLI only works if SGBench was installed with `pip install sgbench[all]`.

    sgbench --help

### SGBench File Format

When evaluating using the command line interface, SGBench requires a specific file format. For the ground truth, SGBench uses the same file format as [OpenPSG](https://github.com/Jingkang50/OpenPSG). If you are only using bounding boxes, just omit the PNG files and the segmentation mask related keys.

For the prediction file, SGBench uses its own format as described in the following.

#### SGG With Bounding Boxes

If your model returns bounding boxes, SGBench expects a JSON file in the following format:

``` json
{
    "version": 1,
    "images": [
        {
            // id of the ground truth image (must match image_id in ground truth file)
            "id": "12345",
            // bounding boxes are defined as pixels in xyxy format
            "bboxes": [
                [23.1, 50, 100, 120.0],
                // ...
            ],
            // categories are the class ids of the respective bounding boxes
            "categories": [3, 51, 100, 23],
            // triplets are encoded as subject id, object id, predicate
            // ordering matters! the first triplet is the most confident one
            "triplets": [
                [3, 5, 2],
                [12, 0, 1],
                // ...
            ],
        },
        // ...
    ]
}
```

#### SGG With Segmentation Masks

If your model returns segmentation masks, SGBench expects a ZIP file that contains a JSON file and multiple TIFF files for the segmentation masks.

Each TIFF file stores the segmentation masks for a single image. Use a separate layer for every mask. You can check `sgbench.io.write_layered_seg_mask` for an example of how to write the TIFF files.

The JSON file must be in the following format:

``` json
{
    "version": 1,
    "images": [
        {
            // id of the ground truth image (must match image_id in ground truth file)
            "id": "12345",
            // name of the corresponding TIFF file
            "seg_filename": "seg12345.tiff",
            // categories stores the class ids for each layer in the corresponding TIFF file
            "categories": [3, 51, 100, 23],
            // triplets are encoded as subject id, object id, predicate
            // ordering matters! the first triplet is the most confident one
            "triplets": [
                [3, 5, 2],
                [12, 0, 1],
                // ...
            ],
        },
        // ...
    ]
}
```

## Citation

If you find this work useful, please consider citing our paper:

``` bibtex
@misc{lorenz2024sgbench,
      title={A Review and Efficient Implementation of Scene Graph Generation Metrics},
      author={Julian Lorenz and Robin Schön and Katja Ludwig and Rainer Lienhart},
      year={2024},
      eprint={2404.09616},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
