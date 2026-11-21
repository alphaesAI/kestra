# Nodule Spotter

Nodule Spotter is the detection stage of LungNoduleAgent. It runs independent
detection experts, clusters their proposals by overlap, requires agreement from
multiple distinct experts, and emits the highest-ranked localization mask.

The public release includes a reproducible three-expert MMDetection roster:

| Expert | Architecture | Released artifact |
| --- | --- | --- |
| `retinanet_r18_fpn` | RetinaNet, ResNet-18 + FPN | config + checkpoint |
| `faster_rcnn_r50_fpn` | Faster R-CNN, ResNet-50 + FPN | config + checkpoint |
| `dino_r50_4scale` | DINO, ResNet-50, 4-scale | config + checkpoint |

The files are hosted in the
[LungNoduleAgent DetectionModel collection](https://huggingface.co/YangC777/LungNoduleAgent/tree/main/DetectionModel).
The release configuration pins an immutable repository revision and verifies
the SHA-256 digest of every checkpoint before MMDetection loads it.

## Installation

Create an environment with a PyTorch build appropriate for your CUDA driver,
then install the core and detection dependencies:

```bash
pip install -r requirements.txt
pip install -r nodule_spotter/requirements-detection.txt
```

If a prebuilt MMCV wheel is required for your platform, install the pinned
`mmcv==2.1.0` build with OpenMIM before installing the remaining requirements.
The tested MMDetection stack is MMDetection 3.3.0, MMEngine 0.10.7, and
MMCV 2.1.0.

## Run the released MoE

Input is a directory of axial, grayscale 2D CT slices in PNG, JPEG, or NumPy
format:

```bash
python -m nodule_spotter.main \
  --config nodule_spotter/configs/moe_detection_release.yaml \
  --slices-dir /path/to/grayscale-ct-slices \
  --out-dir outputs/moe-detection
```

The first inference resolves the three configs and checkpoints into the normal
Hugging Face cache. Construction and configuration inspection do not download
or load models. The release defaults to `cuda:0`; edit each detector's `device`
for another accelerator.

Each detector keeps class `0` proposals with score at least `0.3`. DBSCAN uses
IoU distance, and a cluster must contain proposals from at least two distinct
experts. External VLM judges are disabled. The output mask is an ellipse
rasterized inside the selected detection box: it localizes a nodule for the
downstream pipeline, but it is not a pixel-accurate segmentation mask.

## Use another detector

All experts implement the same `BaseDetector` contract and return
`DetectorProposal` objects, so the MoE is backend-independent. Built-in
configuration adapters cover:

- MMDetection with either local config/checkpoint paths or pinned Hugging Face
  files.
- Ultralytics YOLO with a local `weights_path`.
- Custom Python detector classes or registered detector types.

A local MMDetection expert can be configured as:

```yaml
- type: mmdet
  name: local_detector
  config_path: /path/to/model.py
  checkpoint_path: /path/to/model.pth
  class_ids: [0]
```

A YOLO expert can be configured as:

```yaml
- type: yolo
  name: local_yolo
  weights_path: /path/to/yolo_lung.pt
  class_ids: [0]
```

YOLO model loading is supported, but this release does not claim or publish a
lung-nodule YOLO checkpoint. A custom backend only needs to subclass
`BaseDetector`, provide a unique `name`, and implement `detect(slice_image)`.

## Reproducibility and scope

The release config intentionally uses one model at a time
(`max_concurrency: 1`) to control GPU memory and stops on expert failures
(`fail_fast: true`). It returns at most one final mask per slice. These settings
are editable for other environments.

The repository tests cover configuration, lazy artifact resolution, checksums,
proposal conversion, expert failure policy, clustering, and backend adapters
without downloading or loading checkpoints:

```bash
python -m unittest discover -s tests -v
```

The datasets used in the paper are private and are not part of this release.
