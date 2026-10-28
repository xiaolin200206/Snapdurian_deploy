"""Evaluate one ONNX model on any YOLO folder (images/ + labels/), e.g. merged_v5 val.
Used by notebooks/check_evaluator_colab.ipynb to compare with Ultralytics val."""
from pathlib import Path

import numpy as np

from .common import PRODUCT, images_in, read_yolo, xywhn_to_xyxy_px
from .infer import Detector, load_image
from .metrics import ap_per_class, match_tp


def label_path_for(img, labels_dir=None):
    if labels_dir:
        return Path(labels_dir) / f"{img.stem}.txt"
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def evaluate_folder(model, images_dir, labels_dir=None, merge=None, progress=100):
    det = Detector(model)
    tps, confs, pcs, gcs = [], [], [], []
    imgs = images_in(images_dir)
    for k, p in enumerate(imgs):
        im = load_image(p); w, h = im.size
        g = read_yolo(label_path_for(p, labels_dir)); g = np.zeros((0, 5)) if g is None else g
        gcls = g[:, 0].astype(int); gbox = xywhn_to_xyxy_px(g, w, h)
        b, s, c = det.predict(im, conf=0.001, iou=0.7, max_det=300)
        if merge:
            for a, t in merge.items():
                gcls = np.where(gcls == a, t, gcls); c = np.where(c == a, t, c)
        tps.append(match_tp(gcls, gbox, c, b, 0.5)); confs.append(s); pcs.append(c); gcs.append(gcls)
        if progress and (k + 1) % progress == 0:
            print(f"  {k + 1}/{len(imgs)}")
    ap = ap_per_class(np.concatenate(tps), np.concatenate(confs), np.concatenate(pcs), np.concatenate(gcs))
    return {"n_images": len(imgs), "mAP50": float(np.mean(list(ap.values()))) if ap else float("nan"),
            "AP50": {PRODUCT[c]: v for c, v in sorted(ap.items())}}
