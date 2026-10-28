"""ONNX inference for offline scoring.

Pre-processing is identical to the deployed SnapDurian server (server/infer.py, `Detector.letterbox`):
EXIF transpose, RGB, letterbox to 640 x 640 with grey (114) padding, PIL bilinear resize.
Post-processing follows Ultralytics validation: multi-label candidates (class score > conf),
top 30,000 by score, class-aware NMS (IoU 0.7), at most 300 detections.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMGSZ = 640
MAX_WH = 7680
MAX_NMS = 30000


def load_image(p):
    im = Image.open(p)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def letterbox(im):
    w, h = im.size
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (IMGSZ - nw) / 2, (IMGSZ - nh) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), r, left, top


def nms(boxes, scores, iou_thr, max_keep):
    """Greedy NMS on score-sorted boxes; stops once max_keep boxes are kept (same result as full NMS + top-k)."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


class Detector:
    def __init__(self, model_path, threads=0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.path = Path(model_path)
        self.session = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        import ast
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names = [names[i] for i in sorted(names)]

    def predict(self, im, conf=0.001, iou=0.7, max_det=300, multi_label=True):
        """-> boxes (n, 4) xyxy in original pixels, scores (n,), classes (n,)"""
        x, r, left, top = letterbox(im)
        p = self.session.run(None, {self.input_name: x})[0][0].T          # (anchors, 4 + nc)
        cls_scores = p[:, 4:]
        if multi_label:
            i, j = np.nonzero(cls_scores > conf)
            scores, cls, xywh = cls_scores[i, j], j, p[i, :4]
        else:
            j = cls_scores.argmax(1)
            s = cls_scores[np.arange(len(j)), j]
            m = s > conf
            scores, cls, xywh = s[m], j[m], p[m, :4]
        if not len(scores):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        if len(scores) > MAX_NMS:
            top_k = np.argsort(-scores, kind="stable")[:MAX_NMS]
            scores, cls, xywh = scores[top_k], cls[top_k], xywh[top_k]
        boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        keep = nms(boxes + cls[:, None] * MAX_WH, scores, iou, max_det)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        w, h = im.size
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / r, 0, w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / r, 0, h)
        return boxes.astype(np.float64), scores.astype(np.float64), cls.astype(int)
