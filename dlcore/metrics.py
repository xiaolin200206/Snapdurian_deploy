"""Detection metrics, written to follow Ultralytics 8.4.182 validation:
- matching at IoU 0.5 as in `BaseValidator.match_predictions` (non-scipy branch): detections in descending
  confidence order, each claims the best still-unclaimed target of its class with IoU >= 0.5,
- AP per class as in `ap_per_class` / `compute_ap`: precision envelope with the 8.4 end sentinels (precision drops
  to 0 at the last recall reached, then stays 0 to recall 1), 101-point interpolation, trapezoidal integration.
Own implementation (not copied); checked against Ultralytics val by notebooks/check_evaluator_colab.ipynb.
Evaluator version 2 (D11): version 1 used the Ultralytics 8.3 forms of both functions (IoU-sorted matching, and a
precision envelope interpolated linearly from the last recall to recall 1), which overstated AP (D10 check, +0.0405).
"""
EVALUATOR_VERSION = 2
import numpy as np

EPS = 1e-16


def box_iou(a, b):
    """a (n,4), b (m,4) xyxy -> (n,m)"""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def match_tp(gt_cls, gt_boxes, pr_cls, pr_boxes, thr=0.5, conf=None):
    """True-positive flag for each prediction (Ultralytics 8.4 match_predictions, non-scipy branch, one threshold).

    Predictions are taken in descending confidence (the NMS output order; if `conf` is given they are ordered by it,
    stable); each claims the best still-unclaimed ground-truth box of the same class if that IoU is >= thr."""
    n = len(pr_cls)
    tp = np.zeros(n, dtype=bool)
    if not len(gt_cls) or not n:
        return tp
    iou = box_iou(gt_boxes, pr_boxes) * (np.asarray(gt_cls)[:, None] == np.asarray(pr_cls)[None, :])
    order = np.arange(n) if conf is None else np.argsort(-np.asarray(conf), kind="stable")
    claimed = np.zeros(len(gt_cls), dtype=bool)
    for j in order:
        if not (iou[:, j] >= thr).any():
            continue
        avail = np.where(claimed, 0.0, iou[:, j])
        k = int(avail.argmax())
        if avail[k] >= thr:
            tp[j] = True
            claimed[k] = True
    return tp


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.interp(x, mrec, mpre), x))


def ap_per_class(tp, conf, pred_cls, target_cls):
    """-> {class: AP50} for every class present in target_cls"""
    order = np.argsort(-conf, kind="stable")
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    out = {}
    classes, counts = np.unique(target_cls, return_counts=True)
    for c, n_l in zip(classes, counts):
        m = pred_cls == c
        if not m.any():
            out[int(c)] = 0.0
            continue
        t = tp[m].astype(float)
        tpc, fpc = t.cumsum(), (1 - t).cumsum()
        recall = tpc / (n_l + EPS)
        precision = tpc / (tpc + fpc)
        out[int(c)] = compute_ap(recall, precision)
    return out


def greedy_pairs(iou, thr=0.5, allowed=None):
    """One-to-one greedy matching by descending IoU. -> list of (i, j)."""
    if not iou.size:
        return []
    ok = iou >= thr if allowed is None else (iou >= thr) & allowed
    cand = np.array(np.nonzero(ok)).T
    if not len(cand):
        return []
    cand = cand[np.argsort(-iou[cand[:, 0], cand[:, 1]], kind="stable")]
    used_i, used_j, pairs = set(), set(), []
    for i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); pairs.append((int(i), int(j)))
    return pairs
