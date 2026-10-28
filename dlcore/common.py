"""Shared constants and helpers: class list, label IO, hashing, sites, versions log."""
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PRELOCK = ROOT / "sequential" / "seq_prelock.json"
VERSIONS_LOG = ROOT / "sequential" / "versions_log.md"
MYT = dt.timezone(dt.timedelta(hours=8))

# Product label space (order = the deployed model's class indices)
PRODUCT = ["algal_spot", "leaf_spot_blight", "leafhopper_damage", "pink_disease", "psyllid", "psyllid_damage",
           "root_disease", "scale_insect", "stem_borer", "web_blight", "weevil_damage"]
IDX = {n: i for i, n in enumerate(PRODUCT)}
CORE = ["algal_spot", "leaf_spot_blight", "leafhopper_damage", "psyllid", "psyllid_damage"]
MERGE = {IDX["scale_insect"]: IDX["psyllid"]}      # applied to GT and predictions before scoring
MIN_GT = 10
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TREE_RE = re.compile(r"^[a-z]+\d*t(?P<tree>\d+)_", re.I)
RF_SUFFIX = re.compile(r"^(?P<stem>.+?)_(?:jpg|jpeg|png|webp)\.rf\.[0-9a-f]+$", re.I)

ALIASES = {
    "algal": "algal_spot", "alga": "algal_spot", "algal_spot": "algal_spot",
    "leaf_spot_blight": "leaf_spot_blight", "leaf_spot": "leaf_spot_blight", "leaf_blight": "leaf_spot_blight",
    "leafhopper_damage": "leafhopper_damage", "leaf_hopper_damage": "leafhopper_damage", "leafhopper": "leafhopper_damage",
    "pink_disease": "pink_disease", "pink_diease": "pink_disease", "pink": "pink_disease",
    "psyllid": "psyllid", "psyllid_damage": "psyllid_damage",
    "root_disease": "root_disease", "root": "root_disease",
    "scale_insect": "scale_insect", "scale": "scale_insect",
    "stem_borer": "stem_borer", "stemborer": "stem_borer",
    "web_blight": "web_blight", "weevil_damage": "weevil_damage",
}


def die(msg):
    sys.exit("✗ " + msg)


def canon(name):
    k = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    if k not in ALIASES:
        die(f"未知的类别名 {name!r}。如果它是 11 类里的某一类的另一种写法，在 dlcore/common.py 的 ALIASES 里加一行。"
            f"（产品 11 类不拆 Phomopsis / Leaf_rot：这一篇用 leaf_spot_blight。）")
    return ALIASES[k]


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(paths, base):
    """Order-independent fingerprint of a set of files: sha256 over sorted 'relpath sha' lines."""
    lines = sorted(f"{Path(p).relative_to(base).as_posix()} {sha256_file(p)}" for p in paths)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), len(lines)


def now_myt():
    return dt.datetime.now(MYT).isoformat(timespec="seconds")


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jd) + "\n", encoding="utf-8")


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def norm_stem(stem):
    """Undo Roboflow's renaming (name_jpg.rf.<hash>)."""
    m = RF_SUFFIX.match(stem)
    return m["stem"] if m else stem


def tree_of(stem):
    m = TREE_RE.match(stem)
    return int(m["tree"]) if m else None


def images_in(d):
    d = Path(d)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


# ---------- YOLO labels ----------
def read_yolo(p):
    """-> array (n, 5): cls, cx, cy, w, h (normalised). Missing file -> None."""
    p = Path(p)
    if not p.exists():
        return None
    rows = []
    for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        t = line.split()
        if not t:
            continue
        if len(t) < 5:
            die(f"{p} 第 {ln} 行格式不对：{line!r}（只支持检测框，不支持分割多边形）")
        if len(t) > 5:
            die(f"{p} 第 {ln} 行有 {len(t)} 个数：像是分割多边形，导出时请选 YOLO 检测框格式")
        c, *xywh = t
        rows.append([float(c), *map(float, xywh)])
    a = np.array(rows, dtype=float).reshape(-1, 5)
    if len(a) and ((a[:, 1:] < -1e-6).any() or (a[:, 1:] > 1 + 1e-6).any()):
        die(f"{p} 里有超出 0–1 的坐标")
    return a


def write_yolo(p, a):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in a), encoding="utf-8")


def xywhn_to_xyxy_px(a, w, h):
    if a is None or not len(a):
        return np.zeros((0, 4))
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def export_names(folder):
    """Class names of an annotation export (data.yaml / classes.txt / obj.names)."""
    folder = Path(folder)
    for y in sorted(folder.rglob("*.yaml")) + sorted(folder.rglob("*.yml")):
        try:
            import yaml
        except ImportError:
            die("缺少 pyyaml：pip install pyyaml")
        d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        if "names" in d:
            n = d["names"]
            return [n[k] for k in sorted(n, key=int)] if isinstance(n, dict) else list(n)
    for name in ["classes.txt", "obj.names"]:
        for t in folder.rglob(name):
            return [x.strip() for x in t.read_text(encoding="utf-8").splitlines() if x.strip()]
    die(f"{folder} 里找不到 data.yaml 或 classes.txt，无法知道类别顺序")


def remap_to_product(a, names):
    if a is None or not len(a):
        return np.zeros((0, 5))
    m = np.array([IDX[canon(n)] for n in names])
    b = a.copy()
    if (b[:, 0] >= len(m)).any():
        die("标注里的类别编号超出了导出的类别表")
    b[:, 0] = m[b[:, 0].astype(int)]
    return b


def label_files(folder):
    """{normalised stem: path} for every .txt under folder (labels/ subfolders included)."""
    out = {}
    for p in Path(folder).rglob("*.txt"):
        if p.name in ("classes.txt", "README.txt", "readme.txt"):
            continue
        s = norm_stem(p.stem)
        if s in out:
            die(f"同一张图有两个标注文件：{out[s]} 和 {p}")
        out[s] = p
    return out


# ---------- sites / versions ----------
def load_site(site):
    p = ROOT / "sites" / site / "site.json"
    if not p.exists():
        die(f"没有这个站：{site}（sites/{site}/site.json 不存在）")
    return json.loads(p.read_text(encoding="utf-8"))


def site_data(site):
    return DATA / "sites" / site


def prelock():
    return json.loads(PRELOCK.read_text(encoding="utf-8"))


def versions_log():
    """{version: {'site':..., 'file':..., 'sha256':...}} from sequential/versions_log.md"""
    out = {}
    for line in VERSIONS_LOG.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.fullmatch(r"v\d+", cells[0]) and re.fullmatch(r"[0-9a-f]{64}", cells[3] or ""):
            out[cells[0]] = {"site": cells[1], "file": cells[2], "sha256": cells[3], "committed": cells[4] if len(cells) > 4 else ""}
    return out
