#!/usr/bin/env python3
"""Sequential deploy-collect-retrain study: one command per step.

  python dl.py status                                   每个站做到哪一步了
  python dl.py rename       --site pahang               PWA 导出 → 按树号改名（p1t3_…），原始预标另存
  python dl.py pick-scratch --site pahang               抽 40 张单数号树的图做“从零标”（H4，和 CEA 共用）
  python dl.py lock-scratch --site pahang               从零标完 → 锁定（必须在 QC 这些图之前）
  python dl.py split        --site pahang               QC 完 → 建双数号树测试集 + 全部树的训练集
  python dl.py lock         --site pahang --qc-hours 8  锁定测试集（打分前，只能一次）
  python dl.py score        --site penang               离线打分（所有已登记的版本 + 可选 --extra）
  python dl.py qc           --site pahang               H2：每张图改了多少框
  python dl.py anchor       --site pahang               H4：预标有没有把人带偏
  python dl.py register     --version v1 --onnx best.onnx        登记新版本（去下一站之前 commit）
  python dl.py build-train  --name v1 --merged DIR --sites pahang   生成 v1 的训练列表和配置
  python dl.py build-swap   --name S1 --merged DIR --manifest CSV --new-site pahang   H3 的等量替换
  python dl.py pack-train   --site pahang               把锁定的训练数据打包，上传 Drive 给训练 notebook
  python dl.py eval-folder  --model m.onnx --images DIR 对任意 YOLO 文件夹打分（和 Ultralytics 对账用）
  python dl.py summary                                  汇总 H1–H4

Paths: images, labels and models live under data/ and models/ (not committed).
"""
import argparse
import csv
import datetime as dt
import io
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dlcore.common import (CORE, DATA, IDX, IMG_EXT, MERGE, MIN_GT, MODELS, MYT, PRELOCK, PRODUCT, RESULTS, ROOT,
                           VERSIONS_LOG, canon, die, export_names, fingerprint, images_in, label_files, load_site,
                           norm_stem, now_myt, prelock, read_yolo, remap_to_product, sha256_file, site_data, tree_of,
                           versions_log, write_json, write_yolo, xywhn_to_xyxy_px)
from dlcore.metrics import ap_per_class, box_iou, greedy_pairs, match_tp

SITES_DIR = ROOT / "sites"


def site_paths(site):
    d = site_data(site)
    s = SITES_DIR / site
    return dict(d=d, s=s, export=d / "export.zip", export_dir=d / "export", renamed=d / "renamed",
                images=d / "renamed" / "images", pre=d / "renamed" / "prelabels", photos=d / "renamed" / "photos.csv",
                unassigned=d / "unassigned", scratch_in=d / "scratch", scratch_upload=d / "scratch_upload",
                scratch_lbl=d / "scratch_product" / "labels", final_in=d / "final", final_lbl=d / "final_product" / "labels",
                test=d / "test_even", train=d / "train_all", preds=d / "preds",
                trees=s / "trees.csv", rename_report=s / "rename_report.txt", subset=s / "scratch_subset.txt",
                scratch_lock=s / "scratch_lock.json", split_report=s / "split_report.txt", lock=s / "lock.json",
                results=RESULTS / site)


def img_size(p):
    from PIL import Image
    with Image.open(p) as im:
        w, h = im.size
        try:
            o = im.getexif().get(0x0112, 1)
        except Exception:
            o = 1
    return (h, w) if o in (5, 6, 7, 8) else (w, h)


def parse_ts(s):
    if s is None or str(s).strip() in ("", "None", "nan"):
        return None
    s = str(s).strip().replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(s)
    return t.replace(tzinfo=MYT) if t.tzinfo is None else t.astimezone(MYT)


def parse_local(s, date):
    s = s.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        if not date:
            die(f"trees.csv 里的时间 {s!r} 没有日期：加 --date YYYY-MM-DD，或者写成 '2026-10-30 09:15'")
        s = f"{date} {s}"
    s = re.sub(r"(?<![\d:])(\d):(\d{2})", r"0\1:\2", s)      # 8:56 -> 08:56
    t = dt.datetime.fromisoformat(s)
    return t.replace(tzinfo=MYT) if t.tzinfo is None else t.astimezone(MYT)


def deployed_model_stem(site_cfg):
    if site_cfg.get("expected_model_stem"):
        return site_cfg["expected_model_stem"]
    v = versions_log().get(site_cfg["deployed_version"])
    if not v:
        return None
    m = re.search(r"server:\s*([^\s)]+)", v["file"])
    return Path(m[1] if m else v["file"].split()[0]).stem


# ============================================================ status
def cmd_status(a):
    vl = versions_log()
    reg = ", ".join(f"{k} ({v['sha256'][:8]}…)" for k, v in vl.items()) or "无"
    print(f"已登记的版本：{reg}")
    for sd in sorted(SITES_DIR.iterdir(), key=lambda p: load_site(p.name)["order"] if (p / 'site.json').exists() else 99):
        if not (sd / "site.json").exists():
            continue
        cfg = load_site(sd.name); P = site_paths(sd.name)
        steps = [("rename", P["rename_report"].exists()), ("pick-scratch", P["subset"].exists()),
                 ("lock-scratch", P["scratch_lock"].exists()), ("split", P["split_report"].exists()),
                 ("lock", P["lock"].exists()), ("score", (P["results"] / "scores.json").exists()),
                 ("qc", (P["results"] / "qc.json").exists()), ("anchor", (P["results"] / "anchor.json").exists())]
        dep = cfg["deployed_version"]
        print(f"\n{cfg['order']}. {sd.name:<8} 部署 {dep}{'（已登记）' if dep in vl else '（⚠ 还没登记，去之前先 register 并 commit）'}")
        print("   " + "  ".join(("✓ " if ok else "· ") + n for n, ok in steps))


# ============================================================ rename
SLACK_S = 120
PAUSE_WEIGHT = 1.0   # a pause outside the noted change must be longer than (weight x) its distance from it


def assign_trees(trees, times):
    """trees: [(tree, start, end)] sorted by tree number; times: {i: datetime}.
    Each boundary between consecutive trees is cut in a pause between photos. The noted change (end of one tree to
    start of the next, to the minute) is trusted first: a pause outside it wins only if it is longer than its distance
    from the noted change, so a real walk between trees can correct a note that is a bit off, but an ordinary gap
    between two photos of the same tree cannot.
    A photo belongs to tree k if it lies between the two cuts and at most SLACK_S outside tree k's noted range."""
    ts = sorted((t, i) for i, t in times.items() if t is not None)
    cuts, notes = [], []
    for (t1, s1, e1), (t2, s2, e2) in zip(trees, trees[1:]):
        wlo, whi = min(e1, s2), max(e1, s2)                     # the noted change
        lo, hi = wlo - dt.timedelta(seconds=60), whi + dt.timedelta(seconds=60)
        pts = [lo] + [t for t, _ in ts if lo < t < hi] + [hi]
        best = None
        for a_, b_ in zip(pts, pts[1:]):
            gap = (b_ - a_).total_seconds()
            dist = max(0.0, (wlo - b_).total_seconds(), (a_ - whi).total_seconds())
            score = gap - PAUSE_WEIGHT * dist
            if best is None or score > best[0]:
                best = (score, a_ + (b_ - a_) / 2, gap)
        _, cut, gap = best
        cuts.append(cut)
        notes.append(f"t{t1}|t{t2} at {cut:%H:%M:%S} (pause {int(gap)} s)")
    lows = [None] + cuts; highs = cuts + [None]
    assign, slack = {}, 0
    for (tn, s, e), lo, hi in zip(trees, lows, highs):
        for t, i in ts:
            if (lo is None or t > lo) and (hi is None or t <= hi) and \
               s - dt.timedelta(seconds=SLACK_S) <= t <= e + dt.timedelta(seconds=SLACK_S):
                assign[i] = tn
                slack += not (s <= t <= e)
    return assign, notes, slack


def tree_contact_sheet(out_rows, P, size=180):
    """One row per tree: its first two and last two photos with their times, so each boundary can be eyeballed."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    def font(n):
        try:
            return ImageFont.load_default(size=n)
        except Exception:
            return ImageFont.load_default()
    big, small = font(30), font(16)
    by = defaultdict(list)
    for r in sorted(out_rows, key=lambda r: r["taken_at_myt"]):
        by[r["tree"]].append(r)
    if not by:
        return None
    trees = sorted(by)
    left, gap = 90, 6
    sheet = Image.new("RGB", (left + 4 * (size + gap) + 24, len(trees) * (size + 28)), "white")
    d = ImageDraw.Draw(sheet)
    for row, tn in enumerate(trees):
        y = row * (size + 28)
        rs = by[tn]
        show = [("first", rs[0]), ("", rs[1] if len(rs) > 1 else None), ("", rs[-2] if len(rs) > 3 else None),
                ("last", rs[-1] if len(rs) > 2 else None)]
        d.text((8, y + size // 2 - 22), f"t{tn}", fill="black", font=big)
        d.text((8, y + size // 2 + 14), f"{len(rs)} photos", fill="grey", font=small)
        for col, (tag, r) in enumerate(show):
            x = left + col * (size + gap) + (24 if col >= 2 else 0)
            if r is None:
                continue
            try:
                im = ImageOps.exif_transpose(Image.open(P["images"] / r["file"])).convert("RGB")
                im.thumbnail((size, size))
                sheet.paste(im, (x, y))
            except Exception:
                pass
            d.text((x, y + size + 4), f"{r['taken_at_myt'][11:19]}  {tag}", fill="black", font=small)
        d.text((left + 2 * (size + gap) + 4, y + size // 2 - 10), "…", fill="black", font=big)
    out = P["d"] / "tree_check.jpg"
    sheet.save(out, quality=85)
    return out


def cmd_rename(a):
    cfg = load_site(a.site); P = site_paths(a.site)
    if P["rename_report"].exists() and not a.force:
        die(f"{a.site} 已经改过名了（{P['rename_report']}）。确定要重做就加 --force。")
    src = Path(a.export) if a.export else (P["export"] if P["export"].exists() else P["export_dir"])
    if not src.exists():
        die(f"找不到 PWA 导出：{src}。在服务器上跑 admin.py export，把 zip 放到 {P['export']}")
    if src.is_file():
        tmp = P["d"] / "_export_unzipped"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    photos_csv = next(src.rglob("photos.csv"), None)
    if not photos_csv:
        die("导出里没有 photos.csv（是不是用 admin.py export 导出的？）")
    names = export_names(src)
    if [canon(n) for n in names] != PRODUCT:
        print(f"（提示）导出的类别顺序和产品 11 类不同，会按名字重新映射：{names}")
    rows = list(csv.DictReader(io.StringIO(photos_csv.read_text(encoding="utf-8-sig"))))
    trees_p = Path(a.trees) if a.trees else P["trees"]
    if not trees_p.exists():
        die(f"没有树的时间表 {trees_p}。照 sites/{a.site}/trees_template.csv 填好（每棵树的开始和结束时间）。")
    trees = []
    for r in csv.DictReader(io.StringIO(trees_p.read_text(encoding="utf-8-sig"))):
        if not (r.get("tree") or "").strip():
            continue
        st, en = r["start"].strip().replace(".", ":"), r["end"].strip().replace(".", ":")
        e = parse_local(en, a.date)
        if en.count(":") == 1:                       # only minutes noted: the tree ran to the end of that minute
            e += dt.timedelta(seconds=59)
        trees.append((int(r["tree"]), parse_local(st, a.date), e))
    if len({t for t, _, _ in trees}) != len(trees):
        die("trees.csv 里有重复的树号")
    trees.sort(key=lambda t: t[0])
    for (t1, s1, e1), (t2, s2, e2) in zip(trees, trees[1:]):
        if s2 < s1 or e1 < s1:
            die(f"树号顺序和时间对不上：树 {t1} {s1:%H:%M}–{e1:%H:%M}，树 {t2} {s2:%H:%M}–{e2:%H:%M}")
        if (e1 - s2).total_seconds() > 120:
            die(f"树 {t1} 和树 {t2} 的时间段重叠超过 2 分钟：{s1:%H:%M:%S}–{e1:%H:%M:%S} / {s2:%H:%M:%S}–{e2:%H:%M:%S}")
    expect = a.expect_model or deployed_model_stem(cfg)
    for k in ("images", "pre"):
        shutil.rmtree(P[k], ignore_errors=True); P[k].mkdir(parents=True)
    shutil.rmtree(P["unassigned"], ignore_errors=True)
    img_dir = next((d for d in src.rglob("images") if d.is_dir()), None)
    if img_dir is None:
        die("导出里没有 images/ 文件夹")
    lbl = label_files(src)
    times = {i: (parse_ts(r.get("taken_at")) or parse_ts(r.get("received_at"))) for i, r in enumerate(rows)}
    assign, boundary_notes, slack_used = assign_trees(trees, times)
    out_rows, per_tree, models, unassigned = [], Counter(), Counter(), []
    for i, r in enumerate(rows):
        stem = Path(r["file"]).stem
        t = times[i]
        tree = assign.get(i)
        imgp = img_dir / r["file"]
        if not imgp.exists():
            die(f"导出里缺图片 {imgp}")
        models[r.get("model", "")] += 1
        if tree is None:
            unassigned.append((r["file"], t.isoformat() if t else "无时间"))
            P["unassigned"].mkdir(parents=True, exist_ok=True); shutil.copy2(imgp, P["unassigned"] / r["file"])
            continue
        new = f"{cfg['prefix']}t{tree}_{stem}"
        shutil.copy2(imgp, P["images"] / f"{new}{imgp.suffix.lower()}")
        pre = remap_to_product(read_yolo(lbl[stem]) if stem in lbl else np.zeros((0, 5)), names)
        write_yolo(P["pre"] / f"{new}.txt", pre)
        per_tree[tree] += 1
        out_rows.append({"file": f"{new}{imgp.suffix.lower()}", "export_file": r["file"], "tree": tree,
                         "set": "odd (CEA)" if tree % 2 else "even (this study)", "taken_at_myt": t.isoformat(),
                         "model": r.get("model", ""), "lat": r.get("lat", ""), "lon": r.get("lon", ""),
                         "n_prelabels": len(pre)})
    with open(P["photos"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]) if out_rows else ["file"]); w.writeheader(); w.writerows(out_rows)
    odd = sum(n for t, n in per_tree.items() if t % 2); even = sum(n for t, n in per_tree.items() if not t % 2)
    lines = [f"site {a.site}  renamed {now_myt()}", f"export: {src.name}  photos in export: {len(rows)}",
             f"assigned to trees: {len(out_rows)}  (odd/CEA {odd}, even/this study {even})  unassigned: {len(unassigned)}",
             "per tree: " + ", ".join(f"t{t}={n}" for t, n in sorted(per_tree.items())),
             f"pre-labelling model(s) in export: {dict(models)}", f"expected deployed model: {expect}"]
    bad_model = {m: n for m, n in models.items() if expect and m != expect}
    if bad_model:
        lines.append(f"WARNING: photos pre-labelled by a different model: {bad_model}")
    lines.append("tree boundaries (cut in the longest pause between photos near each noted change): " + "; ".join(boundary_notes))
    if slack_used:
        lines.append(f"photos up to 2 min outside their tree's noted range, assigned to it: {slack_used}")
    if unassigned:
        lines.append("unassigned (outside every tree time range): " + "; ".join(f"{f} @ {t}" for f, t in unassigned[:50]))
    sheet = tree_contact_sheet(out_rows, P)
    if sheet:
        lines.append(f"check sheet (first two and last two photos of every tree): {sheet.name}")
    P["rename_report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Roboflow-ready copy (YOLO format: images/ + labels/ + data.yaml), for QC with pre-labels
    rf = P["d"] / "roboflow_upload"
    shutil.rmtree(rf, ignore_errors=True)
    (rf / "images").mkdir(parents=True); (rf / "labels").mkdir()
    for p in images_in(P["images"]):
        shutil.copy2(p, rf / "images" / p.name)
        shutil.copy2(P["pre"] / f"{p.stem}.txt", rf / "labels" / f"{p.stem}.txt")
    import yaml
    (rf / "data.yaml").write_text(yaml.safe_dump({"train": "images", "val": "images", "nc": len(PRODUCT), "names": PRODUCT},
                                                 sort_keys=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRoboflow 用的文件夹：{rf}（整个文件夹拖进 Roboflow，选 YOLO 格式）")
    print("⚠ 先跑 pick-scratch：它会把从零标的 40 张从这个文件夹里移出去，之后剩下的才能上传 QC。")
    if bad_model:
        print("\n⚠ 有照片不是由这一站应该用的模型预标的。检查服务器上切换模型的时间（admin.py models），再决定怎么处理。")
    print(f"\n下一步：python dl.py pick-scratch --site {a.site}")


# ============================================================ scratch subset (H4, shared with CEA)
def cmd_pick_scratch(a):
    cfg = load_site(a.site); P = site_paths(a.site)
    if P["subset"].exists():
        die(f"已经抽过了：{P['subset']}（抽样只能做一次）")
    imgs = [p for p in images_in(P["images"]) if (tree_of(p.stem) or 0) % 2 == 1]
    if not imgs:
        die("没有单数号树的图。先跑 rename。")
    rng = random.Random(f"{a.site}-{cfg.get('scratch_seed', 0)}")
    pick = sorted(rng.sample(imgs, min(a.n, len(imgs))), key=lambda p: p.name)
    P["subset"].write_text("".join(p.stem + "\n" for p in pick), encoding="utf-8")
    shutil.rmtree(P["scratch_upload"], ignore_errors=True); P["scratch_upload"].mkdir(parents=True)
    for p in pick:
        shutil.copy2(p, P["scratch_upload"] / p.name)
    # take these images out of the pre-labelled Roboflow folder: they are corrected separately, >= 3 days after lock-scratch
    rf, later = P["d"] / "roboflow_upload", P["d"] / "roboflow_upload_scratch40"
    if rf.exists():
        shutil.rmtree(later, ignore_errors=True)
        (later / "images").mkdir(parents=True); (later / "labels").mkdir()
        for p in pick:
            for sub, name in (("images", p.name), ("labels", f"{p.stem}.txt")):
                if (rf / sub / name).exists():
                    shutil.move(str(rf / sub / name), str(later / sub / name))
        if (rf / "data.yaml").exists():
            shutil.copy2(rf / "data.yaml", later / "data.yaml")
    print(f"抽了 {len(pick)} 张（单数号树，共 {len(imgs)} 张），列表：{P['subset']}")
    print("这份列表现在就 commit。CEA 那边的 40 张从零标用的也是这一批。\n")
    if rf.exists():
        n_rest = len(images_in(rf / "images"))
        print(f"现在可以 QC：{rf}（{n_rest} 张，带预标，这 40 张已经移出去了）")
        print(f"从零标：{P['scratch_upload']}（{len(pick)} 张，不带预标）")
        print(f"这 40 张的 QC：{later}（带预标）。要等从零标完、lock-scratch 之后至少 3 天才能打开。")
    else:
        print(f"从零标：{P['scratch_upload']}（上传到标注工具时不要带预标）")


def _import_labels(folder, wanted, missing_as_empty, what):
    if not folder.exists():
        die(f"找不到{what}导出 {folder}（labels/ + data.yaml 或 classes.txt）")
    names = export_names(folder)
    lf = label_files(folder)
    out, missing = {}, []
    for s in wanted:
        if s in lf:
            out[s] = remap_to_product(read_yolo(lf[s]), names)
        elif missing_as_empty:
            out[s] = np.zeros((0, 5))
        else:
            missing.append(s)
    if missing:
        die(f"{what}导出里缺 {len(missing)} 张图的标注，例如 {missing[:5]}。"
            f"没有病斑的图要导出成空标注；确实是空的就加 --missing-as-empty。")
    extra = sorted(set(lf) - set(wanted))
    if extra:
        print(f"（提示）{what}导出里有 {len(extra)} 个不在本站列表里的标注，已忽略，例如 {extra[:3]}")
    return out


def cmd_lock_scratch(a):
    P = site_paths(a.site)
    if P["scratch_lock"].exists():
        die("从零标已经锁定过了。")
    if not P["subset"].exists():
        die("先跑 pick-scratch。")
    subset = P["subset"].read_text(encoding="utf-8").split()
    labs = _import_labels(Path(a.scratch) if a.scratch else P["scratch_in"], subset, a.missing_as_empty, "从零标")
    shutil.rmtree(P["scratch_lbl"], ignore_errors=True)
    for s, arr in labs.items():
        write_yolo(P["scratch_lbl"] / f"{s}.txt", arr)
    fp, n = fingerprint(sorted(P["scratch_lbl"].glob("*.txt")), P["scratch_lbl"])
    lock = {"stage": "scratch-lock", "site": a.site, "created": now_myt(), "n_images": n,
            "subset_sha256": sha256_file(P["subset"]), "labels_fingerprint": fp,
            "n_boxes": int(sum(len(v) for v in labs.values())),
            "note": "from-scratch annotation with pre-labels hidden; pre-label correction of these images must start >= 3 days after this lock"}
    write_json(P["scratch_lock"], lock)
    print(f"已锁定从零标：{n} 张，{lock['n_boxes']} 个框。指纹 {fp[:16]}…")
    print(f"现在 commit {P['scratch_lock'].relative_to(ROOT)}。这批图的 QC（在预标上改）至少等 3 天以后再做。")


# ============================================================ merge-qc (Roboflow QC export -> final labels)
QC_SUFFIX = re.compile(r"_(missing|modify|modified)$")
LEAF_SPLIT = {"leaf_rot": "leaf_rot", "leafrot": "leaf_rot", "phomopsis": "phomopsis"}   # CEA keeps these apart


def qc_class(name):
    """Roboflow class name -> (product name, CEA-side name, suffix). Product merges leaf_rot / phomopsis into
    leaf_spot_blight; the CEA-side name keeps them apart (the CEA script maps the rest)."""
    from dlcore.common import ALIASES
    k = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    sufs = []
    while (m := QC_SUFFIX.search(k)):                 # also handles stacked suffixes such as algal_missing_modify
        sufs.append("modify" if m.group(1).startswith("modif") else "missing")
        k = k[:m.start()]
    suf = "missing" if "missing" in sufs else ("modify" if sufs else None)   # an added box stays "missing"
    if k in LEAF_SPLIT:
        return "leaf_spot_blight", LEAF_SPLIT[k], suf
    prod = ALIASES.get(k)
    return prod, prod, suf


def cmd_merge_qc(a):
    from dlcore.common import ALIASES  # noqa: F401  (qc_class uses it)
    import yaml
    P = site_paths(a.site)
    if P["lock"].exists() and not a.cea_only:
        die("这一站已经锁定，不能再合并 QC。只想重出 CEA 那份（比如把 leaf_spot_blight 拆成 Phomopsis / leaf_rot 之后），加 --cea-only。")
    if a.cea_only and not (P["final_in"] / "labels").exists():
        die("--cea-only 要在第一次 merge-qc 之后才能用。")
    imgs = images_in(P["images"])
    if not imgs:
        die("先跑 rename。")
    src = Path(a.export)
    if not src.exists():
        die(f"找不到 Roboflow 导出：{src}")
    if src.is_file():
        tmp = P["d"] / "_qc_export_unzipped"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    names = export_names(src)
    if not names:
        die("导出里找不到类别表（data.yaml）。")
    mapped = [qc_class(n) for n in names]
    bad = [n for n, (pr, _, _) in zip(names, mapped) if pr is None]
    if bad:
        die(f"认不出这些类别名：{bad}。如果是 11 类里某一类的另一种写法，在 dlcore/common.py 的 ALIASES 里加一行。")
    lf = label_files(src)
    stems = [p.stem for p in imgs]
    missing = [s_ for s_ in stems if s_ not in lf]
    extra = sorted(set(lf) - set(stems))
    if missing and not a.missing_as_empty:
        die(f"Roboflow 导出里缺 {len(missing)} 张图的标注：\n  " + "\n  ".join(missing[:20]) +
            "\n多半是在 Roboflow 里删掉了，或者没标也没点 Null（没点 Null 的图不会被导出）。"
            "\n确认这几张都是没有病斑的，就加 --missing-as-empty 当作 Null；否则回 Roboflow 补上再导出。")
    cea_names = sorted({c for _, c, _ in mapped})
    prod_of = np.array([IDX[pr] for pr, _, _ in mapped])
    cea_of = np.array([cea_names.index(c) for _, c, _ in mapped])
    fin = P["final_in"]; cea = P["d"] / "cea_export"
    for d in ((cea,) if a.cea_only else (fin, cea)):
        shutil.rmtree(d, ignore_errors=True)
    (cea / "images").mkdir(parents=True); (cea / "labels").mkdir()
    if not a.cea_only:
        (fin / "labels").mkdir(parents=True)
        (fin / "data.yaml").write_text(yaml.safe_dump({"nc": len(PRODUCT), "names": PRODUCT}, sort_keys=False), encoding="utf-8")
    unresolved_ci = {i for i, c in enumerate(cea_of) if cea_names[c] in ("leaf_spot_blight", "web_blight")}
    changed, dropped = [], Counter()
    tot_odd = Counter(); n_odd = 0
    (cea / "data.yaml").write_text(yaml.safe_dump({"nc": len(cea_names), "names": cea_names}, sort_keys=False), encoding="utf-8")
    by_name, by_suffix = Counter(), Counter()
    fin_cls, pre_cls, matched = Counter(), Counter(), Counter()
    tot = Counter(); n_null = 0; odd_unresolved = 0; all_unresolved = 0
    for p in imgs:
        raw = read_yolo(lf[p.stem]) if p.stem in lf else None
        raw = np.zeros((0, 5)) if raw is None else raw
        if len(raw) and (raw[:, 0] >= len(names)).any():
            die(f"{lf[p.stem]} 里的类别编号超出了导出的类别表")
        ci = raw[:, 0].astype(int)
        for c in ci:
            pr, _, suf = mapped[c]
            by_name[names[c]] += 1; by_suffix[(pr, suf or "kept")] += 1
        prod = raw.copy(); cea_l = raw.copy()
        if len(raw):
            prod[:, 0] = prod_of[ci]; cea_l[:, 0] = cea_of[ci]
        t = tree_of(p.stem); odd = bool(t and t % 2 == 1)
        unres = np.array([c in unresolved_ci for c in ci], dtype=bool)
        all_unresolved += int(unres.sum()); odd_unresolved += int(unres.sum()) if odd else 0
        if a.cea_drop_unresolved and unres.any():
            dropped["odd" if odd else "even"] += int(unres.sum())
            cea_l = cea_l[~unres]
        if a.cea_only:
            old = read_yolo(fin / "labels" / f"{p.stem}.txt")
            old = np.zeros((0, 5)) if old is None else old
            if old.shape != prod.shape or (len(old) and not np.allclose(np.round(old, 6), np.round(prod, 6), atol=2e-6)):
                changed.append(p.stem)
        else:
            write_yolo(fin / "labels" / f"{p.stem}.txt", prod)
        write_yolo(cea / "labels" / f"{p.stem}.txt", cea_l)
        shutil.copy2(p, cea / "images" / p.name)
        n_null += not len(raw)
        pre = read_yolo(P["pre"] / f"{p.stem}.txt")
        pre = np.zeros((0, 5)) if pre is None else pre
        w, h = img_size(p)
        mc = match_counts(pre, prod, w, h)
        tot.update(mc)
        if odd:
            tot_odd.update(mc); n_odd += 1
        for c in range(len(PRODUCT)):
            f_c, p_c = prod[prod[:, 0] == c], pre[pre[:, 0] == c]
            fin_cls[c] += len(f_c); pre_cls[c] += len(p_c)
            if len(f_c) and len(p_c):
                iou = box_iou(xywhn_to_xyxy_px(f_c, w, h), xywhn_to_xyxy_px(p_c, w, h))
                matched[c] += len(greedy_pairs(iou, 0.5))
    if changed:
        shutil.rmtree(cea, ignore_errors=True)
        die(f"--cea-only：有 {len(changed)} 张图的框和第一次合并时不一样（这一篇的标注已经锁定，只能改名，不能加、删、移框），"
            f"例如 {changed[:5]}。CEA 那份没有写出。")
    lines = [f"site {a.site}  merge-qc{' --cea-only' if a.cea_only else ''} {now_myt()}", f"export: {Path(a.export).name}",
             f"images: {len(imgs)} renamed, {len(imgs) - len(missing)} with QC labels"
             + (f", {len(missing)} missing in export -> treated as empty: {missing}" if missing else ""),
             f"images with no box (Null): {n_null}"]
    if extra:
        lines.append(f"labels in the export that match no renamed image (ignored): {len(extra)}, e.g. {extra[:3]}")
    lines.append("class names in the export (boxes): " + ", ".join(f"{n}={by_name[n]}" for n in names if by_name[n]))
    lines.append("")
    lines.append("per class (product names; leaf_rot + phomopsis counted as leaf_spot_blight):")
    lines.append(f"  {'class':<20}{'final':>7}{'v0':>7}{'v0 kept':>9}{'v0 found':>10}{'_missing':>10}{'_modify':>9}")
    for c in range(len(PRODUCT)):
        if not (fin_cls[c] or pre_cls[c]):
            continue
        pr = PRODUCT[c]
        found = f"{matched[c] / fin_cls[c]:.0%}" if fin_cls[c] else "-"
        kept = f"{matched[c] / pre_cls[c]:.0%}" if pre_cls[c] else "-"
        lines.append(f"  {pr:<20}{fin_cls[c]:>7}{pre_cls[c]:>7}{kept:>9}{found:>10}"
                     f"{by_suffix[(pr, 'missing')]:>10}{by_suffix[(pr, 'modify')]:>9}")
    lines.append("  final = boxes after QC; v0 = v0 pre-labels (conf >= 0.25); "
                 "v0 kept = share of v0 boxes you kept (same class, IoU >= 0.5); v0 found = share of your final boxes v0 had found")
    lines.append(f"all boxes vs v0 pre-labels: unchanged {tot['unchanged']}, relabelled {tot['relabelled']}, "
                 f"added {tot['added']}, removed {tot['removed']}  (per image: "
                 f"{(tot['added'] + tot['removed'] + tot['relabelled']) / len(imgs):.2f} changes)")
    lines.append("")
    m_ = re.search(r"photos in export: (\d+)", P["rename_report"].read_text(encoding="utf-8")) if P["rename_report"].exists() else None
    lines.append("CEA side (odd trees):")
    lines.append(f"  boxes not split into Phomopsis / leaf_rot (leaf_spot_blight or web_blight): {odd_unresolved} on odd trees "
                 f"({all_unresolved} on all trees)"
                 + (f"; dropped from the CEA copy: {dropped['odd']} odd-tree, {dropped['even']} even-tree boxes" if a.cea_drop_unresolved
                    else ("  <- the CEA prepare step will stop on these" if odd_unresolved else "")))
    lines.append(f"  numbers for the CEA 3_lock.sh: N_PHOTOS={m_.group(1) if m_ else '?'} (all photos in the PWA export), "
                 f"N_ANNOTATED={n_odd} (odd-tree photos), N_ADDED={tot_odd['added']}, N_REMOVED={tot_odd['removed']}, "
                 f"N_RELABELLED={tot_odd['relabelled']}")
    lines.append(f"written: {cea} (split names, for the CEA prepare step)" if a.cea_only else
                 f"written: {fin} (11 product classes, for dl.py split) and {cea} (split names, for the CEA prepare step)")
    P["s"].mkdir(parents=True, exist_ok=True)
    rep_name = "cea_merge_report.txt" if a.cea_only else "qc_merge_report.txt"
    (P["s"] / rep_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if a.cea_only:
        print(f"\n报告存在 sites/{a.site}/{rep_name}。把 {cea} 整个传到 GPU 机器，跑 CEA 的 2_prepare.sh。")
    else:
        print(f"\n报告也存在 sites/{a.site}/{rep_name}。下一步：python dl.py split --site {a.site}"
              + (" --missing-as-empty" if missing else ""))


# ============================================================ split
def cmd_split(a):
    cfg = load_site(a.site); P = site_paths(a.site)
    if P["lock"].exists():
        die("这一站已经锁定，不能再 split。")
    if P["split_report"].exists() and not a.force:
        die("已经 split 过了；QC 有改动要重做就加 --force（锁定之前可以）。")
    imgs = images_in(P["images"])
    if not imgs:
        die("先跑 rename。")
    labs = _import_labels(Path(a.final) if a.final else P["final_in"], [p.stem for p in imgs], a.missing_as_empty, "QC 终标")
    if P["scratch_lock"].exists():
        t0 = dt.datetime.fromisoformat(json.loads(P["scratch_lock"].read_text(encoding="utf-8"))["created"])
        days = (dt.datetime.now(MYT) - t0).total_seconds() / 86400
        if days < 3:
            print(f"⚠ 距离从零标锁定只有 {days:.1f} 天。预锁定要求从零标的那批图，QC 至少在锁定 3 天之后才开始。"
                  "如果那批图确实是 3 天后才 QC 的，就没问题。")
    else:
        print("⚠ 还没有从零标的锁定（lock-scratch）。H4 需要它；没有的话这一站不进 H4。")
    for k in ("test", "train"):
        shutil.rmtree(P[k], ignore_errors=True)
    shutil.rmtree(P["final_lbl"], ignore_errors=True)
    cnt = {"even": Counter(), "all": Counter()}; n_even = 0; trees_even = set()
    for p in imgs:
        arr = labs[p.stem]; t = tree_of(p.stem)
        write_yolo(P["final_lbl"] / f"{p.stem}.txt", arr)
        for k in ("train",):
            (P[k] / "images").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, P[k] / "images" / p.name); write_yolo(P[k] / "labels" / f"{p.stem}.txt", arr)
        cnt["all"].update(PRODUCT[int(c)] for c in arr[:, 0])
        if t % 2 == 0:
            (P["test"] / "images").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, P["test"] / "images" / p.name); write_yolo(P["test"] / "labels" / f"{p.stem}.txt", arr)
            cnt["even"].update(PRODUCT[int(c)] for c in arr[:, 0]); n_even += 1; trees_even.add(t)
    merged = Counter()
    for k, v in cnt["even"].items():
        merged["psyllid" if k == "scale_insect" else k] += v
    core_used = [c for c in CORE if merged[c] >= MIN_GT]
    lines = [f"site {a.site}  split {now_myt()}",
             f"images: all trees {len(imgs)}, even-tree test set {n_even} ({len(trees_even)} trees: {sorted(trees_even)})",
             "boxes, even-tree test set: " + ", ".join(f"{c}={cnt['even'][c]}" for c in PRODUCT if cnt['even'][c]),
             "boxes, all trees (training data for the next version): " + ", ".join(f"{c}={cnt['all'][c]}" for c in PRODUCT if cnt['all'][c]),
             f"core classes with >= {MIN_GT} boxes on the test set (scale_insect merged into psyllid): {core_used}",
             f"core classes excluded at this site: {[c for c in CORE if c not in core_used]}"]
    P["split_report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n确认无误后：python dl.py lock --site {a.site} --qc-hours <QC 总小时数>")


# ============================================================ lock
def cmd_lock(a):
    cfg = load_site(a.site); P = site_paths(a.site)
    if P["lock"].exists():
        die("已经锁定过了，锁定只能一次。")
    if not P["split_report"].exists():
        die("先跑 split。")
    vl = versions_log(); dep = cfg["deployed_version"]
    if dep not in vl:
        die(f"versions_log.md 里还没有 {dep} 的哈希。这一站的预标模型必须先登记。")
    pre = prelock()
    if dep == "v0" and vl["v0"]["sha256"] != pre["v0"]["onnx"]["sha256"]:
        die("versions_log 里 v0 的哈希和预锁定不一致！")
    tf, tn = fingerprint(sorted((P["test"] / "images").iterdir()) + sorted((P["test"] / "labels").iterdir()), P["test"])
    ff, fn = fingerprint(sorted(P["final_lbl"].glob("*.txt")), P["final_lbl"])
    pf, pn = fingerprint(sorted(P["pre"].glob("*.txt")), P["pre"])
    imf, imn = fingerprint(images_in(P["images"]), P["images"])
    lock = {"stage": "lock", "site": a.site, "order": cfg["order"], "created": now_myt(),
            "prelock_sha256": sha256_file(PRELOCK), "deployed_version": dep, "deployed_sha256": vl[dep]["sha256"],
            "versions_log_sha256": sha256_file(VERSIONS_LOG),
            "test_set": {"fingerprint": tf, "n_files": tn, "n_images": len(images_in(P["test"] / "images"))},
            "all_images": {"fingerprint": imf, "n": imn},
            "final_labels_all": {"fingerprint": ff, "n": fn},
            "prelabels_all": {"fingerprint": pf, "n": pn},
            "scratch_lock_sha256": sha256_file(P["scratch_lock"]) if P["scratch_lock"].exists() else None,
            "split_report": P["split_report"].read_text(encoding="utf-8"),
            "qc_hours": a.qc_hours, "visit_date": a.visit_date, "district": a.district, "note": a.note or ""}
    write_json(P["lock"], lock)
    print(f"已锁定 {a.site}：测试集 {lock['test_set']['n_images']} 张，指纹 {tf[:16]}…")
    print(f"锁定文件 SHA-256：{sha256_file(P['lock'])}")
    print(f"👉 commit {P['lock'].relative_to(ROOT)} 并把这个哈希发邮件给自己，然后再打分。")


def check_lock(a):
    P = site_paths(a.site)
    if not P["lock"].exists():
        die("还没锁定：先跑 lock。")
    lock = json.loads(P["lock"].read_text(encoding="utf-8"))
    tf, _ = fingerprint(sorted((P["test"] / "images").iterdir()) + sorted((P["test"] / "labels").iterdir()), P["test"])
    if tf != lock["test_set"]["fingerprint"]:
        die("⚠ 测试集和锁定时不一样（图片或标注被改过）。停止。")
    return P, lock


# ============================================================ score
def run_model(path, P, label):
    from dlcore.infer import Detector, load_image
    sha = sha256_file(path)
    cache = P["preds"] / f"{label}_{sha[:12]}.npz"
    imgs = images_in(P["test"] / "images")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["stems"]) == [p.stem for p in imgs]:
            return sha, z
    det = Detector(path)
    if [canon(n) for n in det.names] != PRODUCT:
        die(f"{path} 的类别不是产品 11 类：{det.names}")
    idx, boxes, scores, cls, sizes = [], [], [], [], []
    for i, p in enumerate(imgs):
        im = load_image(p)
        b, s, c = det.predict(im, conf=0.001, iou=0.7, max_det=300)
        idx.append(np.full(len(s), i)); boxes.append(b); scores.append(s); cls.append(c); sizes.append(im.size)
        if (i + 1) % 50 == 0:
            print(f"    {label}: {i + 1}/{len(imgs)}")
    P["preds"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, stems=np.array([p.stem for p in imgs]), idx=np.concatenate(idx) if idx else np.zeros(0, int),
                        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4)), scores=np.concatenate(scores) if scores else np.zeros(0),
                        cls=np.concatenate(cls) if cls else np.zeros(0, int), sizes=np.array(sizes))
    return sha, np.load(cache, allow_pickle=False)


def merge_cls(c):
    c = np.asarray(c, dtype=int).copy()
    for k, v in MERGE.items():
        c[c == k] = v
    return c


def per_image_records(P, z):
    """For one model: per image (tp, conf, pred_cls, gt_cls) after the scale_insect -> psyllid merge."""
    imgs = images_in(P["test"] / "images")
    recs = []
    for i, p in enumerate(imgs):
        w, h = z["sizes"][i]
        g = read_yolo(P["test"] / "labels" / f"{p.stem}.txt")
        g = np.zeros((0, 5)) if g is None else g
        gcls = merge_cls(g[:, 0]); gbox = xywhn_to_xyxy_px(g, w, h)
        m = z["idx"] == i
        pcls = merge_cls(z["cls"][m]); pbox = z["boxes"][m]; conf = z["scores"][m]
        tp = match_tp(gcls, gbox, pcls, pbox, 0.5)
        recs.append({"stem": p.stem, "tree": tree_of(p.stem), "tp": tp, "conf": conf, "pcls": pcls, "gcls": gcls})
    return recs


def maps_from(recs, core_idx):
    if not recs:
        return float("nan"), {}
    tp = np.concatenate([r["tp"] for r in recs]); conf = np.concatenate([r["conf"] for r in recs])
    pc = np.concatenate([r["pcls"] for r in recs]); gc = np.concatenate([r["gcls"] for r in recs])
    ap = ap_per_class(tp, conf, pc, gc)
    core = [ap[c] for c in core_idx if c in ap]
    return (float(np.mean(core)) if core else float("nan")), ap


def cmd_score(a):
    cfg = load_site(a.site)
    P, lock = check_lock(a)
    pre = prelock(); vl = versions_log()
    models = []
    for v in cfg["versions_to_score"]:
        p = MODELS / f"{v}.onnx"
        if not p.exists():
            die(f"缺模型 {p}（用 register 登记后会自动放到 models/）")
        sha = sha256_file(p)
        if v not in vl or vl[v]["sha256"] != sha:
            die(f"{p} 的哈希和 versions_log.md 里登记的 {v} 不一致。")
        if v == "v0" and sha != pre["v0"]["onnx"]["sha256"]:
            die("models/v0.onnx 不是预锁定里的 v0！")
        models.append((v, p, "registered"))
    for e in a.extra or []:
        if "=" not in e:
            die("--extra 的格式是 NAME=PATH，例如 S1_s0=runs/S1_s0/weights/best.onnx")
        n, p = e.split("=", 1)
        if not Path(p).exists():
            die(f"找不到 {p}")
        models.append((n, Path(p), "extra"))
    recs = {}
    out = {"site": a.site, "created": now_myt(), "lock_sha256": sha256_file(P["lock"]),
           "evaluation": pre["evaluation"], "models": {}}
    gt_all = None
    for name, path, kind in models:
        print(f"  推理 {name} …")
        sha, z = run_model(path, P, name)
        r = per_image_records(P, z); recs[name] = r
        if gt_all is None:
            gt_all = Counter(int(c) for rr in r for c in rr["gcls"])
        out["models"][name] = {"kind": kind, "sha256": sha}
    core_idx = [IDX[c] for c in CORE if gt_all.get(IDX[c], 0) >= MIN_GT]
    out["core_classes_used"] = [PRODUCT[c] for c in core_idx]
    out["gt_boxes_merged"] = {PRODUCT[c]: n for c, n in sorted(gt_all.items())}
    out["n_images"] = len(images_in(P["test"] / "images"))
    trees = sorted({rr["tree"] for rr in next(iter(recs.values()))})
    out["trees"] = trees
    for name in recs:
        m_core, ap = maps_from(recs[name], core_idx)
        out["models"][name].update({"mAP50_core": m_core,
                                    "mAP50_all": float(np.mean(list(ap.values()))) if ap else float("nan"),
                                    "AP50": {PRODUCT[c]: v for c, v in sorted(ap.items())}})
    # tree-level bootstrap
    rng = np.random.default_rng(0); B = a.bootstrap
    by_tree = {name: defaultdict(list) for name in recs}
    for name, rr in recs.items():
        for r in rr:
            by_tree[name][r["tree"]].append(r)
    boot = {name: np.empty(B) for name in recs}
    for b in range(B):
        samp = rng.choice(trees, size=len(trees), replace=True)
        for name in recs:
            boot[name][b] = maps_from([r for t in samp for r in by_tree[name][t]], core_idx)[0]
        if (b + 1) % 500 == 0:
            print(f"    bootstrap {b + 1}/{B}")
    for name in recs:
        out["models"][name]["mAP50_core_ci95"] = [float(np.nanpercentile(boot[name], 2.5)), float(np.nanpercentile(boot[name], 97.5))]
    dep = cfg.get("h1_newer", cfg["deployed_version"]); prev = cfg.get("h1_older", cfg.get("previous_version"))
    if prev and dep in recs and prev in recs:
        d = boot[dep] - boot[prev]
        out["H1_pair"] = {"deployed": dep, "previous": prev,
                          "diff": out["models"][dep]["mAP50_core"] - out["models"][prev]["mAP50_core"],
                          "diff_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
    P["results"].mkdir(parents=True, exist_ok=True)
    write_json(P["results"] / "scores.json", out)
    write_json(P["results"] / "bootstrap.json", {"seed": 0, "B": B, "trees": trees, "mAP50_core": {k: v.tolist() for k, v in boot.items()}})
    lines = [f"Site {a.site} (order {cfg['order']}), {out['n_images']} test images, trees {trees}",
             f"core classes used: {out['core_classes_used']}", ""]
    for name, m in out["models"].items():
        lines.append(f"{name:<10} mAP50-core {m['mAP50_core']:.4f}  [{m['mAP50_core_ci95'][0]:.4f}, {m['mAP50_core_ci95'][1]:.4f}]"
                     f"   mAP50-all {m['mAP50_all']:.4f}   ({m['kind']}, {m['sha256'][:12]}…)")
    if "H1_pair" in out:
        h = out["H1_pair"]
        lines += ["", f"H1 pair: {h['deployed']} - {h['previous']} = {h['diff']:+.4f}  95% CI [{h['diff_ci95'][0]:+.4f}, {h['diff_ci95'][1]:+.4f}]"]
    (P["results"] / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


# ============================================================ H2 / H4
def match_counts(pre, fin, w, h, keep=None):
    if keep is not None:
        pre = pre[np.isin(pre[:, 0].astype(int), keep)]; fin = fin[np.isin(fin[:, 0].astype(int), keep)]
    pb, fb = xywhn_to_xyxy_px(pre, w, h), xywhn_to_xyxy_px(fin, w, h)
    iou = box_iou(fb, pb)
    same = (fin[:, 0][:, None] == pre[:, 0][None, :]) if len(fin) and len(pre) else np.zeros(iou.shape, bool)
    p1 = greedy_pairs(iou, 0.5, same)
    used_f, used_p = {i for i, _ in p1}, {j for _, j in p1}
    rest_f = [i for i in range(len(fin)) if i not in used_f]; rest_p = [j for j in range(len(pre)) if j not in used_p]
    p2 = greedy_pairs(iou[np.ix_(rest_f, rest_p)], 0.5) if rest_f and rest_p else []
    relab = len(p2)
    return {"unchanged": len(p1), "relabelled": relab, "added": len(fin) - len(p1) - relab, "removed": len(pre) - len(p1) - relab}


def cmd_qc(a):
    P, lock = check_lock(a)
    ff, _ = fingerprint(sorted(P["final_lbl"].glob("*.txt")), P["final_lbl"])
    pf, _ = fingerprint(sorted(P["pre"].glob("*.txt")), P["pre"])
    if ff != lock["final_labels_all"]["fingerprint"] or pf != lock["prelabels_all"]["fingerprint"]:
        die("⚠ 终标或预标和锁定时不一样。停止。")
    core_keep = [IDX[c] for c in CORE] + [IDX["scale_insect"]]
    tot, tot_core, n = Counter(), Counter(), 0
    for p in images_in(P["images"]):
        w, h = img_size(p)
        pre = read_yolo(P["pre"] / f"{p.stem}.txt"); fin = read_yolo(P["final_lbl"] / f"{p.stem}.txt")
        pre = np.zeros((0, 5)) if pre is None else pre; fin = np.zeros((0, 5)) if fin is None else fin
        tot.update(match_counts(pre, fin, w, h)); tot_core.update(match_counts(pre, fin, w, h, core_keep)); n += 1
    rate = lambda c: (c["added"] + c["removed"] + c["relabelled"]) / max(n, 1)
    out = {"site": a.site, "created": now_myt(), "n_images": n, "all_classes": dict(tot), "rate_per_image": rate(tot),
           "core_classes": dict(tot_core), "rate_per_image_core": rate(tot_core), "qc_hours": lock.get("qc_hours"),
           "prelabel_threshold": "as exported by the PWA (conf >= 0.25)"}
    write_json(P["results"] / "qc.json", out)
    print(f"{a.site}: {n} 张图；加 {tot['added']}、删 {tot['removed']}、改类 {tot['relabelled']}、没动 {tot['unchanged']}；"
          f"每张图平均改 {out['rate_per_image']:.2f} 个框（核心类 {out['rate_per_image_core']:.2f}）")


def share_matching_pre(ann, pre, w, h):
    iou = box_iou(xywhn_to_xyxy_px(ann, w, h), xywhn_to_xyxy_px(pre, w, h))
    return len(greedy_pairs(iou, 0.5)), len(ann)


def cmd_anchor(a):
    P, lock = check_lock(a)
    if not P["scratch_lock"].exists():
        die("这一站没有从零标的锁定，不进 H4。")
    sl = json.loads(P["scratch_lock"].read_text(encoding="utf-8"))
    sf, _ = fingerprint(sorted(P["scratch_lbl"].glob("*.txt")), P["scratch_lbl"])
    if sf != sl["labels_fingerprint"] or sha256_file(P["scratch_lock"]) != lock["scratch_lock_sha256"]:
        die("⚠ 从零标和锁定时不一样。停止。")
    subset = P["subset"].read_text(encoding="utf-8").split()
    mc = nc = ms = ns = 0
    for s in subset:
        p = next(q for q in images_in(P["images"]) if q.stem == s)
        w, h = img_size(p)
        pre = read_yolo(P["pre"] / f"{s}.txt"); fin = read_yolo(P["final_lbl"] / f"{s}.txt"); scr = read_yolo(P["scratch_lbl"] / f"{s}.txt")
        pre, fin, scr = [np.zeros((0, 5)) if x is None else x for x in (pre, fin, scr)]
        m, k = share_matching_pre(fin, pre, w, h); mc += m; nc += k
        m, k = share_matching_pre(scr, pre, w, h); ms += m; ns += k
    out = {"site": a.site, "created": now_myt(), "n_images": len(subset),
           "corrected": {"matched_to_prelabel": mc, "boxes": nc, "share": mc / nc if nc else None},
           "scratch": {"matched_to_prelabel": ms, "boxes": ns, "share": ms / ns if ns else None}}
    write_json(P["results"] / "anchor.json", out)
    print(f"{a.site}: 预标后修改 {mc}/{nc} 个框和预标重合；从零标 {ms}/{ns}。")


# ============================================================ versions / training lists
# ============================================================ preview (before lock; descriptive, not the locked result)
def cmd_preview(a):
    cfg = load_site(a.site); P = site_paths(a.site)
    imgs = images_in(P["images"])
    if not imgs:
        die("先跑 rename。")
    even = [p for p in imgs if (tree_of(p.stem) or 1) % 2 == 0]
    labs = _import_labels(Path(a.final) if a.final else P["final_in"], [p.stem for p in even], a.missing_as_empty,
                          "QC 终标（双数号树）")
    v = cfg["deployed_version"]; mp = MODELS / f"{v}.onnx"
    if not mp.exists():
        die(f"缺模型 {mp}。v0 的话跑：python dl.py register --version v0 --onnx <freeze_v0 里的 durian_11cls_640_v5.onnx>")
    if v in versions_log() and versions_log()[v]["sha256"] != sha256_file(mp):
        die(f"{mp} 的哈希和 versions_log.md 里登记的 {v} 不一致。")
    Q = dict(P); Q["test"] = P["d"] / "preview_test"; Q["preds"] = P["d"] / "preview_preds"
    shutil.rmtree(Q["test"], ignore_errors=True)
    (Q["test"] / "images").mkdir(parents=True); (Q["test"] / "labels").mkdir()
    for p in even:
        shutil.copy2(p, Q["test"] / "images" / p.name); write_yolo(Q["test"] / "labels" / f"{p.stem}.txt", labs[p.stem])
    print(f"  {v} 推理 {len(even)} 张双数号树的图 …")
    _, z = run_model(mp, Q, v)
    recs = per_image_records(Q, z)
    gt = Counter(int(c) for r in recs for c in r["gcls"])
    core_idx = [IDX[c] for c in CORE if gt.get(IDX[c], 0) >= MIN_GT]
    m, ap = maps_from(recs, core_idx)
    trees = sorted({r["tree"] for r in recs}); by = defaultdict(list)
    for r in recs:
        by[r["tree"]].append(r)
    rng = np.random.default_rng(0)
    boot = [maps_from([r for t in rng.choice(trees, size=len(trees), replace=True) for r in by[t]], core_idx)[0]
            for _ in range(a.bootstrap)]
    ci = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    out = {"site": a.site, "created": now_myt(), "model": v, "n_images": len(even), "trees": trees,
           "core_classes_used": [PRODUCT[c] for c in core_idx], "mAP50_core": m, "mAP50_core_ci95": ci,
           "AP50": {PRODUCT[c]: float(x) for c, x in sorted(ap.items())},
           "gt_boxes": {PRODUCT[c]: n for c, n in sorted(gt.items())},
           "note": "preview before lock; the locked score (dl.py score) is the one reported"}
    write_json(P["d"] / "preview.json", out)
    print(f"\n预览（还没锁定，只给你自己看）：{a.site}，{v}，双数号树 {len(trees)} 棵 {len(even)} 张")
    print(f"mAP50-core = {m:.3f}  (95% CI {ci[0]:.3f}–{ci[1]:.3f}，按树 bootstrap)")
    print("核心类：" + ", ".join(f"{PRODUCT[c]} {ap[c]:.3f} ({gt[c]} 框)" for c in core_idx if c in ap))
    rest = [c for c in sorted(ap) if c not in core_idx]
    if rest:
        print("其他类：" + ", ".join(f"{PRODUCT[c]} {ap[c]:.3f} ({gt.get(c, 0)} 框)" for c in rest))
    print("锁定之后跑 score 得到的才是正式数字；只要双数号树的标注不再改，两个数会一样。")


def cmd_register(a):
    vl = versions_log()
    p = Path(a.onnx)
    if not p.exists():
        die(f"找不到 {p}")
    sha = sha256_file(p)
    from dlcore.infer import Detector
    names = Detector(p).names
    if [canon(n) for n in names] != PRODUCT:
        die(f"类别不是产品 11 类：{names}")
    if a.version in vl and vl[a.version]["sha256"] != sha and not a.force:
        die(f"{a.version} 已经登记过另一个哈希 {vl[a.version]['sha256'][:12]}…")
    if any(v["sha256"] == sha for k, v in vl.items() if k != a.version):
        die("这个文件和另一个已登记的版本完全一样。")
    MODELS.mkdir(exist_ok=True)
    shutil.copy2(p, MODELS / f"{a.version}.onnx")
    if a.version in vl and vl[a.version]["sha256"] == sha:
        print(f"{a.version} 早就登记过、哈希一致，只把文件放到 models/{a.version}.onnx，登记表不动。")
        return
    text = VERSIONS_LOG.read_text(encoding="utf-8").splitlines()
    fname = f"{p.name}" + (f" (server: {a.server_file})" if a.server_file else "")
    done = False
    for i, line in enumerate(text):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == a.version:
            text[i] = f"| {a.version} | {cells[1]} | {fname} | {sha} | {now_myt()} |"; done = True
    if not done:
        die(f"versions_log.md 里没有 {a.version} 这一行")
    VERSIONS_LOG.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"已登记 {a.version}：{sha}")
    print("👉 现在就 commit + push sequential/versions_log.md，再去这个版本部署的那一站。服务器上 admin.py use 这个文件。")


def merged_train_images(merged):
    d = Path(merged) / "images" / "train"
    imgs = images_in(d)
    if not imgs:
        die(f"{d} 里没有图片（--merged 指向解压后的 merged_v5 文件夹）")
    return imgs


def site_train_images(site):
    """All-tree training images of a locked site, checked against the lock fingerprints."""
    P = site_paths(site)
    if not P["lock"].exists():
        die(f"{site} 还没锁定，不能拿来训练。")
    lock = json.loads(P["lock"].read_text(encoding="utf-8"))
    imgs = images_in(P["train"] / "images")
    if not imgs:
        die(f"{P['train'] / 'images'} 里没有图片（在 Colab 上要先解压 {site}_train_all.zip）")
    fi, _ = fingerprint(imgs, P["train"] / "images")
    fl, _ = fingerprint(sorted((P["train"] / "labels").glob("*.txt")), P["train"] / "labels")
    if fi != lock["all_images"]["fingerprint"] or fl != lock["final_labels_all"]["fingerprint"]:
        die(f"⚠ {site} 的训练数据和锁定时不一样。停止。")
    return imgs


def cmd_export_cea(a):
    """Odd-tree (CEA) photos and labels from cea_export -> data/sites/<site>/<site>_cea_odd.zip (for the CEA repo)."""
    import yaml
    P = site_paths(a.site)
    cea = P["d"] / "cea_export"
    if not (cea / "labels").exists():
        die("还没有 cea_export：先跑 merge-qc。")
    names = yaml.safe_load((cea / "data.yaml").read_text(encoding="utf-8"))["names"]
    out_dir = P["d"] / "cea_odd"
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True); (out_dir / "labels").mkdir()
    shutil.copy2(cea / "data.yaml", out_dir / "data.yaml")
    per_tree, boxes, unres = Counter(), Counter(), 0
    for p in images_in(cea / "images"):
        t = tree_of(p.stem)
        if not t or t % 2 == 0:
            continue
        shutil.copy2(p, out_dir / "images" / p.name)
        lab = cea / "labels" / f"{p.stem}.txt"
        shutil.copy2(lab, out_dir / "labels" / lab.name)
        arr = read_yolo(lab)
        arr = np.zeros((0, 5)) if arr is None else arr
        for c in arr[:, 0].astype(int):
            boxes[names[c]] += 1; unres += names[c] in ("leaf_spot_blight", "web_blight")
        per_tree[t] += 1
    n = sum(per_tree.values())
    note = [f"{a.site} CEA test set (odd trees only), made {now_myt()} from Snapdurian_deploy data/sites/{a.site}/cea_export",
            f"images: {n}  trees: " + ", ".join(f"t{t}={k}" for t, k in sorted(per_tree.items())),
            "boxes: " + ", ".join(f"{k}={v}" for k, v in sorted(boxes.items())),
            f"boxes not yet split into Phomopsis / leaf_rot (leaf_spot_blight, web_blight): {unres}"
            + ("  -> the CEA prepare step stops on these; split them in Roboflow (rename only), rerun "
               "`dl.py merge-qc --cea-only` and `dl.py export-cea` in Snapdurian_deploy" if unres else ""),
            "YOLO format: images/ labels/ data.yaml (class names are mapped by the CEA prepare step)."]
    (out_dir / "README.txt").write_text("\n".join(note) + "\n", encoding="utf-8")
    zp = P["d"] / f"{a.site}_cea_odd.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out_dir).as_posix())
    print("\n".join(note))
    print(f"\n已打包：{zp}（只含单数号树；不要 commit 这个 zip）")


def cmd_pack_train(a):
    P = site_paths(a.site)
    site_train_images(a.site)
    out = P["d"] / f"{a.site}_train_all.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted((P["train"]).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(P["train"]).as_posix())
    print(f"已打包：{out}（上传到 Drive 的 mixed_dataset/sites/，训练 notebook 会用到）")


def cmd_eval_folder(a):
    from dlcore.evalset import evaluate_folder
    r = evaluate_folder(a.model, a.images, a.labels, MERGE if a.merge else None)
    print(json.dumps(r, indent=2, ensure_ascii=False))


def write_train_files(name, merged, train_imgs, seed, out, extra_info):
    out = Path(out or DATA / "train" / name); out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("".join(p.resolve().as_posix() + "\n" for p in train_imgs), encoding="utf-8")
    val = (Path(merged) / "images" / "val").resolve()
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump({"train": (out / "train.txt").resolve().as_posix(), "val": val.as_posix(),
                                                   "nc": len(PRODUCT), "names": PRODUCT}, sort_keys=False), encoding="utf-8")
    rec = prelock()["recipe_for_v1_v3"]
    kw = {k: v for k, v in rec["fixed_args"].items() if k not in ("mode", "task", "model")}
    kw.update({"data": (out / "data.yaml").resolve().as_posix(), "seed": seed, "name": name})
    (out / "train_kwargs.json").write_text(json.dumps(kw, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    info = {"name": name, "created": now_myt(), "n_train_images": len(train_imgs), "seed": seed,
            "ultralytics_version": rec["ultralytics_version"], "model": rec["fixed_args"]["model"], **extra_info}
    write_json(out / "build_info.json", info)
    print(f"{name}: 训练图 {len(train_imgs)} 张，val {len(images_in(val))} 张 → {out}")
    print(f"训练：在 Colab 里用 notebooks/train_version_colab.ipynb（Ultralytics=={rec['ultralytics_version']}）")
    return out


def cmd_build_train(a):
    imgs = merged_train_images(a.merged)
    added = {}
    for s in a.sites:
        si = site_train_images(s); imgs += si; added[s] = len(si)
    write_train_files(a.name, a.merged, imgs, a.seed, a.out, {"kind": "version", "sites_added": added})


def cmd_build_swap(a):
    base = merged_train_images(a.merged)
    add_base = []
    for s in a.base_sites:
        add_base += site_train_images(s)
    new = site_train_images(a.new_site)
    n = len(new)
    groups = {}
    for r in csv.DictReader(io.StringIO(Path(a.manifest).read_text(encoding="utf-8-sig"))):
        if r.get("split") != "train":
            continue
        st = Path(re.sub(r"^\(未写出\)\s*", "", r["file"]).lstrip("./")).stem
        groups[st] = r.get("group") or st
    by_group = defaultdict(list); unmatched = 0
    for p in base:
        g = groups.get(p.stem) or groups.get(re.sub(r"_v\d+$", "", p.stem))
        if g is None:
            unmatched += 1; g = f"single:{p.stem}"
        by_group[g].append(p)
    rng = random.Random(a.removal_seed)
    order = sorted(by_group); rng.shuffle(order)
    removed = []
    for g in order:
        if len(removed) >= n:
            break
        members = sorted(by_group[g], key=lambda p: p.name)
        removed += members[: n - len(removed)]
    if len(removed) < n:
        die(f"merged_v5 训练集不够删：要删 {n} 张")
    rm = set(removed)
    imgs = [p for p in base if p not in rm] + add_base + new
    out = write_train_files(a.name, a.merged, imgs, a.seed, a.out,
                            {"kind": "swap", "removed_from_merged_v5": len(removed), "removal_seed": a.removal_seed,
                             "added_site": a.new_site, "added": n, "base_sites": a.base_sites,
                             "manifest_unmatched_images": unmatched})
    (out / "removed.txt").write_text("".join(p.name + "\n" for p in sorted(removed, key=lambda p: p.name)), encoding="utf-8")
    if unmatched:
        print(f"（提示）{unmatched} 张训练图在 manifest 里找不到分组，各自当成一组。")


# ============================================================ summary
def cmd_summary(a):
    S = {}
    for sd in sorted(SITES_DIR.iterdir()):
        if (sd / "site.json").exists():
            cfg = load_site(sd.name); r = RESULTS / sd.name
            S[sd.name] = {"cfg": cfg, **{k: json.loads((r / f"{k}.json").read_text(encoding="utf-8"))
                                         for k in ("scores", "qc", "anchor") if (r / f"{k}.json").exists()}}
    order = sorted(S, key=lambda s: S[s]["cfg"]["order"])
    L = ["# Summary (generated by `python dl.py summary`)", "", f"Generated {now_myt()}", "", "## H1"]
    diffs = []
    boots = []
    for s in order:
        sc = S[s].get("scores", {})
        if "H1_pair" in sc:
            h = sc["H1_pair"]; diffs.append(h["diff"])
            L.append(f"- {s}: {h['deployed']} - {h['previous']} = {h['diff']:+.4f} (95% CI {h['diff_ci95'][0]:+.4f} to {h['diff_ci95'][1]:+.4f})")
            bj = RESULTS / s / "bootstrap.json"
            if bj.exists():
                b = json.loads(bj.read_text(encoding="utf-8"))["mAP50_core"]
                boots.append(np.array(b[h["deployed"]]) - np.array(b[h["previous"]]))
    if diffs:
        L.append(f"- mean of {len(diffs)} paired differences: {np.mean(diffs):+.4f}; positive at {sum(d > 0 for d in diffs)} of {len(diffs)} sites")
        if len(boots) == len(diffs):
            m = np.mean(np.stack(boots), 0)
            L.append(f"- bootstrap 95% CI of the mean (sites resampled independently): {np.nanpercentile(m, 2.5):+.4f} to {np.nanpercentile(m, 97.5):+.4f}")
        L.append(f"- **H1 {'supported' if np.mean(diffs) > 0 else 'not supported'}** (primary criterion: mean > 0)"
                 + ("" if len(diffs) == 3 else f" — only {len(diffs)} of 3 sites available"))
    L += ["", "## H2"]
    for s in order:
        q = S[s].get("qc")
        if q:
            L.append(f"- {s} ({S[s]['cfg']['deployed_version']} pre-labels): {q['rate_per_image']:.3f} corrections/image (core {q['rate_per_image_core']:.3f}), QC hours {q.get('qc_hours')}")
    if "qc" in S.get(order[0], {}) and "qc" in S.get(order[-1], {}):
        f, l = S[order[0]]["qc"]["rate_per_image"], S[order[-1]]["qc"]["rate_per_image"]
        L.append(f"- **H2 {'supported' if l < f else 'not supported'}**: {order[-1]} {l:.3f} vs {order[0]} {f:.3f}")
    L += ["", "## H3 (swap S_k vs recipe of v_{k-1}; s0 of the recipe is the deployed version itself)"]
    wins = 0; steps = 0
    for s in order:
        sc = S[s].get("scores", {}); k = S[s]["cfg"]["order"] - 1
        if k < 1:
            continue
        sm = [m["mAP50_core"] for n, m in sc.get("models", {}).items() if re.fullmatch(fr"S{k}_s\d", n)]
        prev = f"v{k - 1}"
        rm = [m["mAP50_core"] for n, m in sc.get("models", {}).items() if re.fullmatch(fr"R{k - 1}_s\d", n) or n == prev]
        if sm and rm:
            steps += 1; win = np.mean(sm) > np.mean(rm); wins += win
            L.append(f"- step {k} at {s}: S{k} mean {np.mean(sm):.4f} (n={len(sm)}) vs v{k - 1} recipe mean {np.mean(rm):.4f} (n={len(rm)}) → {'S wins' if win else 'recipe wins'}")
    if steps:
        verdict = ("supported" if wins >= 2 else "not supported") if steps == 3 or wins >= 2 or steps - wins >= 2 else "undecided so far"
        L.append(f"- **H3 {verdict}**: S wins at {wins} of {steps} steps (needs >= 2 of 3)")
    L += ["", "## H4"]
    mc = nc = ms = ns = 0
    for s in order:
        an = S[s].get("anchor")
        if an:
            mc += an["corrected"]["matched_to_prelabel"]; nc += an["corrected"]["boxes"]
            ms += an["scratch"]["matched_to_prelabel"]; ns += an["scratch"]["boxes"]
            f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
            L.append(f"- {s}: corrected {f3(an['corrected']['share'])} ({an['corrected']['boxes']} boxes), scratch {f3(an['scratch']['share'])} ({an['scratch']['boxes']} boxes)")
    if nc and ns:
        n_sites = sum(1 for s in order if S[s].get("anchor"))
        L.append(f"- pooled over {n_sites} site(s): corrected {mc / nc:.3f} vs scratch {ms / ns:.3f} → **H4 {'supported' if mc / nc > ms / ns else 'not supported'}**"
                 + ("" if n_sites == len(order) else " (not all sites yet)"))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("rename"); s.add_argument("--site", required=True); s.add_argument("--export"); s.add_argument("--trees")
    s.add_argument("--date", help="trees.csv 只写了时分时用的日期 YYYY-MM-DD"); s.add_argument("--expect-model"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("pick-scratch"); s.add_argument("--site", required=True); s.add_argument("--n", type=int, default=40)
    s = sub.add_parser("lock-scratch"); s.add_argument("--site", required=True); s.add_argument("--scratch"); s.add_argument("--missing-as-empty", action="store_true")
    s = sub.add_parser("merge-qc"); s.add_argument("--site", required=True); s.add_argument("--export", required=True)
    s.add_argument("--missing-as-empty", action="store_true")
    s.add_argument("--cea-only", action="store_true", help="after lock: only rebuild the CEA copy; product labels must be unchanged")
    s.add_argument("--cea-drop-unresolved", action="store_true", help="drop leaf_spot_blight / web_blight boxes from the CEA copy")
    s = sub.add_parser("split"); s.add_argument("--site", required=True); s.add_argument("--final"); s.add_argument("--missing-as-empty", action="store_true"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("lock"); s.add_argument("--site", required=True); s.add_argument("--qc-hours", type=float)
    s.add_argument("--visit-date"); s.add_argument("--district"); s.add_argument("--note")
    s = sub.add_parser("score"); s.add_argument("--site", required=True); s.add_argument("--extra", nargs="*"); s.add_argument("--bootstrap", type=int, default=2000)
    s = sub.add_parser("qc"); s.add_argument("--site", required=True)
    s = sub.add_parser("anchor"); s.add_argument("--site", required=True)
    s = sub.add_parser("register"); s.add_argument("--version", required=True); s.add_argument("--onnx", required=True)
    s.add_argument("--server-file"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("build-train"); s.add_argument("--name", required=True); s.add_argument("--merged", required=True)
    s.add_argument("--sites", nargs="*", default=[]); s.add_argument("--seed", type=int, default=0); s.add_argument("--out")
    s = sub.add_parser("build-swap"); s.add_argument("--name", required=True); s.add_argument("--merged", required=True)
    s.add_argument("--manifest", required=True); s.add_argument("--base-sites", nargs="*", default=[]); s.add_argument("--new-site", required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--removal-seed", type=int, default=0); s.add_argument("--out")
    sub.add_parser("summary")
    s = sub.add_parser("export-cea"); s.add_argument("--site", required=True)
    s = sub.add_parser("pack-train"); s.add_argument("--site", required=True)
    s = sub.add_parser("preview"); s.add_argument("--site", required=True); s.add_argument("--final")
    s.add_argument("--missing-as-empty", action="store_true"); s.add_argument("--bootstrap", type=int, default=2000)
    s = sub.add_parser("eval-folder"); s.add_argument("--model", required=True); s.add_argument("--images", required=True)
    s.add_argument("--labels"); s.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    {"status": cmd_status, "rename": cmd_rename, "pick-scratch": cmd_pick_scratch, "merge-qc": cmd_merge_qc, "lock-scratch": cmd_lock_scratch,
     "split": cmd_split, "lock": cmd_lock, "score": cmd_score, "qc": cmd_qc, "anchor": cmd_anchor, "register": cmd_register,
     "build-train": cmd_build_train, "build-swap": cmd_build_swap, "summary": cmd_summary,
     "pack-train": cmd_pack_train, "export-cea": cmd_export_cea, "eval-folder": cmd_eval_folder, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
