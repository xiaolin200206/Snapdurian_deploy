"""Facts about the Pahang data for the CEA record. Run from the repo root:  python audit_pahang.py
Writes data/sites/pahang/audit_for_cea.txt and data/sites/pahang/cea_unresolved_boxes.csv (both stay private)."""
import csv, io, re, sys, zipfile, tempfile, datetime as dt
from collections import Counter
from pathlib import Path
import numpy as np
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
import dl
from dlcore.common import PRODUCT, read_yolo, tree_of, images_in, export_names, label_files, xywhn_to_xyxy_px

SITE, DATE = "pahang", "2026-10-30"
P = dl.site_paths(SITE)
out = []
say = out.append

# ---------------- 1. photos in the questioned time windows
src = P["export_dir"] if P["export_dir"].exists() else None
if src is None:
    tmp = Path(tempfile.mkdtemp()); zipfile.ZipFile(P["export"]).extractall(tmp); src = tmp
pcsv = next(src.rglob("photos.csv"))
rows = list(csv.DictReader(io.StringIO(pcsv.read_text(encoding="utf-8-sig"))))
trees = []
for r in csv.DictReader(io.StringIO(P["trees"].read_text(encoding="utf-8-sig"))):
    st, en = r["start"].strip().replace(".", ":"), r["end"].strip().replace(".", ":")
    e = dl.parse_local(en, DATE) + (dt.timedelta(seconds=59) if en.count(":") == 1 else dt.timedelta())
    trees.append((int(r["tree"]), dl.parse_local(st, DATE), e))
trees.sort()
times = {i: dl.parse_ts(r.get("taken_at")) or dl.parse_ts(r.get("received_at")) for i, r in enumerate(rows)}
assign, notes, _ = dl.assign_trees(trees, times)
cuts = {}
for n in notes:
    m = re.match(r"t(\d+)\|t(\d+) at (\d\d:\d\d:\d\d)", n); cuts[(int(m[1]), int(m[2]))] = m[3]
rng = {t: (s, e) for t, s, e in trees}
renamed = {r["export_file"]: r["file"] for r in csv.DictReader(open(P["photos"], encoding="utf-8"))}

def rule(i):
    t = times[i]; k = assign.get(i)
    if k is None:
        return "not assigned: more than 120 s outside every tree's noted range"
    s, e = rng[k]
    if s <= t <= e:
        why = f"inside t{k}'s noted range {s:%H:%M}-{e:%H:%M:%S}"
    else:
        off = int(((s - t) if t < s else (t - e)).total_seconds())
        why = f"{off} s {'before' if t < s else 'after'} t{k}'s noted range {s:%H:%M}-{e:%H:%M:%S} (within the 120 s allowance)"
    sides = [f"t{a}|t{b} cut at {c}" for (a, b), c in cuts.items() if k in (a, b)]
    return why + "; " + ", ".join(sides)

windows = [("10:47:00", "10:47:59", "t11 and t12 both noted 10:47"), ("09:46:00", "09:53:59", "between t5 (ends 09:45) and t6 (starts 09:54)"),
           ("10:02:00", "10:12:59", "between t6 (ends 10:01) and t7 (starts 10:13)"), ("10:20:00", "10:20:59", "between t7 (ends 10:19) and t8 (starts 10:21)")]
say("1. Photographs in the questioned time windows (assignment rule: each boundary is cut in a pause between photos near "
    "the noted change, preferring the noted times; a photo goes to a tree if it lies between that tree's cuts and at most "
    "120 s outside its noted range; otherwise it is not assigned)")
say("   boundaries: " + "; ".join(notes))
for a, b, label in windows:
    lo, hi = dl.parse_local(a, DATE), dl.parse_local(b, DATE)
    sel = sorted((t, i) for i, t in times.items() if t and lo <= t <= hi)
    say(f"\n   {a[:5]}-{b[:5]} ({label}): {len(sel)} photo(s)")
    for t, i in sel:
        f = rows[i]["file"]; k = assign.get(i)
        say(f"     {t:%H:%M:%S}  -> {'t' + str(k) if k else 'none'}  ({renamed.get(f, f)})  {rule(i)}")

# ---------------- 2. odd trees vs v0 pre-labels
imgs = [p for p in images_in(P["images"]) if (tree_of(p.stem) or 2) % 2 == 1]
tot = Counter()
for p in imgs:
    pre = read_yolo(P["pre"] / f"{p.stem}.txt"); fin = read_yolo(P["final_lbl"] / f"{p.stem}.txt")
    pre = np.zeros((0, 5)) if pre is None else pre; fin = np.zeros((0, 5)) if fin is None else fin
    w, h = dl.img_size(p)
    tot.update(dl.match_counts(pre, fin, w, h))
say(f"\n2. Odd trees ({len(imgs)} photos), final labels vs v0 pre-labels (conf >= 0.25), 11 product classes, IoU 0.5:")
say(f"   added {tot['added']}, removed {tot['removed']}, relabelled {tot['relabelled']}, unchanged {tot['unchanged']}")
say("   (Phomopsis and leaf_rot both count as leaf_spot_blight here, so splitting a box into them is not a relabel.)")
qz = Path(ARGS[0]) if ARGS else P["d"] / "qc_export.zip"
tq = Path(tempfile.mkdtemp()); zipfile.ZipFile(qz).extractall(tq)
names = export_names(tq); lf = label_files(tq)
split, unresolved = Counter(), []
for p in imgs:
    if p.stem not in lf:
        continue
    lab = read_yolo(lf[p.stem]); lab = np.zeros((0, 5)) if lab is None else lab
    pre = read_yolo(P["pre"] / f"{p.stem}.txt"); pre = np.zeros((0, 5)) if pre is None else pre
    w, h = dl.img_size(p)
    for r in lab:
        prod, cea, suf = dl.qc_class(names[int(r[0])])
        if prod != "leaf_spot_blight" and cea not in ("web_blight",):
            continue
        origin = "_missing (added)" if suf == "missing" else "no _missing suffix"
        if cea in ("leaf_spot_blight", "web_blight"):
            x1, y1, x2, y2 = xywhn_to_xyxy_px(r[None], w, h)[0]
            unresolved.append({"file": p.name, "tree": tree_of(p.stem), "class_now": cea, "origin": origin,
                               "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)})
            split[(cea, "not split yet")] += 1
        else:
            split[(cea, origin)] += 1
say("   leaf-spot / leaf-blight boxes on odd trees, by CEA name:")
for (c, o), n in sorted(split.items()):
    say(f"     {c:<18} {o:<22} {n}")
say(f"   not yet split into Phomopsis / leaf_rot: {len(unresolved)} (listed in cea_unresolved_boxes.csv)")
with open(P["d"] / "cea_unresolved_boxes.csv", "w", newline="", encoding="utf-8") as f:
    w_ = csv.DictWriter(f, fieldnames=["file", "tree", "class_now", "origin", "x1", "y1", "x2", "y2"]); w_.writeheader(); w_.writerows(unresolved)
missing_odd = [p.name for p in imgs if p.stem not in lf]
say(f"   odd-tree photos absent from the QC export (entered with no boxes): {missing_odd or 'none'}")

# ---------------- 3. what the PWA received
days = Counter((dl.parse_ts(r.get("received_at")) or dt.datetime.min).strftime("%Y-%m-%d") for r in rows)
say(f"\n3. PWA export used for Pahang: {len(rows)} photos (received dates: {dict(days)}); "
    f"{len(renamed)} fall inside a tree's range, {len(rows) - len(renamed)} not assigned.")
say("   To confirm on the server how many photos it holds for that day (and that none were removed before export), run there:")
say("   docker compose exec app python -c 'import sqlite3; c=sqlite3.connect(\"/data/snapdurian.db\"); "
    "print(c.execute(\"SELECT code, COUNT(*), MIN(received_at), MAX(received_at) FROM photos WHERE day=? GROUP BY code\", "
    "(\"2026-10-30\",)).fetchall())'")
rep = P["d"] / "audit_for_cea.txt"
rep.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out)); print(f"\n-> {rep}\n-> {P['d'] / 'cea_unresolved_boxes.csv'}")
