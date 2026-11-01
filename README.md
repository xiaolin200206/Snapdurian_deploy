# Snapdurian_deploy

A pre-registered study of a **deploy–collect–retrain loop** for a durian leaf-disease detector running in a
progressive web app (SnapDurian), across four Malaysian states (Pahang → Penang → Malacca → Johor), 2026.

After each field site the detector is retrained on everything collected so far and redeployed before the next
site. The study asks how much each update improves accuracy on the next, not-yet-visited farm, and how the
human correction effort changes as the model improves.

**The protocol is fixed in [`sequential/prelock_note.md`](sequential/prelock_note.md)** (with
[`sequential/seq_prelock.json`](sequential/seq_prelock.json)), committed before the first site was visited.
If anything in this README disagrees with the prelock, the prelock wins.

## Hypotheses (summary; exact criteria in the prelock)

| | Prediction | Test data |
|---|---|---|
| H1 | Each redeployed version beats the previous one on the next farm (mAP50 over core classes) | even-numbered trees of sites 2–4 |
| H2 | Box corrections per image fall from the first site (v0 pre-labels) to the last (v3) | all photographs of each site |
| H3 | Swapping old images for the same number of new-farm images beats the previous recipe (test of Ruigrok et al. 2023) | even-numbered trees of sites 2–4 |
| H4 | Pre-labels anchor the annotator: corrected labels agree more with the model than from-scratch labels | 40 images per site |

A companion prospective test of farm-level generalisation uses the **odd-numbered trees** of the same visits
([durian-site-split](https://github.com/xiaolin200206/durian-site-split)); the two never share test images.

## Integrity chain

1. **Prelock** (this repository's first commit): v0 hashes, training recipe, evaluation rules, H1–H4.
2. **Version log** ([`sequential/versions_log.md`](sequential/versions_log.md)): the SHA-256 of every deployed
   version is committed before the site where it is first used.
3. **Per-site locks** (`sites/<site>/scratch_lock.json`, `sites/<site>/lock.json`): fingerprints of the
   from-scratch labels (before QC), of the test set, of all final labels and of the raw pre-labels.
4. **Scoring** (`dl.py score`) refuses to run if the test set, a model or the version log does not match.
5. **Deviations** ([`sequential/deviations.md`](sequential/deviations.md)): any change of plan is recorded
   before the data it affects exist; the prelock files are never edited.

## Layout

```
sequential/      prelock (protocol + JSON) and the version log
sites/<site>/    site.json, steps.md (step-by-step guide), trees.csv, reports and locks
dlcore/          inference (server-identical pre-processing), metrics (Ultralytics-style AP50), helpers
dl.py            one command per step: rename, pick-scratch, lock-scratch, split, lock, score, qc, anchor,
                 register, build-train, build-swap, pack-train, eval-folder, summary, status
notebooks/       Colab: evaluator check against Ultralytics; training v1–v3 and H3 models
results/<site>/  scores, bootstrap replicates, QC and anchoring results; results/summary.md
data/, models/   local only (not committed): images, labels, ONNX files
```

## Running

```
pip install -r requirements.txt
python dl.py status
python dl.py register --version v0 --onnx freeze/v0/durian_11cls_640_v5.onnx   # puts v0 at models/v0.onnx
```

Then follow `sites/<site>/steps.md` for each site, in order. Scoring runs on CPU (ONNX Runtime). Training runs
in Colab with Ultralytics 8.4.182 (`notebooks/train_version_colab.ipynb`). Before any site is scored, run
`notebooks/check_evaluator_colab.ipynb` once and commit `results/evaluator_check.json`.

## Data and model availability

- Test images (common classes), final labels, raw pre-labels, per-version predictions and all scores will be
  released with the paper. Images containing rare classes are withheld; locations are reported at district level.
- Weights of the deployed versions are **not** released (commercial reasons). Their SHA-256 are public here and
  the files can be provided confidentially to editors and reviewers for verification.
- Planned but not yet pre-registered: an offline replication on GWHD 2021; its plan will be committed before any
  GWHD model is trained or scored.

## License

No license has been chosen yet; until one is added, all rights are reserved.

---

### 作者用：每一站的顺序（详细的看 `sites/<站>/steps.md`）

1. 出发前：`python dl.py status`，这一站要用的版本已经登记、已经 push。
2. 现场：按走的顺序给树编号；填 `sites/<站>/trees.csv`（每棵树的开始和结束时间）。
3. 回来：导出 → `rename` → `pick-scratch`（commit）→ 从零标 → `lock-scratch`（commit）→ 隔 3 天 QC →
   `split` → `lock`（commit + 发邮件）→ `score` → `qc` → `anchor` → commit `results/`。
4. 训练下一版：`pack-train` → Colab 训练 → `register` → **commit + push 之后**才去下一站。
