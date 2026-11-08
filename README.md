# Snapdurian_deploy

A pre-registered study of a **deploy–collect–retrain loop** for a durian leaf-disease detector running in a
progressive web app (SnapDurian), on new farms in Malaysia, 2026: Pahang → Penang → Klang (Selangor) → Malacca →
Johor (sites 4 and 5 provisional; see deviation D7).

After each field site the detector is retrained on everything collected so far and redeployed before the next
site. The study asks how much each update improves accuracy on the next, not-yet-visited farm, and how the
human correction effort changes as the model improves.

**The protocol is fixed in [`sequential/prelock_note.md`](sequential/prelock_note.md)** (with
[`sequential/seq_prelock.json`](sequential/seq_prelock.json)), committed before the first site was visited.
Every later change of plan is in [`sequential/deviations.md`](sequential/deviations.md) (D1–D12 so far), recorded
before the data it affects exist. If this README disagrees with the prelock plus deviations, those win.

## Hypotheses (summary; exact criteria in the prelock and deviations)

| | Prediction | Test data |
|---|---|---|
| H1 | Each redeployed version beats the previous one on the next farm (mAP50 over core classes); primary criterion: mean of the paired differences over all new-farm sites > 0 (D7) | even-numbered trees, sites 2–5 |
| H2 | Box corrections per image are lower at the last new-farm site than at the first (v0 pre-labels) | all photographs of each site |
| H3 | Swapping old images for the same number of new-farm images beats the previous recipe (test of Ruigrok et al. 2023); 3 steps, scored at Penang, Klang, Malacca (D7) | even-numbered trees |
| H4 | *Dropped before any H4 data were collected (D4).* | – |

Exploratory, not part of the predictions: 640 vs 1280 input size (D6; shadow models in
[`sequential/shadow_models.md`](sequential/shadow_models.md)) and what the app told the grower per photo
(`dl.py farmer`).

A companion prospective test of farm-level generalisation uses the **odd-numbered trees** of the same visits
([durian-site-split](https://github.com/xiaolin200206/durian-site-split)); the two never share test images.

## Integrity chain

1. **Prelock** (this repository's first commit): v0 hashes, training recipe, evaluation rules, hypotheses.
2. **Version log** ([`sequential/versions_log.md`](sequential/versions_log.md)): the SHA-256 of every deployed
   version is committed before the site where it is first used. Shadow models: `sequential/shadow_models.md`.
3. **Per-site locks** (`sites/<site>/lock.json`): fingerprints of the test set, of all final labels and of the raw
   pre-labels; the SHA-256 of each lock is also e-mailed to the author on the day it is made.
4. **Scoring** (`dl.py score`) refuses to run if the test set, a model or the version log does not match.
5. **Deviations** (`sequential/deviations.md`): the prelock files are never edited.

## Layout

```
sequential/      prelock (protocol + JSON), version log, deviations, shadow models
sites/<site>/    site.json, steps.md (step-by-step guide), trees.csv, reports and lock
dlcore/          inference (server-identical pre-processing), metrics (Ultralytics-style AP50), helpers
dl.py            one command per step: status, rename, merge-qc, split, lock, score, qc, farmer, export-cea,
                 register, build-train, build-swap, pack-train, eval-folder, preview, summary, release, tidy
notebooks/       Colab: evaluator check against Ultralytics; training of versions, shadow and H3 models
results/<site>/  scores, bootstrap replicates, QC and per-photo app results; results/summary.md
data/, models/   local only (not committed): images, labels, ONNX files
```

## Running

```
pip install -r requirements.txt
python dl.py status
python dl.py register --version v0 --onnx freeze/v0/durian_11cls_640_v5.onnx   # puts v0 at models/v0.onnx
```

Then follow `sites/<site>/steps.md` for each site, in order. Scoring runs on CPU (ONNX Runtime). Training runs
in Colab with Ultralytics 8.4.182 (`notebooks/train_version_colab.ipynb`). The evaluator is checked once against
Ultralytics validation of v0 (`notebooks/check_evaluator_colab.ipynb`, result in `results/evaluator_check.json`).

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

1. 出发前：`python dl.py status`，这一站要用的版本已经登记、已经 push；影子模型的哈希已经写进 shadow_models.md。
2. 现场：按走的顺序给树编号；填 `sites/<站>/trees.csv`（每棵树的开始和结束时间）。
3. 回来：服务器导出 → `rename` → Roboflow 逐框 QC（D5 规则）→ `merge-qc` → `split` → `lock`（commit + push + 发邮件）
   → `score`（带上 1280 影子和 H3 的模型）→ `qc` → `farmer` → commit `results/`。
4. 训练下一版：`pack-train` → Colab 训练 → `register` → **commit + push 之后**才去下一站。
   同一份数据顺便训 1280 影子和 H3 的模型（`dl.py status` 会显示还差哪些）。
