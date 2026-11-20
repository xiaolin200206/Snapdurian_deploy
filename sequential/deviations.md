# Deviations from the prelock

Each entry is committed before the data it affects exist. The prelock files themselves are never edited.

## D1 — Malacca and Johor visited in one trip (recorded 2026-10-29, before either site was visited)

- Malacca (site 3) and Johor (site 4) will be visited on consecutive days in a single trip.
- There is no time to QC the Malacca photographs and train v3 between the two visits, so **Johor is
  pre-labelled by v2**, not v3. This is the case foreseen in prelock section 2 ("If a version cannot be
  trained in time, the site is pre-labelled by the latest available version").
- v3 is trained after the trip exactly as planned (v2's training set + all Malacca images, v0 recipe) and is
  scored offline on the Johor even-tree test set. **H1 at Johor is unchanged: v3 vs v2.**
- **H2**: the last point (Johor) uses v2 pre-labels. H2 is reported as written (Johor < Pahang) with this
  noted; the pre-labelling version of every site is reported next to its correction rate.
- H3 (step 3: S3 vs the v2 recipe, scored on Johor) and H4 are unaffected.
- Code: `sites/johor/site.json` records `deployed_version: v2` and `h1_newer: v3`, `h1_older: v2`.

## D2 — Exploratory two-phone comparison (recorded 2026-10-29, before any site was visited)

Added to address the single-phone limitation. **Exploratory: not part of H1–H4, never in any test or training set
of this study, and not part of the CEA prospective test.**

- **When**: at each site, after the last tree of the main walk, so these photographs fall outside every tree time
  range in `trees.csv` and `dl.py rename` leaves them out of the main data automatically.
- **What**: a few trees already photographed in the main walk (2–3 per site); on each, 10 photographs with the
  main phone (iPhone 16 Pro Max, PWA, main invite code) and 10 with a second phone (Xiaomi Mi 10S, PWA, a separate
  invite code), in the same session, of symptomatic leaves of that tree. Default camera settings on both phones.
- **Labels**: pre-labelled by the deployed version and corrected under the same rules (prelock section 4).
- **Analysis (planned, descriptive)**: for each version that exists when a site is scored, mAP50-core on the iPhone
  photographs and on the Xiaomi photographs of the same trees, and the difference, per site and pooled; no
  prediction is made. Because the two phones photograph the same trees but not necessarily the same leaves, the
  difference combines the effect of the phone with leaf-to-leaf variation; this is stated when reported.
- These photographs are never added to the training data of any version.

## D3 — Pahang: two farms, 15 trees (recorded 2026-11-01, after the visit, before any scoring)

- Pahang was visited on 2026-10-30 (Bentong district). The first farm was uniform in its symptoms, so the walk
  continued at a second farm nearby: **t1–t5 at farm A, t6–t15 at farm B**. Both are new to the corpus.
- 15 trees were photographed (planned: about 20), 321 photographs; 320 fall inside a tree's time range and are
  used, 1 (taken between farms) is outside every range and is not used (`sites/pahang/rename_report.txt`).
- The two-phone comparison (D2) was not carried out at Pahang.
- The tree rule is unchanged: even trees (7 trees) are this study's Pahang test set; all 15 trees go into v1's
  training set. Tree-level bootstrap treats each tree as one unit regardless of farm.

## D4 — H4 dropped (recorded 2026-11-01, before any H4 data were locked or analysed)

- H4 (anchoring of the annotator by pre-labels) is dropped for all sites: the from-scratch annotation was judged not
  worth its cost for a deployment-focused study. No H4 result will be reported.
- At Pahang the 40-image subset was drawn (`sites/pahang/scratch_subset.txt`) and partly annotated from scratch;
  that annotation was never locked and is not used. The subset images were corrected with all other photographs.

## D5 — Annotation rules, settled during QC of site 1 and applied unchanged at all later sites (recorded 2026-11-01, before any scoring)

These refine prelock section 4; they apply to every photograph of every site, test and training alike.

1. A lesion is boxed only if the annotator is confident of its class and its extent can be boxed at the default
   full-image view of the annotation tool. Uncertain, unboxable or too-small lesions are left unlabelled, and a
   pre-label on such a lesion is deleted. Labels reflect visual identification from the photograph by the author
   (with a field practitioner's treatment-based class definitions); no laboratory confirmation was made.
2. Leaf-spot / leaf-blight lesions are labelled as Phomopsis or leaf_rot where the annotator can tell them apart
   (leaf_rot covers Rhizoctonia, Phytophthora and Colletotrichum foliar lesions), and as leaf_spot_blight where
   the annotator is confident it is one of these but cannot tell which. In this study all three are leaf_spot_blight.
3. No photograph is excluded for being blurred or ambiguous. A photograph without a confidently identifiable lesion
   is kept with no boxes. At Pahang two photographs were left unannotated in the tool and enter with no boxes.
4. Added and changed boxes may carry a `_missing` / `_modify` suffix in the annotation tool for the annotator's own
   bookkeeping; suffixes are removed when the export is merged (`dl.py merge-qc`). QC counts (H2) are always computed
   by matching the final labels against the stored pre-labels, never from the suffixes.

## D6 — Exploratory input-size comparison, 640 vs 1280 (recorded 2026-11-02, before any 1280 model was trained or scored)

Motivation: the deployed models see each photograph letterboxed to 640 x 640, and the Pahang QC showed that most of
what v0 missed were small lesions in wider shots. This comparison is **exploratory**: not part of H1–H4, no 1280 model
is deployed or used for pre-labelling during the study, and nothing in the prelock changes.

- **Shadow models.** For each deployed version v_k, a shadow model v_k_1280 is trained on exactly v_k's training set
  with v_k's recipe (same Ultralytics version, `yolo11s.pt` start, 150 epochs, AdamW, lr0 0.001, seed 0, close_mosaic 10,
  same fixed validation split), changing only `imgsz` 640 → 1280 and, if GPU memory requires it, `batch` 16 → 8
  (Ultralytics accumulates gradients to a nominal batch of 64, so the optimisation schedule is unchanged) and
  `cache` ram → disk. The exact arguments of every shadow run are in its `build_info.json` (`exploratory_overrides`).
- **Scoring.** v_k_1280 is scored with the locked evaluator on the same even-tree test set as v_k, at the same site
  (v0_1280 vs v0 on Pahang, v1_1280 vs v1 on Penang, and so on); the only difference in evaluation is the letterbox
  size, which is read from each ONNX model's input shape. Reported: mAP50-core of both, and the paired tree-bootstrap
  difference v_k_1280 − v_k (2000 replicates, seed 0), overall and on two image subsets split at the site median of
  each image's median relative ground-truth box side ("near" = larger lesions, "far" = smaller lesions; images
  without boxes excluded). The split depends on the labels only, never on a model.
- **Decision rule, fixed now.** 1280 is adopted for the product after the study if, pooled over the sites where both
  models were scored, the mean paired difference v_k_1280 − v_k is > 0 with a 95% bootstrap interval excluding 0, and
  the server's median per-photo inference time at 1280 is at most 3 x that at 640 on the same CPU. Otherwise 640 stays.
- Shadow models are kept in `freeze/<name>/` and listed with their SHA-256 in `results/<site>/scores.json`
  (`kind: extra`); they are not entered in `versions_log.md`.

## D7 — Klang (Selangor) added as site 3; Malacca and Johor move to sites 4 and 5 (recorded 2026-11-05, before Klang is visited)

- **New site.** A farm near Klang, Selangor (introduced by the field practitioner; owner contacted 2026-11-05) is
  visited about 2026-11-14 as **site 3, pre-labelled by v2** (v2 = v1's training set + all Penang images, v0 recipe,
  registered and pushed before the visit). Site name `klang`, file-name prefix `kg1`. The new-farm rule applies: if the
  farm turns out to be one already in the training corpus, the site is excluded from H1/H2 and its images are used
  for training only. H1 at Klang: v2 vs v1 on the Klang even-tree test set.
- **Series extended.** The prelock planned four sites and three H1 steps. The series is now Pahang (v0), Penang (v1),
  Klang (v2), then Malacca and Johor as sites 4 and 5 (planned pre-labelling versions v3 and v4; a visit to both in
  one trip would be handled as D1 described: the second farm pre-labelled by the latest available version, its newer
  version scored offline). Order and versions for sites 4 and 5 are provisional and will be confirmed in a deviation
  recorded before each visit. D1 as written (Malacca then Johor the next day, Johor by v2) no longer applies.
- **Hypotheses.** H1's primary criterion becomes the mean of the paired differences over **all** new-farm sites from
  site 2 on (at least the three originally planned); each site's pair is still (deployed version) − (previous version).
  H2 compares the **last new-farm site** with Pahang. H3 stays at the three originally planned steps (sites 2–4,
  now Penang, Klang, Malacca). D6 shadow models continue: one v_k_1280 per deployed v_k, scored on the site it shadows.
- **Possible revisit.** If time allows, a final version (v5 or the last trained) may be tested on a farm already in
  the training data (Pahang or Seremban). A revisit is not a new-farm site: it is excluded from H1 and H2 and reported
  separately as exploratory (within-farm change over time), with its own deviation recorded before the visit.
- **CEA.** Klang is not part of the CEA prospective series unless the CEA repository pre-registers it before the
  visit; otherwise Klang's odd-numbered trees are used for training only. The odd/even rule and all other prelock
  rules are unchanged.

## D8 — Penang: 8 trees, mostly young (recorded 2026-11-07, after the visit, before any scoring)

- Penang was visited on 2026-11-06 (Barat Daya district, Balik Pulau). The farm is new to the corpus. Only
  **8 trees** could be photographed (planned: about 20): most trees on the farm are young, and on the older, taller
  trees the leaves are out of reach from the ground. About 113 photographs were taken; 109 fall inside a tree's time
  range and are used (`sites/penang/rename_report.txt`, `sites/penang/trees.csv`). Trees 5 and 6 share the minute
  15:04; photographs in that minute are assigned at the pause between the two trees, by the same rule as at every
  site.
- The owner reports spraying little; insect pests (psyllid) were abundant on the photographed trees.
- Photographs deleted in the annotation tool as background are kept with no boxes (D5 rule 3; merged with
  `--missing-as-empty`).
- The tree rule is unchanged: the even trees (t2, t4, t6, t8; 4 trees) are this study's Penang test set and all
  8 trees go into v2's training set. With 4 test trees the Penang tree-bootstrap interval will be wide; this follows
  from the farm, not from any choice made after seeing results. Penang stays in H1 (v1 − v0) and is not the H2 site.

## D9 — H3 step-1 models trained after Penang was scored (recorded 2026-11-08, before any H3 model was trained)

- The H3 models of step 1 (S1_s0, S1_s1, S1_s2, R0_s1, R0_s2) were not trained before Penang was scored for H1
  (v0, v1) and D6 (v1_1280): the workflow checklist did not list H3, and this was noticed on 2026-11-07, after that
  scoring. H3 is kept as pre-registered.
- The step-1 models are trained now with the prelock recipe, removal seed 0 and training seeds 0–2, from merged_v5
  and Pahang only; no Penang image or label enters their training, so having seen Penang's v0/v1 scores cannot
  influence them. Their SHA-256 are committed in `sequential/h3_models.md` before they are scored on the locked
  Penang test set (lock unchanged). Adding models to `dl.py score` leaves the v0, v1 and v1_1280 results unchanged
  (deterministic inference; the tree bootstrap draws one resample per replicate, shared by all models, seed 0).
- Steps 2 and 3 follow the prelock as written; their hashes are likewise committed before the site is scored.

## D10 — Evaluator check run after Pahang and Penang were scored (recorded 2026-11-08, before the check was run)

- The prelock (section 5; `seq_prelock.json`, `evaluation.implementation`) requires the evaluator of `dl.py score`
  to be checked against Ultralytics 8.4 validation of v0 on the merged_v5 validation split before any site is
  scored (tolerance 0.01 all-class mAP50). The check was not run before Pahang (scored 2026-11-02) or Penang
  (scored 2026-11-07); this was noticed on 2026-11-08.
- The check is run now with `notebooks/check_evaluator_colab.ipynb` as committed (v0 ONNX verified against the
  prelock hash; conf 0.001, NMS IoU 0.7, imgsz 640) and its output is committed as `results/evaluator_check.json`
  before any further scoring (H3 step 1 at Penang, then Klang). The check uses only v0 and the merged_v5 validation
  images, no site data, so having seen the site scores cannot influence it. For reference, the prelock already holds
  an Ultralytics validation of the v0 ONNX on the same split, recorded before any site was visited
  (`v0.reference_val_onnx`, all-class mAP50 0.3932); the notebook prints it alongside.
- If the all-class mAP50 difference is ≤ 0.01, the Pahang and Penang results stand unchanged. If it is larger, the
  difference is reported, the evaluator is corrected as the prelock specifies, Pahang and Penang are re-scored with
  the corrected evaluator on their unchanged locked test sets, and both the original and the re-scored results are
  reported, the re-scored ones as primary.

## D11 — Evaluator corrected after the D10 check failed (recorded 2026-11-08, before any re-scoring)

- The D10 check failed: all-class mAP50 of v0 on the merged_v5 validation split was 0.4337 with this repository's
  evaluator and 0.3932 with Ultralytics 8.4.182 (identical to the prelock reference `v0.reference_val_onnx`), a
  difference of +0.0405 (tolerance 0.01). Every class was higher; the largest differences were in rare classes
  (root_disease +0.089, stem_borer +0.086, web_blight +0.075). The failed result is kept as
  `results/evaluator_check_v1_failed.json`.
- Cause: `dlcore` (evaluator version 1) followed the Ultralytics 8.3 form of two functions, while the prelock names
  Ultralytics 8.4. (a) `compute_ap`: 8.4 drops the precision envelope to 0 at the last recall reached; version 1
  interpolated linearly from the last recall to recall 1, adding area that is largest for classes with low recall.
  (b) `match_predictions`: 8.4 takes detections in descending confidence and lets each claim the best unclaimed target
  of its class; version 1 used IoU-sorted matching. (c) Version 1 also clipped predicted boxes to the image before
  matching; Ultralytics validation matches unclipped boxes.
- Correction (evaluator version 2; `EVALUATOR_VERSION = 2` in `dlcore/metrics.py`): (a)–(c) changed to the
  Ultralytics 8.4.182 behaviour. Checked against the Ultralytics 8.4.182 functions themselves on about 30,000 random
  predictions: identical matching and identical AP. Pre-processing is unchanged (server-identical PIL letterbox, as the
  prelock specifies), so remaining differences from Ultralytics validation come only from pre-processing. Prediction
  caches carry the evaluator version in their names, so version-1 predictions are not reused. `scores.json` records
  `evaluator_version`.
- The check is re-run with the unchanged notebook and committed as `results/evaluator_check.json`. Only if it passes
  (|difference| <= 0.01) are Pahang and Penang re-scored with version 2 on their unchanged locked test sets. Their
  version-1 results are first copied to `results/<site>/evaluator_v1/` and are reported alongside; the version-2
  results are primary, as D10 states. If the check fails again, nothing is scored until the cause is found.
- The correction depends on no site result: it only makes the evaluator match the implementation the prelock names.

## D12 — Evaluator check: implementation verified; the remaining difference is the pre-registered server pre-processing (recorded 2026-11-08, before any re-scoring)

- With evaluator version 2 (D11) the check still differed by +0.0333 (`results/evaluator_check_v2_failed.json`).
  A diagnosis on the same merged_v5 validation split with v0 (`notebooks/diagnose_evaluator_colab.ipynb`, output in
  `results/evaluator_diagnosis.json`) changed one thing at a time:
  - with Ultralytics' own pre-processing (OpenCV decoding, long side resized to 640 with INTER_LINEAR, `LetterBox`),
    evaluator version 2 gives all-class mAP50 0.3932, identical to Ultralytics 8.4.182 validation (difference 0.0000;
    every class within 0.0001);
  - Ultralytics' label rules (duplicate boxes removed, corrupt images skipped) change nothing on this split;
  - with the server pre-processing (PIL bilinear) it gives 0.4265 (+0.0333); with OpenCV INTER_AREA 0.4169 (+0.0237).
- So the whole remaining difference is pre-processing. PIL's bilinear resize filters (anti-aliases) when it shrinks an
  image; OpenCV INTER_LINEAR, used by Ultralytics for validation and training, does not. The merged_v5 images are
  mostly phone originals of about 4000 px, shrunk about six times to 640.
- The prelock fixed both the server-identical pre-processing and the 0.01 check, assuming pre-processing differences
  would be negligible; on this split they are not. The purpose of the check, that `dl.py score` computes Ultralytics 8.4
  AP correctly, is met: given the same pre-processed images it reproduces Ultralytics exactly. Scoring therefore keeps
  the pre-registered server-identical pre-processing, which is what growers receive. No site result was used for this
  decision; both failed checks and the diagnosis are committed.
- Pahang and Penang are now re-scored with evaluator version 2 as D11 describes (version-1 results kept in
  `results/<site>/evaluator_v1/`). Site test images are app uploads (long side at most 1600 px), so the pre-processing
  effect there may differ from merged_v5 validation. As an exploratory sensitivity analysis, H1 will also be reported
  with Ultralytics pre-processing after the last site.
- Exploratory observation: training used OpenCV INTER_LINEAR while the server uses PIL; on merged_v5 validation this
  mismatch favoured the server pre-processing for v0 (+0.0333 mAP50).

## D13 — Klang cancelled; Johor becomes site 3 with v2, Malacca site 4 and the last new farm (recorded 2026-11-16, before either visit)

- The Klang grower did not give an address or a date on two attempts (2026-11-13 and 2026-11-15), so Klang (D7,
  site 3) is cancelled. No Klang photograph was taken; nothing about Klang enters any analysis.
- **Site 3 is a farm in the Bukit Gambir area, Johor**, introduced through a third party, never visited and not in any
  training data (the author had visited only Muar farms before; none of them is in merged_v5 or in any site of this
  study). Prefix `jh1`. The deployed version is **v2**, registered and pushed in a898c41 and live on the server since
  2026-11-08; nothing was retrained for this change. The visit date and district are recorded in the lock.
- Johor was already in the CEA prelock, so its odd-numbered trees go to the CEA test set as that prelock specifies.
- What moves from Klang to Johor, all committed before this deviation: the v2_1280 shadow model (D6) and H3 step 2
  (S2_s0–s2, R1_s1–s2; `sequential/h3_models.md`). H3 step 3 (S3 with Johor as the new site, R2) is trained after the
  Johor visit and scored at Malacca.
- **Site 4 is Malacca** with v3 (trained on merged_v5 + Pahang + Penang + Johor, prelock recipe), registered and
  pushed before that visit. Malacca is the last new-farm site: H1 has three paired differences (Penang v1−v0, Johor
  v2−v1, Malacca v3−v2), the "at least three originally planned" steps of D7; H2 compares Malacca with Pahang; H3 has
  its three steps at Penang, Johor and Malacca. There is no fifth site and no revisit.
- If Malacca cannot be visited, Johor becomes the last new-farm site and H1 and H3 have two steps; this is recorded in
  a further deviation as soon as the Malacca visit is known to be off, and the reduced design is reported as such.

## D14 — Johor field and QC notes (recorded 2026-11-20, after Johor was scored, before the Malacca visit)

None of these changes a hypothesis, a decision rule, a model or the Johor lock (`b0d542dc…`).

- **Rain.** Johor was visited on 2026-11-19 in the monsoon season; it rained through the working window, and 11 trees
  were photographed (191 photographs; 114 on odd-numbered trees for the CEA test set, 77 on 5 even-numbered test trees).
- **No separator photographs.** Trees were assigned from the change times noted in the field: each boundary was cut at
  the longest pause between consecutive photographs near the noted time (pauses of 35–94 s), and the first two and last
  two photographs of every tree were checked by eye (`tree_check.jpg`). No photograph was left unassigned. All 191
  photographs were pre-labelled by the deployed v2 (`durian_11cls_640_v2_20261107_0802aeb0`).
- **Rain reflections pre-labelled as psyllid.** Wet-leaf highlights were often detected as `psyllid` by v2. They were
  removed in QC; after QC, psyllid has 4 test boxes and leafhopper_damage 2, so both fall below the 10-box threshold and
  the Johor core set is algal_spot, leaf_spot_blight and psyllid_damage. In the grower view (`results/johor/farmer.json`)
  the app named psyllid on 106 photographs, 7 of which showed psyllid after QC.
- **One pink_disease box deleted by mistake in QC**, noticed after the lock. It was not restored, so the locked
  annotation is used as it is; the tree of that image was not identified. pink_disease is not a core class at Johor.
- **QC time was not recorded** for Penang or Johor (only the number of box changes, `dl.py qc`). From Malacca on, QC
  hours, hours on site, travel time, weather on the day and recent spraying (asked from the grower) are recorded.
- **Pre-label anchoring, flagged before the next site is scored.** Johor test labels were corrected from v2 pre-labels
  with fewer changes than at earlier sites (5.63 box changes per image, against 11.69 at Pahang and 13.18 at Penang).
  This can favour v2 over the other models scored at Johor (§7 limitation; H4 was cancelled under D4). In the
  exploratory size split, v2 scores 0.622 on the larger-lesion half against 0.297–0.435 for every other model, a gap
  much larger than on the smaller-lesion half. The Johor H1 difference (v2 − v1 = +0.0965, 95% CI [+0.0214, +0.1254])
  is reported with this caveat.
- **H3 after two steps.** S beats the recipe at Penang (+0.1042) and at Johor (S2 mean 0.4964 vs recipe mean 0.4550,
  +0.0413), so the prelock rule (at least 2 of 3 steps) is already met. Step 3 at Malacca is still trained (committed in
  e08b223), scored and reported in full; the rule is not changed.
- **v3** was registered and pushed in e08b223 and has been live on the server since 2026-11-20, before the Malacca
  visit; the v3_1280, S3 and R2 hashes are in the same commit.
