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
