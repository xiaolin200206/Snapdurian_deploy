# Sequential deploy–collect–retrain study: prelock

Written and committed BEFORE the first site of the series (Pahang) was visited.
Companion to, and separate from, the CEA prospective test (odd-numbered trees);
this study uses the even-numbered trees as test sets.

## 1. Question

In a deployed system (SnapDurian PWA), the detector is retrained after each field site and
redeployed before the next. How much does each update improve accuracy on the next, not yet
visited farm, and how does the human correction effort change as the model improves?

## 2. Sites and versions (in visiting order)

| Site (order) | Pre-labelling / deployed version | Trained on |
|---|---|---|
| 1 Pahang  | v0 | merged_v5 (frozen; hashes in `seq_prelock.json`) |
| 2 Penang  | v1 | v0 training set + all Pahang images |
| 3 Malacca | v2 | v1 training set + all Penang images |
| 4 Johor   | v3 | v2 training set + all Malacca images |

- "All images" of a site = every photograph of every tree (odd and even), annotated in the
  11-class product label space after QC. New-site images go to the training split only; the
  validation split (merged_v5 val, 451 images) is fixed for all versions and used only for
  checkpoint selection.
- Each v_k is trained with the v0 recipe (Ultralytics 8.4.182, YOLO11s, args recorded in
  `seq_prelock.json`) with only the dataset path, run name and output directory changed;
  initialised from the same COCO-pretrained `yolo11s.pt`, i.e. a full retrain, not a fine-tune.
- The deployed version is `best.pt` of that run (Ultralytics fitness on the fixed val split),
  exported to ONNX with the v0 export settings (opset 12, simplify, static batch 1, 640 x 640).
- The SHA-256 of each v_k ONNX is committed to the repository (`versions_log.md`) before the
  site where it is first deployed is visited. All photographs of a site are uploaded before
  the next version is switched on; the per-photo `model` field in the PWA export is checked.
- If a version cannot be trained in time, the site is pre-labelled by the latest available
  version; the comparisons it affects are reported as such. A cancelled site is reported and
  its comparisons dropped.
- Novelty rule as in the CEA prelocks: a site that turns out to be a farm already in the corpus
  is excluded from the tests (its images may still be used for training).

## 3. Test sets

- Even-numbered trees (t2, t4, ...) of each site, all photographs, fully annotated.
- These images are never in the training set of any version evaluated on them (v_k is trained
  only on sites visited before site k+1).

## 4. Annotation rules (product label space, 11 classes)

- Same rules as the CEA prelock notes, except that `leaf_spot_blight` is NOT split into
  Phomopsis / Leaf_rot (product label space).
- White waxy insects on the upper leaf surface are labelled `psyllid`; `scale_insect` only when
  confirmed on the leaf underside. Undersides are not deliberately photographed.
- Rare classes (`pink_disease`, `root_disease`, `stem_borer`, `web_blight`, `weevil_damage`)
  are labelled when seen; no prediction is made about them.

## 5. Evaluation (offline, after each visit)

- Each version is run offline on the test images: ONNX, 640 x 640, conf 0.001, NMS IoU 0.7,
  max 300 detections (the stored PWA detections, thresholded at 0.25, are NOT used).
  Pre-processing is identical to the deployed server (EXIF transpose, letterbox to 640 with grey
  114 padding, PIL bilinear resize); post-processing follows Ultralytics validation (multi-label
  candidates with class score > 0.001, top 30,000 by score, class-aware NMS, at most 300 boxes).
- Implementation: `dl.py score` in this repository. Before any site is scored, the evaluator is
  checked against Ultralytics 8.4 validation of v0 on the merged_v5 val split
  (`notebooks/check_evaluator_colab.ipynb`); if all-class mAP50 differs by more than 0.01, the
  difference is reported and the evaluator is corrected before scoring.
- Before scoring, `scale_insect` is mapped to `psyllid` in both ground truth and predictions.
- AP50 per class as in Ultralytics 8.4 (`ap_per_class`, 101-point interpolation, IoU 0.5).
- **Primary metric, mAP50-core**: mean AP50 over the core classes
  {algal_spot, leaf_spot_blight, leafhopper_damage, psyllid (incl. scale_insect), psyllid_damage}
  that have at least 10 ground-truth boxes on that site's test set. At a given site the same
  class set is used for every version compared.
- Secondary (reported, no prediction): mAP50 over all classes with at least 1 box.
- Uncertainty: tree-level bootstrap (resampling trees with replacement within site, 2,000
  replicates, seed 0) for each site's score and for paired differences between versions; a core
  class with no box in a replicate is left out of that replicate's mean. The pooled mean of the
  three H1 differences gets its interval by resampling each site independently.

## 6. Predictions

- **H1 (updates help on the next farm).** At sites 2–4, the deployed version scores higher in
  mAP50-core than the version before it on the same test images:
  v1 > v0 at Penang, v2 > v1 at Malacca, v3 > v2 at Johor.
  Primary criterion: the mean of the three paired differences is > 0.
  Also reported: the number of sites (of 3) with a positive difference, each site's bootstrap
  95% interval, and every earlier version (v0 ... v_{k-1}) on every later site.
- **H2 (correction effort falls).** The QC correction rate, i.e. (boxes added + removed +
  relabelled) per image over all photographs of the site, against the pre-labels the PWA shows
  (conf >= 0.25), is lower at Johor (v3 pre-labels) than at Pahang (v0 pre-labels).
  Computed by script (`dl.py qc`) from the raw export and the final labels, one-to-one greedy
  matching by descending IoU: a final box matched to a pre-label of the same class at IoU >= 0.5
  is unchanged; among the rest, a match to a different class at IoU >= 0.5 is relabelled;
  unmatched final boxes are added; unmatched pre-labels are removed.
  Also reported: all four sites in order, the core-class-only rate, and QC hours per site.
- **H3 (new-farm diversity, not quantity; test of Ruigrok et al. 2023, CEA 204:107554).**
  For each update step k = 1, 2, 3: a swap model S_k is trained on v_{k-1}'s training set with
  n_k randomly chosen images removed and the n_k images of the newly visited site added
  (same total size as v_{k-1}'s training set). Removed images are drawn by `group`
  (tree / burst, from the manifest) from the merged_v5 training split only, seeded.
  S_k and v_{k-1}'s recipe are each trained with seeds 0, 1, 2; the deployed v_{k-1} counts as
  the seed-0 run of its recipe. Removal uses a fixed seed (0) for all three training seeds.
  Prediction: the 3-seed mean mAP50-core of S_k exceeds that of the v_{k-1} recipe on the next
  site's test set, at the majority of steps (at least 2 of 3; step 3 uses Johor).
- **H4 (pre-labels anchor the annotator).** On the from-scratch subset (at least 40 images
  per site; the same random subset used for the agreement check of the CEA prospective test,
  drawn from the odd-numbered trees after the visit, so not known at capture time), the share of final
  boxes that match a pre-label box (IoU >= 0.5, any class) is higher in the pre-label-corrected
  annotation than in the from-scratch annotation of the same images (pooled over sites).
  Order fixed: the from-scratch annotation is done first, with pre-labels hidden; the
  pre-label correction of the same images follows at least 3 days later.
  Known limitations, stated now: the same person (the author) does both annotations, and the
  PWA displays its output at capture time; both bias towards agreement between the two
  annotations, i.e. against H4.

## 7. Known limitations, stated now

- Part of merged_v5 was annotated class by class (merge report: of 1,752 images with known
  history, 1,172 were checked for one class only), whereas new-site images are fully annotated.
  Part of any gain may therefore come from label completeness rather than new farms.
- Each deployed version is a single training run (seed 0); seed variation is estimated only
  through the H3 runs.

## 8. Not yet pre-registered

- An offline replication on GWHD 2021 (sequential addition of acquisition sessions, swap
  control) is planned. Its plan will be committed before any model is trained or scored on
  GWHD; as of this prelock no GWHD data have been analysed.
