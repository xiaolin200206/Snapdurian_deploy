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
