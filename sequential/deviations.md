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
