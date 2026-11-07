# Shadow models (D6: exploratory 640 vs 1280)

Each shadow model is trained on exactly the deployed version's training set with the deployed version's recipe
(Ultralytics 8.4.182, YOLO11s from COCO-pretrained `yolo11s.pt`, 150 epochs, AdamW, lr0 0.001, seed 0, close_mosaic 10,
fixed merged_v5 validation split), changing only `imgsz` 640 -> 1280 and `batch` 16 -> 8. Shadow models are never
registered in `versions_log.md`, never deployed and never used for pre-labelling. They are scored with
`dl.py score --site <site> --extra <name>=<onnx> --pair <name>:<version>` on the same locked even-tree test set as
the version they shadow; results go to `results/<site>/scores.json` (`kind: extra`). See `deviations.md`, D6.

Hashes are recorded here before the site on which the shadow model is scored is visited.

| name | shadows | scored on | ONNX file | ONNX SHA-256 | best.pt SHA-256 | trained |
|---|---|---|---|---|---|---|
| v0_1280 | v0 | Pahang | durian_11cls_1280_v0_1280_20261102_76734ad1.onnx | 76734ad115a94d891ed7ec45cbda7eaa5731921dd2e8fdc4c540f204c0aaa8df | 91687ba9c199a8de1735adaa56726b9d4778b8fee98ad5992fcbbbe84328dcc9 | 2026-11-02, A100, 1.28 h |
| v1_1280 | v1 | Penang | durian_11cls_1280_v1_1280_20261102_3805d2c6.onnx | 3805d2c62e704bf97f753d7d31e440be7aadc7c315cd9dbc169384b2c251eb9d | dd0ca22b1eb6f93e2104424126c4904da8e38bf24cdfe9226adab0e2bc99228d | 2026-11-02, A100, 1.61 h |

Results so far: Pahang, v0_1280 - v0 = +0.0366 mAP50-core, 95% tree-bootstrap CI [-0.0186, +0.0648]; on the
smaller-lesion half of the test images +0.0487 [+0.0173, +0.0789], on the larger-lesion half +0.0116 [-0.1760, +0.0926]
(`results/pahang/scores.json`, `exploratory_size_split`).
Penang, v1_1280 - v1 = +0.0212 [-0.2347, +0.0392] (4 test trees, core classes without leafhopper_damage);
smaller-lesion half +0.0181 [-0.1912, +0.0409], larger-lesion half -0.0070 [-0.0922, +0.0290]
(`results/penang/scores.json`). The decision rule is applied once, pooled over all sites, after the last site.
