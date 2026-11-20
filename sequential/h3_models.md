# H3 models (prelock section 6, H3; steps at Penang, Klang, Malacca per D7)

For each update step k, a swap model S_k is trained on v_{k-1}'s training set with n_k images removed from the
merged_v5 training split (by group, from `durian_11cls_640_v5_manifest.csv`, removal seed 0) and the n_k images of
the newly visited site added; the recipe of v_{k-1} is trained again with seeds 1 and 2 (R_{k-1}_s1, R_{k-1}_s2; the
deployed v_{k-1} is its seed-0 run). All runs use the prelock recipe unchanged (Ultralytics 8.4.182, YOLO11s from
COCO-pretrained `yolo11s.pt`, 150 epochs, batch 16, imgsz 640, AdamW, lr0 0.001, close_mosaic 10, fixed merged_v5
validation split); only the training images and the seed differ. Built with `dl.py build-swap` / `dl.py build-train`,
trained with `notebooks/train_queue_colab.ipynb`. H3 models are never registered, deployed or used for pre-labelling.

| step | scored on | models | training data |
|---|---|---|---|
| 1 | Penang (even trees, lock `5add0800…`) | S1_s0, S1_s1, S1_s2, R0_s1, R0_s2 | merged_v5 (+ Pahang for S1) |
| 2 | Klang | S2_s0, S2_s1, S2_s2, R1_s1, R1_s2 | merged_v5 + Pahang (+ Penang for S2) |
| 3 | Malacca | S3_s0, S3_s1, S3_s2, R2_s1, R2_s2 | merged_v5 + Pahang + Penang (+ Klang for S3) |

Hashes are recorded here and pushed before the models are scored (see D9 for step 1). Scored with
`dl.py score --site <site> --extra S<k>_s0=<onnx> … R<k-1>_s2=<onnx>`; results in `results/<site>/scores.json`.

| name | ONNX file | ONNX SHA-256 | best.pt SHA-256 | trained | training set |
|---|---|---|---|---|---|
| S1_s0 | durian_11cls_640_S1_s0_20261108_3fea8c8e.onnx | 3fea8c8ecdcc1ed97c4e86aa5008fef35148ecb979f3225523e9d543b3894f5d | 3f242342ae1bea5f47ce26f693ded4db5c4217ec05cee937933193c52f6a5de8 | 2026-11-08, L4, 0.65 h | 1589 images: merged_v5 train − 320 (by group, removal seed 0) + Pahang 320 |
| S1_s1 | durian_11cls_640_S1_s1_20261108_682ebece.onnx | 682ebece64aa18e7abcdd64f1c73f6297aca4104f22c66b8af254b2be3215687 | 7e203cce86ac67e4011263a09f5e84ffca1b503a5028ce5f8d3f63acfabf29c3 | 2026-11-08, L4, 0.65 h | 1589 images: merged_v5 train − 320 (by group, removal seed 0) + Pahang 320 |
| S1_s2 | durian_11cls_640_S1_s2_20261108_2a2c8fb6.onnx | 2a2c8fb63345f8a7cba1948af5a123fe378550ab8d45b8c0174b097d8b4eb780 | 629b0ed07ca04baa62526ac635410b0344261a35e2f33017bcb84b972960104e | 2026-11-08, L4, 0.64 h | 1589 images: merged_v5 train − 320 (by group, removal seed 0) + Pahang 320 |
| R0_s1 | durian_11cls_640_R0_s1_20261108_7e4067ef.onnx | 7e4067ef1aebc77e1e6233bc6fe8450385c7030b2ccd464359e9fb6418bbf3f8 | 6cd25625b58a2b309870a52e46c56fdd4593c8d94adc476bacae5fe12f23dec5 | 2026-11-08, L4, 0.65 h | 1589 images: merged_v5 train (v0 recipe) |
| R0_s2 | durian_11cls_640_R0_s2_20261108_9251b501.onnx | 9251b501542e0e9b32dc55df858fdc00413862a1589d83967ddf0669f716e2e1 | 4474d444a2e19bf6cb1cb25aa9523bf45a604608984decb75adf8e3216256c35 | 2026-11-08, L4, 0.65 h | 1589 images: merged_v5 train (v0 recipe) |
| S2_s0 | durian_11cls_640_S2_s0_20261108_0fbe484a.onnx | 0fbe484ae78332edc14c794d4ba6422edf9f62e4f6aeeab66b1fbbae5b7570ed | b1334c48e35733f64023476547a4116bc13ac9785169262ebcadce78e6c9ac59 | 2026-11-08, L4, 0.75 h | 1909 images: merged_v5 train − 109 (by group, removal seed 0) + Pahang 320 + Penang 109 |
| S2_s1 | durian_11cls_640_S2_s1_20261108_f0afd58e.onnx | f0afd58e88504644805393e0a7f5aa4480740f9694c3ea9680ae7799b05619fd | 36a2a258578045cb78ecf267618a871fe65333097ce254bf01eea0b647bd090e | 2026-11-08, L4, 0.75 h | 1909 images: merged_v5 train − 109 (by group, removal seed 0) + Pahang 320 + Penang 109 |
| S2_s2 | durian_11cls_640_S2_s2_20261108_212faf1b.onnx | 212faf1b7d252bbf8d83dbdd85025541e54139ba169ed0998a195584532ff890 | a4b524f60c2900c76caa41b01564916f19f66edef72e63fb12624e8f1cd28740 | 2026-11-08, L4, 0.75 h | 1909 images: merged_v5 train − 109 (by group, removal seed 0) + Pahang 320 + Penang 109 |
| R1_s1 | durian_11cls_640_R1_s1_20261108_911dfe37.onnx | 911dfe371f6907f932425ae09153631ab468f41f15a5797741aa6560c2166994 | 8d38fab7982f9262c9f38cc55a8bce603a497df09e06933cf44c00602243476e | 2026-11-08, L4, 0.74 h | 1909 images: merged_v5 train + Pahang 320 (v1 recipe) |
| R1_s2 | durian_11cls_640_R1_s2_20261108_dda0f659.onnx | dda0f65927fb2e272a41be7c5e6d91b9123a399919894ff6519d4c450adbe0a1 | abf3dee90c342d6fc7e9c373dbd46431e23852dcb9b40cab36aed1fe9fc97b0e | 2026-11-08, L4, 0.75 h | 1909 images: merged_v5 train + Pahang 320 (v1 recipe) |
| S3_s0 | durian_11cls_640_S3_s0_20261120_842f2b10.onnx | 842f2b109542e8c147d8bfdfb73d4544430040dbca48ee3bbe1b880a1c3851f6 | e0802871ad19540e8d0c441712054a1d1c5a52635a2be88a64e231e9308da09e | 2026-11-20, L4, 0.79 h | 2018 images: merged_v5 train − 191 (by group, removal seed 0) + Pahang 320 + Penang 109 + Johor 191 |
| S3_s1 | durian_11cls_640_S3_s1_20261120_519fd7e8.onnx | 519fd7e8b0435dafbc9cdfe93976232343d69582fe90ca0c5e493dc646a71322 | 1ff4dcdb11ae7a200419a9db08a882e4d61fc606c9d2ff5cb2e13bbce81bf0f1 | 2026-11-20, L4, 0.79 h | 2018 images: merged_v5 train − 191 (by group, removal seed 0) + Pahang 320 + Penang 109 + Johor 191 |
| S3_s2 | durian_11cls_640_S3_s2_20261120_d02502f3.onnx | d02502f3f8f324173cef7cd34d9a0ce01aa58c4a8e1f16ed1a71c4c500d4d9e4 | a5032243c25e979938ac36e6b1e7380edc14addf1888d7ae1389505d2a575b81 | 2026-11-20, L4, 0.8 h | 2018 images: merged_v5 train − 191 (by group, removal seed 0) + Pahang 320 + Penang 109 + Johor 191 |
| R2_s1 | durian_11cls_640_R2_s1_20261120_85604f42.onnx | 85604f42715d5a0eac4f320183aa58375c87c5faa456b6e48a401b02a8808dca | 5781070bb63452f324338a9e89b59919aaa1b3b87ca5fd9058e2177689b463d5 | 2026-11-20, L4, 0.8 h | 2018 images: merged_v5 train + Pahang 320 + Penang 109 (v2 recipe) |
| R2_s2 | durian_11cls_640_R2_s2_20261120_d54733f5.onnx | d54733f5130d65525ce40097c09c3699ca80a6b3d14441a0ed9ae074a3c44e2d | 7ccb9d17e80dcd8374c9fdf111450fde85421c23add7d6d520fbe1d3695e974e | 2026-11-20, L4, 0.79 h | 2018 images: merged_v5 train + Pahang 320 + Penang 109 (v2 recipe) |

Note (recorded 2026-11-08, before any S2 model is scored): two Colab sessions of the training queue overlapped, so
S2_s0 was trained twice with the same data, recipe and seed: `durian_11cls_640_S2_s0_20261108_d477e088.onnx`
(finished 14:57 MYT) and `durian_11cls_640_S2_s0_20261108_0fbe484a.onnx` (finished 15:20 MYT). The run in the table
(0fbe484a) is the one whose `S2_s0_info.json` was written last; it was committed in a898c41, before Klang was visited
and before either run was scored. The d477e088 run is not scored or used anywhere; its file is kept, unregistered.

Note (D13, recorded before the visit): Klang was cancelled; step 2 (S2_s0–s2, R1_s1–s2) is scored at Johor (site 3) and step 3 (S3, R2; trained with Johor as the new site) at Malacca (site 4).
