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
