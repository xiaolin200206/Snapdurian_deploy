# Penang（第 2 站，部署 v1，文件名前缀 `pg1`）

所有命令都在仓库根目录跑。`data/` 和 `models/` 不进 Git。

## 出发前
- [ ] `python dl.py status`：确认 v1 已经登记（versions_log.md 里有哈希，而且已经 push）。没有的话先 `python dl.py register --version v1 --onnx <文件>`，commit + push。
- [ ] 服务器上 `admin.py models`：标着“正在使用”的是 v1。
- [ ] 把 `trees_template.csv` 复制成 `trees.csv`，现场一边走一边填每棵树的开始、结束时间（手机看时间就行）。

## 现场
- [ ] 每棵树第一张拍树干或树牌当分隔；按走的顺序编号 t1, t2, …；单数号树给 CEA，双数号树给这一篇。
- [ ] 全部用 PWA 拍；位置共享打开。
- [ ] **全部树拍完以后：换手机对照（偏差 D2）**：回头挑 2–3 棵拍过的树，每棵 iPhone 拍 10 张、小米拍 10 张（有症状的叶子）。
      iPhone 用主邀请码，小米用**另一个邀请码**，都用 PWA。这一段**不要写进 trees.csv**。

## 回来
1. 服务器上导出这一站的照片（只导出这一天、你自己的邀请码）：
   `python admin.py export penang_export.zip --since <访问日期> --code <你的邀请码>`
   下载到 `data/sites/penang/export.zip`。
2. `python dl.py rename --site penang` → 文件名变成 `pg1t3_…`，原始预标另存，报告写进 `rename_report.txt`。
   **先确认所有照片都已上传、而且都是 v1 预标的**（报告里会写），之后才能在服务器上换下一个模型。
3. `python dl.py pick-scratch --site penang` → 抽 40 张单数号树的图，**列表马上 commit**。CEA 那边的从零标用同一批。
4. 把 `data/sites/penang/scratch_upload/` 里的图**不带预标**上传到标注工具，从零标完，导出（YOLO 检测框 + data.yaml）到 `data/sites/penang/scratch/`。
   `python dl.py lock-scratch --site penang` → **commit `scratch_lock.json`**。
5. 至少隔 3 天：把 `renamed/images` + `renamed/prelabels` 作为预标导入标注工具，逐框 QC 全部照片（从零标的那 40 张放到最后改）。
   规则见 `sequential/prelock_note.md` 第 4 节：`leaf_spot_blight` 不拆；叶面正面的白色蜡虫标 `psyllid`。
   导出 11 类终标到 `data/sites/penang/final/`。CEA 那份（单数号树、拆成 Phomopsis / Leaf_rot）在另一个仓库照 CEA 的流程做。
6. `python dl.py split --site penang` → 双数号树测试集 + 全部树的训练集，报告写进 `split_report.txt`。
7. `python dl.py lock --site penang --qc-hours <小时> --visit-date <YYYY-MM-DD> --district "<县/区>"`
   → **commit `lock.json`，把打印出来的 SHA-256 发邮件给自己。**
8. `python dl.py score --site penang`（CPU 就行；会验证锁定和每个模型的哈希）。
   H3（第 1 步）：把等量替换模型和 v0 配方的另外两个种子一起打分：
   `python dl.py score --site penang --extra S1_s0=<onnx> S1_s1=<onnx> S1_s2=<onnx> R0_s1=<onnx> R0_s2=<onnx>`
   （v0 本身就是 R0 的 s0，已经在登记的版本里，不用再给。）
9. `python dl.py qc --site penang`（H2）和 `python dl.py anchor --site penang`（H4）。
10. commit `results/penang/`。

## 训练下一版（v2）
- `python dl.py build-train --name v2 --merged data/merged_v5 --sites pahang penang`
- 在 Colab 用 `notebooks/train_version_colab.ipynb` 训练并导出 ONNX。
- `python dl.py register --version v2 --onnx <导出的 best.onnx> --server-file <服务器上的文件名>`
- **commit + push `sequential/versions_log.md` 之后**，才去下一站；服务器上 `admin.py use` 这个文件。
- H3（可以之后再做）：等量替换 S2，三个种子（把 `_s0` 和 `--seed 0` 换成 1、2 再跑两次）：
  `python dl.py build-swap --name S2_s0 --merged data/merged_v5 --manifest data/durian_11cls_640_v5_manifest.csv --base-sites pahang --new-site penang --seed 0`
  v1 配方的另外两个种子：`python dl.py build-train --name R1_s1 --merged data/merged_v5 --sites pahang --seed 1`（s2 同理）。
  这些在下一站打分时用 `--extra` 一起算。
