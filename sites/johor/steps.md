# Johor（第 5 站，部署 v4，文件名前缀 `jh1`；D7，暂定）

> 顺序和版本是暂定的（D7）：去之前先在 `deviations.md` 记一条确认（D1 原来写的“Johor 用 v2”已经不适用）。如果是最后一个新农场，H2 就拿这一站和 Pahang 比。

所有命令都在仓库根目录跑。`data/` 和 `models/` 不进 Git。

## 出发前
- [ ] `python dl.py status`：确认 v4 已经登记（versions_log.md 里有哈希，而且已经 push）。没有的话先 `python dl.py register --version v4 --onnx <文件>`，commit + push。
- [ ] 服务器上 `admin.py models`：标着“正在使用”的是 v4。
- [ ] 1280 影子 `v4_1280` 的哈希已经写进 `sequential/shadow_models.md` 并 push（D6）。
- [ ] 把 `trees_template.csv` 复制成 `trees.csv`，现场一边走一边填每棵树的开始、结束时间（手机看时间就行）。

## 现场
- [ ] 先确认是语料里没有的新农场。每棵树第一张拍树干或树牌当分隔；按走的顺序编号 t1, t2, …；双数号树是这一篇的测试集；单数号树给 CEA（CEA 的预锁定本来就包括这一站）；全部树都进下一版的训练。
- [ ] 15 棵以上、200 张封顶、每张拍不一样的；比例照 Pahang；拍照时别看 App 的结果决定拍什么。
- [ ] 全部用 PWA 拍；位置共享打开。
- [ ] **全部树拍完以后：换手机对照（偏差 D2）**：回头挑 2–3 棵拍过的树，每棵 iPhone 拍 10 张、小米拍 10 张（有症状的叶子）。
      iPhone 用主邀请码，小米用**另一个邀请码**，都用 PWA。这一段**不要写进 trees.csv**。

## 回来
1. 服务器上导出这一站的照片（只导出这一天、你自己的邀请码）：
   `python admin.py export johor_export.zip --since <访问日期> --code <你的邀请码>`
   下载到 `data/sites/johor/export.zip`。
2. `python dl.py rename --site johor` → 文件名变成 `jh1t3_…`，原始预标另存，报告写进 `rename_report.txt`。
   **先确认所有照片都已上传、而且都是 v4 预标的**（报告里会写），之后才能在服务器上换下一个模型。
3. （H4 已取消，D4：不抽 40 张、不从零标。）
4. （同上。）
5. 把 `renamed/images` + `renamed/prelabels` 作为预标导入标注工具，逐框 QC 全部照片，规矩照 D5（只标有把握的；默认整图大小；分不清 Phomopsis / leaf_rot 就标 leaf_spot_blight；没病斑点 Null）。
   导出（Roboflow YOLO 格式）到 `data/sites/johor/qc_export.zip`，然后 `python dl.py merge-qc --site johor --export data/sites/johor/qc_export.zip --missing-as-empty --cea-drop-unresolved`，再 `python dl.py export-cea --site johor`，把 `data/sites/johor/johor_cea_odd.zip` 交给 CEA 那边（单数号树、拆成 Phomopsis / Leaf_rot；后面的 CEA 步骤在另一个仓库做）。
6. `python dl.py split --site johor --missing-as-empty` → 双数号树测试集 + 全部树的训练集，报告写进 `split_report.txt`。
7. `python dl.py lock --site johor --visit-date <YYYY-MM-DD> --district "<县/区>"`（知道 QC 花了几小时就加 `--qc-hours <小时>`）
   → **commit `lock.json`，把打印出来的 SHA-256 发邮件给自己。**
8. 打分（CPU 就行；会验证锁定和每个模型的哈希），带上 1280 影子：
   `python dl.py score --site johor --extra v4_1280=<onnx> --pair v4_1280:v4`
   （H3 只有三步，到 Malacca 为止；这一站不用 S/R 模型。）
9. `python dl.py qc --site johor`（H2）和 `python dl.py farmer --site johor`。
10. commit `results/johor/`。

## 最后一站之后
- `python dl.py summary` → `results/summary.md`（H1–H3 按预锁定和偏差 D7 的判定标准；H4 已取消）。
- 1280 的决定（D6）：把各站影子模型的配对差合起来看，再量一下服务器 1280 的推理时间。
- 如果还要回访旧农场（Pahang 或 Seremban），去之前先记一条偏差，那一站只算探索性。
