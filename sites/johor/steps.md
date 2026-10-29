# Johor（第 4 站，预标用 v2（偏差 D1），H1 比较 v3 和 v2，文件名前缀 `jh1`）

所有命令都在仓库根目录跑。`data/` 和 `models/` 不进 Git。

## 出发前
- [ ] 偏差 D1：和 Malacca 同一趟，前一天去 Malacca。服务器上保持 v2 不动（`admin.py models` 标着“正在使用”的是 v2）。
- [ ] `sequential/deviations.md` 已经 commit + push（在去 Malacca 之前）。
- [ ] 把 `trees_template.csv` 复制成 `trees.csv`，现场一边走一边填每棵树的开始、结束时间（手机看时间就行）。

## 现场
- [ ] 每棵树第一张拍树干或树牌当分隔；按走的顺序编号 t1, t2, …；单数号树给 CEA，双数号树给这一篇。
- [ ] 全部用 PWA 拍；位置共享打开。
- [ ] **全部树拍完以后：换手机对照（偏差 D2）**：回头挑 2–3 棵拍过的树，每棵 iPhone 拍 10 张、小米拍 10 张（有症状的叶子）。
      iPhone 用主邀请码，小米用**另一个邀请码**，都用 PWA。这一段**不要写进 trees.csv**。

## 回来
1. 服务器上导出这一站的照片（只导出这一天、你自己的邀请码）：
   `python admin.py export johor_export.zip --since <访问日期> --code <你的邀请码>`
   下载到 `data/sites/johor/export.zip`。
2. `python dl.py rename --site johor` → 文件名变成 `jh1t3_…`，原始预标另存，报告写进 `rename_report.txt`。
   **先确认所有照片都已上传、而且都是 v2 预标的**（报告里会写），之后才能在服务器上换下一个模型。
3. `python dl.py pick-scratch --site johor` → 抽 40 张单数号树的图，**列表马上 commit**。CEA 那边的从零标用同一批。
4. 把 `data/sites/johor/scratch_upload/` 里的图**不带预标**上传到标注工具，从零标完，导出（YOLO 检测框 + data.yaml）到 `data/sites/johor/scratch/`。
   `python dl.py lock-scratch --site johor` → **commit `scratch_lock.json`**。
5. 至少隔 3 天：把 `renamed/images` + `renamed/prelabels` 作为预标导入标注工具，逐框 QC 全部照片（从零标的那 40 张放到最后改）。
   规则见 `sequential/prelock_note.md` 第 4 节：`leaf_spot_blight` 不拆；叶面正面的白色蜡虫标 `psyllid`。
   导出 11 类终标到 `data/sites/johor/final/`。CEA 那份（单数号树、拆成 Phomopsis / Leaf_rot）在另一个仓库照 CEA 的流程做。
6. `python dl.py split --site johor` → 双数号树测试集 + 全部树的训练集，报告写进 `split_report.txt`。
7. `python dl.py lock --site johor --qc-hours <小时> --visit-date <YYYY-MM-DD> --district "<县/区>"`
   → **commit `lock.json`，把打印出来的 SHA-256 发邮件给自己。**
8. 先把 v3 训好并 `register`（用 Malacca 锁定后的数据），再 `python dl.py score --site johor`（会算 v0–v3，H1 比较 v3 和 v2）。
   H3（第 3 步）：把等量替换模型和 v2 配方的另外两个种子一起打分：
   `python dl.py score --site johor --extra S3_s0=<onnx> S3_s1=<onnx> S3_s2=<onnx> R2_s1=<onnx> R2_s2=<onnx>`
   （v2 本身就是 R2 的 s0，已经在登记的版本里，不用再给。）
9. `python dl.py qc --site johor`（H2）和 `python dl.py anchor --site johor`（H4）。
10. commit `results/johor/`。

## 最后一站之后
- `python dl.py summary` → `results/summary.md`（H1–H4 按预锁定的判定标准）。
