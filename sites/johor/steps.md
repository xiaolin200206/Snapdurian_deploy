# Johor（第 4 站，部署 v3，文件名前缀 `jh1`）

所有命令都在仓库根目录跑。`data/` 和 `models/` 不进 Git。

## 出发前
- [ ] `python dl.py status`：确认 v3 已经登记（versions_log.md 里有哈希，而且已经 push）。没有的话先 `python dl.py register --version v3 --onnx <文件>`，commit + push。
- [ ] 服务器上 `admin.py models`：标着“正在使用”的是 v3。
- [ ] 把 `trees_template.csv` 复制成 `trees.csv`，现场一边走一边填每棵树的开始、结束时间（手机看时间就行）。

## 现场
- [ ] 每棵树第一张拍树干或树牌当分隔；按走的顺序编号 t1, t2, …；单数号树给 CEA，双数号树给这一篇。
- [ ] 全部用 PWA 拍；位置共享打开。

## 回来
1. 服务器上导出这一站的照片（只导出这一天、你自己的邀请码）：
   `python admin.py export johor_export.zip --since <访问日期> --code <你的邀请码>`
   下载到 `data/sites/johor/export.zip`。
2. `python dl.py rename --site johor` → 文件名变成 `jh1t3_…`，原始预标另存，报告写进 `rename_report.txt`。
   **先确认所有照片都已上传、而且都是 v3 预标的**（报告里会写），之后才能在服务器上换下一个模型。
3. `python dl.py pick-scratch --site johor` → 抽 40 张单数号树的图，**列表马上 commit**。CEA 那边的从零标用同一批。
4. 把 `data/sites/johor/scratch_upload/` 里的图**不带预标**上传到标注工具，从零标完，导出（YOLO 检测框 + data.yaml）到 `data/sites/johor/scratch/`。
   `python dl.py lock-scratch --site johor` → **commit `scratch_lock.json`**。
5. 至少隔 3 天：把 `renamed/images` + `renamed/prelabels` 作为预标导入标注工具，逐框 QC 全部照片（从零标的那 40 张放到最后改）。
   规则见 `sequential/prelock_note.md` 第 4 节：`leaf_spot_blight` 不拆；叶面正面的白色蜡虫标 `psyllid`。
   导出 11 类终标到 `data/sites/johor/final/`。CEA 那份（单数号树、拆成 Phomopsis / Leaf_rot）在另一个仓库照 CEA 的流程做。
6. `python dl.py split --site johor` → 双数号树测试集 + 全部树的训练集，报告写进 `split_report.txt`。
7. `python dl.py lock --site johor --qc-hours <小时> --visit-date <YYYY-MM-DD> --district "<县/区>"`
   → **commit `lock.json`，把打印出来的 SHA-256 发邮件给自己。**
8. `python dl.py score --site johor`（CPU 就行；会验证锁定和每个模型的哈希）。
   H3（第 3 步）：把等量替换模型和 v2 配方的另外两个种子一起打分：
   `python dl.py score --site johor --extra S3_s0=<onnx> S3_s1=<onnx> S3_s2=<onnx> R2_s1=<onnx> R2_s2=<onnx>`
   （v2 本身就是 R2 的 s0，已经在登记的版本里，不用再给。）
9. `python dl.py qc --site johor`（H2）和 `python dl.py anchor --site johor`（H4）。
10. commit `results/johor/`。

## 最后一站之后
- `python dl.py summary` → `results/summary.md`（H1–H4 按预锁定的判定标准）。
