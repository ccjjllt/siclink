# 实验计划

## Kaggle 起步

1. 按鼠和 session 统计总 spike 数、每个 unit 的活动和刺激后群体时间曲线（PSTH）。检查 session 漂移，比较全窗计数与 25 ms 分窗计数。按 session 读取原始事件，缓存小特征。
2. 每只鼠先做单独的 ridge 像素回归。对于已见字的 Task 2/3，另测训练字图库上的近邻模板检索，记录 top-1 字符准确率和 OCF。多 trial 样本先用各 trial 特征的平均作基线。
3. 在相同验证划分上比较灰度输出与二值输出。先在固定抽样上跑 CPU OCF，同时查看按任务、鼠和 session 分组的预测样图。

## 用训练数据模拟官方任务

| 任务 | 本地划分 | 与正式评估的差异 |
| --- | --- | --- |
| Task 1 | 在两只鼠的所有 session 中留出同一批完整字符，按鼠、session、字符聚合 | 训练数据每个组合只有 4 次重复，正式评估有 20 次 |
| Task 2 | 每只鼠留出整个训练 session，在留出的 session 内按字聚合 4 次重复 | 训练 session 留出是跨天漂移的代理实验，正式样本来自 `session5` |
| Task 3 | 每个鼠、session、字符组合留出一次 trial | 本地可训练重复数是 3，正式训练有 4 次 |

固定随机种子并保存划分，特征标准化参数仅从拟合部分估计。每次记录特征、划分、模型、运行时间、三个 OCF 均值和对应样本数。

## 何时上 GPU

Kaggle CPU 足够做数据探索、线性模型、模板检索和抽样 OCF。全量 1 ms 稠密训练输入约 2.33 GB（`uint8`），转 `float32` 约 9.32 GB，建议按 session 读和分批处理。双 T4 用于并行训练与推理。推进到时序编码器、官方条件 DDPM 或全量 OCF 时，先用短跑测吞吐与显存，再决定规模。官方 DDPM 的 CUDA 路径使用 BF16 autocast，原样运行时优先使用支持 BF16 的卡。 [官方参赛者包](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data)

## 已运行

2026-11-24 在 CPU 上运行了数据浏览入口。`mice1/session1` 训练块含 10,992 个 trial，每字 4 次重复，134 个 unit 均有记录。Task 2 的 `mice2/session5` 评测块含 2,748 个样本，每样本 4 个 trial。官方 loader 也在 8 条训练样本的演示数据上成功返回了 spike 矩阵和标签图像。

### 提交 01

特征为 25 ms 窗与整窗 spike counts，逐 trial 做平方根变换和 session 标准化，再聚合重复。Task 1 混合同 session 和跨训练 session 的 ridge 像素预测。Task 2/3 用带协方差正则的训练字形模板检索。Task 2 的新 session 用当前块的无标签信号估计标准化参数。样本 ID 用于封包。

双 T4 启动遇到 Kaggle 批处理并发上限，版本 3 改用 Kaggle CPU，在日志约 35 秒时生成全部 30,200 个预测。提交包按官方样例顺序整理，PNG 内容保持原样。最新脚本直接按该顺序输出。 [Kaggle Notebook](https://www.kaggle.com/code/ccjjllt/siclink-spike-count-baseline)

2026-11-24 14:12（北京时间）以 `ccjjllt` 提交，编号 `58814920`。本次核对时 Kaggle 状态为 `Pending`。 [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)
