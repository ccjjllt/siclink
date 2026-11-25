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

2026-11-24 14:12（北京时间）以 `ccjjllt` 提交，编号 `58814920`。Kaggle 已完成评分，公开 OCF 为 **15.04410**。 [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第一轮改进

固定种子 211。Task 1 在两个 session 中比较 320 个未见字的留出划分，正式任务的 20 次重复在这里用 4 次近似。Task 2 留出 `session4`，Task 3 按同鼠同 session 同字留出一次重复。下列指标均是 Kaggle 训练集留出结果。

| 实验 | 结果 | 样本 |
| --- | --- | --- |
| 原始特征与诱发窗口，Kaggle CPU | 保留 50–250 ms 总计数后，两只鼠的 Task 2 OCF 从 14.43/13.43 增至 21.59/18.09，Task 3 从 12.94/11.67 增至 16.10/15.57 | 每鼠每任务每视图 32 张 OCF，识别准确率使用完整留出集合 |
| 未见字核回归和残差 CNN，Kaggle GPU | 同批样本上，原始线性回归平均 10.87，诱发窗口核回归加自适应二值化为 12.35，CNN 二值输出 14.44，灰度输出 4.87 | 每鼠每 session 每版本 48 张 OCF，两个 session |
| 已见字多模型检索，Kaggle 双 T4 | Task 2 融合后一对一匹配 OCF 22.04/17.84，Top-1 为 8.15%/3.64%。Task 3 融合直接检索 OCF 16.61/14.38，Top-1 为 0.94%/0.51% | 每鼠每任务每版本 48 张 OCF，识别准确率使用完整留出集合 |
| 学习 10 ms 时间滤波，Kaggle GPU | Task 2 OCF 18.54/14.08，Task 3 13.53/13.47。当前对照中保留 50–250 ms 总计数更优 | 同 CPU 窗口诊断的 OCF 样本 |
| 感受野正向编码，Kaggle GPU | 18 个 epoch 后，各块的留出 spike 偏差解释率为 2.77%–4.47%。在 320 张已知候选图里，Poisson 匹配 Top-1 为 10.31%–34.38%。反演的二值 OCF 均值为 11.26，同批核回归对照为 11.83 | 每块 320 个留出字，反演每鼠每 session 32 张 OCF |
| Task 1 诱发窗与轮廓细化，Kaggle GPU | 50–250 ms 总计数组合的二值输出四块 OCF 为 17.78/16.66/14.07/14.71，均值 15.81。多尺度输入同批均值 15.44 | 每鼠每 session 每版本 64 张 OCF，两个 session |
| Task 2/3 直接重建，Kaggle 双 T4 | 轮廓细化优于直接回归，Task 2 OCF 为 13.92/12.93，Task 3 为 13.53/12.17。同批次的融合检索更高 | 每鼠每任务每版本 48 张 OCF |
| 已见字完整字图库 Poisson 检索，Kaggle GPU | Task 2 的 mice1 混合实测与编码放电再一对一匹配，OCF 为 34.25、Top-1 为 12.01%；mice2 此方法 OCF 为 15.87，融合检索仍为 17.84。Task 3 用正向编码和 25 ms 诱发窗计数匹配，两只鼠 OCF 为 17.43/17.06 | 与融合检索相同的 48 张配对 OCF 样本；Top-1 用完整留出块 |

检索的一对一匹配只使用各样本基于 spike 的字形分数，以及正式数据说明给出的每 session 每个已见字出现一次的设计。样本 ID 和文件顺序用于封包。Task 3 在留出一次重复后，分类准确率仍低，因此继续比较直接重建和对轮廓的优化。[赛题数据](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data)

远端运行与指标可在私有 Notebook 的 Output 查看。[时间窗](https://www.kaggle.com/code/ccjjllt/siclink-view-diagnostics) · [未见字](https://www.kaggle.com/code/ccjjllt/siclink-unseen-validation) · [已见字检索](https://www.kaggle.com/code/ccjjllt/siclink-retrieval-validation) · [10 ms 滤波](https://www.kaggle.com/code/ccjjllt/siclink-temporal-validation) · [轮廓细化](https://www.kaggle.com/code/ccjjllt/siclink-unseen-contour-validation) · [直接重建](https://www.kaggle.com/code/ccjjllt/siclink-direct-validation)

正向编码的 320 候选识别是使用留出训练图像的诊断。Task 2/3 的完整已见字图库对照使用 `kaggle_encoding_validation.py --mode seen`。候选版以 Task 1 诱发总计数加二值残差细化、Task 2 分鼠选对照胜出方案、Task 3 正向编码 Poisson 匹配作为初始配置。还需正式评分确认收益。 [正向编码验证](https://www.kaggle.com/code/ccjjllt/siclink-encoding-validation) · [完整候选图库对照](https://www.kaggle.com/code/ccjjllt/siclink-seen-encoding-validation)

Task 1 的条件流匹配已接入 `kaggle_reconstruction_validation.py --views evoked-multiscale --flow-epochs 12`。它用训练字的交叉拟合粗图作空间条件，在与残差 CNN 完全相同的留出字上比较锚点、共识和代表候选，并记录每张配对 OCF 的平均差值。当前只有验证入口，效果待 Kaggle 实测。 [Flow Matching](https://arxiv.org/abs/2210.02747) · [Sensorium-Viz](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202520220)

### 研究依据

- Yoshida 和 Ohki 从小鼠 V1 数据观察到，少数可靠单元能支撑自然图像的线性解码。这里用跨 session 同字重复估计稳定方向，并让试验检验更细的 spike 时序是否有增益。[论文](https://www.nature.com/articles/s41467-020-14645-x)
- 从 spike 回归粗图后做图像模型细化，采用类似 SID 的两阶段设计。细化器的输入由按字交叉拟合生成，以估计未见字上的泛化。Moran 的公开 Kaggle Notebook 也披露了多视图残差和最小 Bayes 风险思路，可用于设计对照。[SID](https://arxiv.org/abs/1904.13007) · [Moran Notebook](https://www.kaggle.com/code/moranzavdi/siclink-ocf-audit-v1-drift-14-point-verifier)
- 正向编码从多尺度方向特征、神经元空间感受野和 session 校准预测 spike 计数，再从真实响应似然反演像素。其有效性用留出字的响应预测和官方 OCF 分别衡量。已发表解码工作使用的神经元数或刺激域与本赛题有差异，效果需要在本赛题验证。[视觉皮层电影重建](https://elifesciences.org/articles/105081) · [MEIcoder 论文](https://arxiv.org/abs/2510.20762) · [MEIcoder 代码](https://github.com/Johnny1188/meicoder)
