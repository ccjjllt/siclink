# 实验计划

## Kaggle 起步

1. 按鼠和 session 统计总 spike 数、每个 unit 的活动和刺激后群体时间曲线（PSTH）。检查 session 漂移，比较全窗计数与 25 ms 分窗计数。按 session 读取原始事件，缓存小特征。
2. 每只鼠先做单独的 ridge 像素回归。对于已见字的 Task 2/3，另测训练字图库上的近邻模板检索，记录 top-1 字符准确率和 OCF。多 trial 样本先用各 trial 特征的平均作基线。
3. 在相同验证划分上比较灰度输出与二值输出。先在固定抽样上跑 CPU OCF，同时查看按任务、鼠和 session 分组的预测样图。

## 用训练数据模拟官方任务

| 任务 | Kaggle 验证划分 | 与正式评估的差异 |
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
| Task 1 条件流匹配，Kaggle GPU | 12 个 epoch 后，生成方案中代表候选最高，平均 OCF 9.69。同批残差 CNN 细化为 15.83 | 每鼠两个 session，每块 64 张成对 OCF |
| 已见字后验轮廓重排，Kaggle 双 T4 | 成对验证中，Task 2 的 Sinkhorn 轮廓重排为 17.22/18.00，融合检索加一对一匹配为 22.01/20.31。Task 3 的三个后验版本均未超过双鼠平均 15.69 的融合检索 | 每鼠每任务 128 张成对 OCF |
| 放电到轮廓签名回归，Kaggle 双 T4 | Task 3 线性签名检索为 18.60/18.54，同批融合检索为 16.87/14.50。与 Poisson 的现有 OCF 抽样大小不同，后续在同一批样本上配对比较 | 每鼠每任务 128 张成对 OCF，完整留出块计算 Top-1 |
| 单次放电 Poisson 与轮廓的配对消融，Kaggle GPU | 原时窗 Poisson 为 17.39/14.59。mice1 的分数融合达到 18.82，mice2 的纯轮廓达到 18.54。下个候选可通过 `--task3-decoder contour-validated` 使用本轮选择 | 同一留出重复、每鼠 128 张成对 OCF |
| 负二项放电似然，Kaggle GPU | Task 2 一对一匹配的成对变化为 +1.56/−0.91，标准误 1.84/0.89。Task 3 变化为 +0.51/+0.25，低于轮廓方案本轮同批对照的增益 | 每鼠每任务 128 张成对 OCF，离散度只拟合训练重复 |
| mice2 跨天放电适配，Kaggle GPU | 按均值适配的逐样本决策在 session2/4 分别改善 +0.80/+4.13 OCF。保持一对一匹配时，两个块上的最大适配差值仍为 −1.15/−0.23，候选继续使用融合检索 | 每个留出 session 128 张成对 OCF，完整块计算 Top-1 |
| 公开字体覆盖，Kaggle CPU | 从 27,550 个公开 Noto Sans 字形中，以留出字图像查询的首选平均 OCF 为 73.85，前四个的最高 OCF 均值为 75.01。Noto Serif 的前四上限为 43.46 | 只用 128 个拟合字估计全局变换，另抽 32 个留出字做形状覆盖诊断 |
| 公开 Noto Sans 字库神经排序，Kaggle GPU | 轮廓与 Poisson 分数融合的首位候选 OCF 为 **18.71**，同批残差 CNN 为 **15.82**。成对增益 **+2.89**，按字汇总的标准误 **0.55** | 两鼠两个 session，每块 64 张，共 256 张，64 个独立留出字 |
| 公开字库核正向编码，Kaggle GPU | 核编码融合排名 OCF 为 17.66。四块留出 spike 偏差解释率为 −14.89% 至 −8.73%，当前仍沿用可学习感受野编码 | 与公开字库神经排序相同的 256 张成对 OCF |
| 训练字字库风格对齐，Kaggle GPU | 仅将拟合字图像与公开渲染对齐后，融合排名 OCF 为 19.37。与原图编码的 18.71 相比，四块中三块提高、一块持平，正向偏差解释率略降 | 与公开字库神经排序相同的 256 张成对 OCF |

检索的一对一匹配只使用各样本基于 spike 的字形分数，以及正式数据说明给出的每 session 每个已见字出现一次的设计。样本 ID 和文件顺序用于封包。Task 3 在留出一次重复后，分类准确率仍低，因此继续比较直接重建和对轮廓的优化。[赛题数据](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data)

远端运行与指标可在私有 Notebook 的 Output 查看。[时间窗](https://www.kaggle.com/code/ccjjllt/siclink-view-diagnostics) · [未见字](https://www.kaggle.com/code/ccjjllt/siclink-unseen-validation) · [已见字检索](https://www.kaggle.com/code/ccjjllt/siclink-retrieval-validation) · [10 ms 滤波](https://www.kaggle.com/code/ccjjllt/siclink-temporal-validation) · [轮廓细化](https://www.kaggle.com/code/ccjjllt/siclink-unseen-contour-validation) · [直接重建](https://www.kaggle.com/code/ccjjllt/siclink-direct-validation)

正向编码的 320 候选识别是使用留出训练图像的诊断。Task 2/3 的完整已见字图库对照使用 `kaggle_encoding_validation.py --mode seen`。候选版以 Task 1 诱发总计数加二值残差细化、Task 2 分鼠选对照胜出方案、Task 3 正向编码 Poisson 匹配作为初始配置。还需正式评分确认收益。 [正向编码验证](https://www.kaggle.com/code/ccjjllt/siclink-encoding-validation) · [完整候选图库对照](https://www.kaggle.com/code/ccjjllt/siclink-seen-encoding-validation)

Task 1 条件流匹配比较了锚点、共识和代表候选。该配置下残差 CNN 仍领先，暂沿用残差 CNN。 [流匹配验证](https://www.kaggle.com/code/ccjjllt/siclink-flow-validation) · [Flow Matching](https://arxiv.org/abs/2210.02747) · [Sensorium-Viz](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202520220)

公开字库覆盖使用 SIL OFL 1.1 的 Noto Sans `Sans2.004`、Noto Serif `Serif2.003`，范围固定为 U+3400–4DBF、U+4E00–9FFF。75.01 来自已知参考图查询得到的候选覆盖。随后只用测得的 spike 和自动重建图排序，参考图只参与留出 OCF。正式候选采用验证胜出的 Noto Sans 融合排名第一字形。 [字体覆盖](https://www.kaggle.com/code/ccjjllt/siclink-font-oracle) · [神经排序](https://www.kaggle.com/code/ccjjllt/siclink-public-gallery-validation) · [Noto Sans](https://github.com/notofonts/noto-cjk/tree/Sans2.004) · [Noto Serif](https://github.com/notofonts/noto-cjk/tree/Serif2.003) · [SIL OFL](https://openfontlicense.org/ofl-faq/)

跨天适配消融用留出块的无标签重复估计各维放电均值和去噪方差，并与已有的 Poisson 原型在同一留出划分上比较。研究源自不同日之间的感觉表征漂移与神经活动子空间对齐。 [跨天消融](https://www.kaggle.com/code/ccjjllt/siclink-session-adaptation) · [小鼠视觉漂移](https://pubmed.ncbi.nlm.nih.gov/34433077/) · [低维空间对齐](https://www.nature.com/articles/s41551-020-0542-9)

备选升级同时使用 `--align-font-training` 和 `--task3-decoder contour-validated`。前者只改变 Task 1 的字库正向编码器，Task 2/3 的放电模板仍从原训练图像拟合。这个组合的正式得分还未验证。 [字库风格对齐](https://www.kaggle.com/code/ccjjllt/siclink-atlas-aligned-validation) · [Task 3 配对对照](https://www.kaggle.com/code/ccjjllt/siclink-contour-poisson)

### 提交 02

2026-11-25 11:18（北京时间）以 `ccjjllt` 提交编号 `58864024`。Kaggle 公开 OCF 为 **20.54607**，比提交 01 提高 **5.50197**。Task 1 用固定 Noto Sans 字库的轮廓与神经似然融合排序。Task 2 在 mice1 用 Poisson 收缩与一对一匹配，在 mice2 用融合检索与一对一匹配。Task 3 用正向编码的时窗 Poisson 似然。正式包由候选 Notebook 版本 3 在 Kaggle GPU 生成，含 30,200 行，SHA-256 为 `4b715b28a7c0b4c1661285ce9684c3b51f1213a1cf452a6b45aa09ff7028327c`。 [候选 Notebook](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v2) · [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

下一版在 Kaggle GPU 用约 173 秒生成。Task 1 使用训练字风格对齐，Task 3 分鼠使用轮廓与 Poisson 融合或纯轮廓匹配。已核对全部 30,200 个 ID、行顺序与 64×64 灰度 PNG。当天提交名额已用完，下一版尚未提交。 [下一版 Notebook](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v3)

### 研究依据

- Yoshida 和 Ohki 从小鼠 V1 数据观察到，少数可靠单元能支撑自然图像的线性解码。这里用跨 session 同字重复估计稳定方向，并让试验检验更细的 spike 时序是否有增益。[论文](https://www.nature.com/articles/s41467-020-14645-x)
- 从 spike 回归粗图后做图像模型细化，采用类似 SID 的两阶段设计。细化器的输入由按字交叉拟合生成，以估计未见字上的泛化。Moran 的公开 Kaggle Notebook 也披露了多视图残差和最小 Bayes 风险思路，可用于设计对照。[SID](https://arxiv.org/abs/1904.13007) · [Moran Notebook](https://www.kaggle.com/code/moranzavdi/siclink-ocf-audit-v1-drift-14-point-verifier)
- 正向编码从多尺度方向特征、神经元空间感受野和 session 校准预测 spike 计数，再从真实响应似然反演像素。其有效性用留出字的响应预测和官方 OCF 分别衡量。已发表解码工作使用的神经元数或刺激域与本赛题有差异，效果需要在本赛题验证。[视觉皮层电影重建](https://elifesciences.org/articles/105081) · [MEIcoder 论文](https://arxiv.org/abs/2510.20762) · [MEIcoder 代码](https://github.com/Johnny1188/meicoder)
- 感觉驱动和试次间增益波动可用 Gamma–Poisson 模型描述。下一轮保持字形模板相同，只比较 Poisson 和用训练重复估计离散度的负二项似然。 [Partitioning neuronal variability](https://www.nature.com/articles/nn.3711) · [Kaggle 消融](https://www.kaggle.com/code/ccjjllt/siclink-negative-binomial)

## 2026-11-25 第二轮研究

| Kaggle 对照 | 留出结果 |
| --- | --- |
| 重复次数，320 个已知候选字的诊断 | 同一 session 的诱发总计数在 1、2、4 次重复下，mice1 Top-1 为 8.52%、12.81%、21.02%，mice2 为 5.39%、8.52%、12.42%。每个重复档每鼠 1,280 个样本 |
| 跨 session 重复代理 | 按已知留出训练字合并四个训练 session 的 16 次重复，mice1/mice2 Top-1 为 52.19%/30.31%，各 320 个字。真实 Task 1 同 session 的 20 次与完整公开字库另有差异 |
| 感受野编码加入候选排序损失 | 同批 256 张图，原字库融合 OCF 19.22，加训练字排序损失为 19.19，两种评分融合为 19.61。相对原方案的融合变化为 +0.39，按字计算的标准误 0.30 |
| 放电与字形 CCA、对比学习 | 同批 256 张图，纯 CCA 为 15.13，纯对比学习 16.87，原字库融合 19.30，加入对比学习后 19.39，配对变化 +0.09，标准误 0.66 |
| 神经群体共享增益 | Task 3 将 50–100 ms 和 100–250 ms 的群体增益分别边际化，相对时窗 Poisson 的变化为 mice1 **+2.62**、mice2 **+2.07**，每鼠 128 张成对 OCF，标准误 1.18/1.06。Task 2 的一对一匹配变化为 +0.42/−0.52，标准误 2.00/1.27 |
| mice2 跨天重复交叉拟合 | 配合已有一对一匹配，留出 session2/4 的适配变化分别为 −0.03/−0.63，标准误 0.19/0.29，每块 128 张成对 OCF。两个半组的 spike 分别用于估计软对应和应用修正 |
| 感受野中心初始化 | 用拟合字的空间相关峰初始化两处感受野后，字库融合 OCF 为 19.08，已有相同划分的随机初始化为 19.37。两项各评估 256 张图 |
| 公开字库渲染细化 | 用拟合字学一个整体几何变换，同 32 个留出参考图和相同候选中，参考图选出的最佳 OCF 由 75.01 升到 79.51，成对增益 +4.50（标准误 1.07）。这是字体覆盖诊断，放电排序后另测 |
| Unihan 使用层级和跨块软共识 | 八块 512 张图，字库融合 19.40，使用层级单独 20.21（+0.81，标准误 0.53），排除拟合字形单独 18.91，使用层级加软共识及排除项最高 20.65（+1.25，标准误 0.77）。组合的增益主要来自 mice1/session2，另做不同字划分验证 |
| Task 1 放电候选深度 | 两鼠各两个 session，16 个独立留出字的首选 OCF 21.00。用参考图仅作诊断，取前 4、16、32 中最高者，分别为 27.50、34.98、39.14；同批字以参考图检索字库为 77.94。前 32 的配对增益为 +18.14（按字标准误 2.62）。候选召回和精排都限制了得分。 [候选深度](https://www.kaggle.com/code/ccjjllt/siclink-atlas-depth)
| Task 2 无标签跨天协方差适配 | 八个留出块各 32 张，LDA 一对一 OCF 19.16。噪声协方差变换与目标块精度更新融合后为 19.99（+0.83，按字标准误 0.44），仅变换为 19.05、仅更新精度为 19.62。全块 Top-1 从 4.22% 到 4.23%，先保留已验证的正式策略。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-coral-validation) · [CORAL 原论文](https://ojs.aaai.org/index.php/AAAI/article/view/10306)
| Task 1 第二个留出字划分，种子 509 | 八块 384 张。原排序 19.17，先验组合 22.31，几何校准单独 19.82，先验组合加校准 **22.94**（+3.77，按字标准误 1.29），八个块都提高。仅添加先验再用一对一指派为 20.96，软跨块奖励后 20.82。候选采用先验组合和渲染校准，保留逐样本首选。 [独立留出对照](https://www.kaggle.com/code/ccjjllt/siclink-atlas-joint-validation)
| Task 3 刺激后早期放电去噪 | 0–25 ms 的单元活动解释留出重复 50–250 ms 波动的约 4.2%/3.7%，对应总计数模板 OCF 仅提高 +0.14/+0.10。改用 0–50 ms 时预测更多试次波动，OCF 变化为 −0.06/−0.03。每鼠每视图 128 张配对图。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-early-noise) · [视觉皮层持续活动研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6525101/)
| Task 3 群体增益与一对一指派 | 每鼠 512 张配对图。v4 策略为 18.73/16.71，当前策略强制一对一后为 17.82/16.01。轮廓与时窗群体增益融合的逐样本策略为 18.83/17.33，配对变化 +0.10/+0.62，标准误 0.34/0.41。mice2 的轮廓与 Poisson 融合为 17.54（+0.83，标准误 0.44），与此前小样本的选择不同，待另一次留出重复确认。 [Kaggle 双 T4 重跑](https://www.kaggle.com/code/ccjjllt/siclink-gain-contour)

重复次数诊断支持用独立试次增强排名。当前对比学习方案的 OCF 增益小，继续保留由正向似然和放电重建轮廓构成的评分。下一组对照只从拟合字及公开 Unihan 使用层级估计字库先验，并比较独立打乱后的放电块是否能提供共识。 [重复诊断](https://www.kaggle.com/code/ccjjllt/siclink-repeat-diagnostics) · [排序损失](https://www.kaggle.com/code/ccjjllt/siclink-ranking-encoder) · [对比学习](https://www.kaggle.com/code/ccjjllt/siclink-contrastive-atlas) · [CLIP](https://arxiv.org/abs/2103.00020) · [Unihan](https://www.unicode.org/reports/tr38/)

群体增益从训练重复中不同单元的同字残差协方差估计。Task 3 扩大抽样，与当前逐鼠轮廓策略和固定权重融合做同批对照。 [共享增益](https://www.kaggle.com/code/ccjjllt/siclink-population-gain) · [跨天交叉拟合](https://www.kaggle.com/code/ccjjllt/siclink-crossday-adaptation) · [Gamma–Poisson 变异模型](https://www.nature.com/articles/nn.3711)

官方参赛包明确 Task 3 每个训练 session 给每个已见字保留一条测试 trial。可以仅用放电分数为整个块求一对一指派。Task 1 每块每个未见字也只产出一张图，由 20 次刺激合并；用公开字库试验矩形一对一指派和软跨块共识，打乱各块的查询顺序后求解。 [官方参赛包](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data) · [感受野初始化](https://www.kaggle.com/code/ccjjllt/siclink-receptive-init) · [渲染细化](https://www.kaggle.com/code/ccjjllt/siclink-font-style-validation)

### 候选 v4

Task 1 在训练字对齐的 Noto Sans 神经排序上加 Unihan 17.0 使用先验与软排除，并对选出的字形应用从 512 对训练字拟合的几何变换。Task 2 沿用逐鼠一对一匹配策略，Task 3 沿用候选 v3 的逐鼠轮廓策略。用 `--unseen-decoder font-fused --align-font-training --font-usage-prior --calibrate-font-output --task3-decoder contour-validated` 在 Kaggle 双 T4 生成，用时约 165 秒。完整核对 30,200 个样本 ID、顺序和 64×64 单通道 PNG。提交包 SHA-256 为 `6571b4619b6c7ec19e5a5ebddf74fa851d9c394ad626cd7ed3005961c3cbc4d5`。2026-11-25 的 UTC 日提交额度已用完，v4 尚未提交。 [Kaggle 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v4) · [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第三轮研究

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 正式重复放电可识别性 | 每个记录块的 340 个样本随机各分成两组不重叠重复。用诱发总计数和训练字估计的噪声协方差，只凭放电在同一块匹配另一组重复。每半 1、2、4、10 次时，mice1 Top-1 为 4.56%、11.10%、27.87%、66.47%，mice2 为 3.01%、6.25%、15.07%、38.90%。此项直接用正式 spikes 和已提供的组内关系，未使用图片或隐藏标签。 [Kaggle 诊断](https://www.kaggle.com/code/ccjjllt/siclink-test-repeatability) |
| Task 1 模板预测误差与试次噪声 | 同一字库渲染和先验下，v4 的首选 OCF 22.55，试次协方差 22.49（−0.06，标准误 0.47），加入留出拟合字估计的预测误差为 22.37（−0.18，标准误 0.53）。在另抽的 8 个字 × 4 块上，试次协方差把前 16 候选中的参考图诊断最高分从 39.56 提到 44.85，加入原分数的融合把前 32 候选最高分从 49.17 提到 52.02。首选每块 64 张。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-predictive-atlas) · [神经群体噪声协方差研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13406934/) |
| Task 1 多录制 GCCA | 320 留出字、八块独立打乱后，放电嵌入在同鼠录制间两两匹配 Top-1 为 13.07%，跨鼠为 4.88%。要求在训练内校准达到 65% 的精度并有至少 12 个配对时，没有可用匹配。进一步测正式 20 次重复拆半后的可重复性。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-multiview-alignment) · [GCCA 研究](https://arxiv.org/abs/1605.09459) |
| Task 2 跨天分层经验贝叶斯 | 同批双鼠留出 session2/4，每块 64 张。现有一对一策略为 21.04，分层模板为 20.70，结合查询重复的噪声协方差为 22.04（+1.00，按字标准误 1.46）。结果随留出录制而变，另用 session1/3 复核和现有策略的融合。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-hierarchical-gallery) |
| Task 3 快时序 | 同批各鼠 256 张图，25 ms 轮廓岭回归为 16.77/16.09。每个 25 ms 窗内增加 5 ms 时序对比后，配对变化 +0.37/+0.42，标准误 0.37/0.51；加入首 spike 与短间隔爆发计数后变化 −0.14/−0.10。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-event-timing) |
| Task 1 正式拆半跨录制匹配 | 先在留出训练字分成 2/2 重复，只有 7 个配对在四种半组比较中都保留，其中 6 个正确。对正式数据分 10/10 后，28 个记录块对有 218 个配对通过同样条件，覆盖每个块对查询的 2.29%，同鼠块对为 4.93%。正式数据的结果只衡量复现率。 [Kaggle 诊断](https://www.kaggle.com/code/ccjjllt/siclink-multiview-repeat-links) |
| Task 1 训练字精确 OCF 重排 | 在噪声和 v4 分支提名的并集上，只用另留的训练字评估约 5,853 对候选，拟合逐鼠重排器。外层训练集留出的 256 张图，v4 为 22.57，带 v4 锚定的重排为 24.34（+1.77，按字标准误 1.40）。mice2 的两个留出块分别提高 +4.03/+1.60。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-exact-atlas-rerank) |
| Task 1 重排的 mice2 复核 | 换用种子 211 的留出字及 session2/3，128 张配对图，v4 为 17.94，带锚定的精确 OCF 重排为 21.12（+3.18，按字标准误 1.59），两块变化 +1.66/+4.71。结合第一次验证，mice2 的四个记录块都提高，因此 v6 先在 mice2 启用重排。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-exact-rerank-mice2-recheck) |
| Task 2 固定权重融合首次复核 | 在独立留出的 session1/3，基准一对一 OCF 19.02，融合 35% 分层分数为 21.21（+2.18，按字标准误 0.91），四个鼠–录制块都提高。每块 96 张。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-hierarchical-blend) |
| Task 2 固定权重融合第二次复核 | 留出 session2/4，每块 96 张。基准为 22.54，融合为 24.30（+1.76，按字标准误 0.86）。两组共 768 张成对评估图，八个块均提高，平均 +1.97。融合前按查询减均值，再分别用全块 RMS 缩放，最后按 65%/35% 融合后一对一指派。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-hierarchical-blend-recheck) |

### 候选 v5

`--task2-decoder validated-hierarchical` 启用经两组留出验证的 65%/35% 融合。模型在 Kaggle 双 T4 上生成完整候选，用时约 169 秒。完整重跑改变了 Task 2 的 2,584 个预测和 Task 1 的 40 个预测。提交包在 Kaggle CPU 用 `kaggle_merge_candidate_outputs.py` 取 v5 的 Task 2 与 v4 的 Task 1/3，方便将正式分数变化归因到已测的升级。已用源包的哈希和官方 ID 顺序逐行核对全部 30,200 个预测，最终 SHA-256 为 `a8c85da2a7f2733c92f7cb8e9b5fc24d341ec1abee1a3d8da13f1075c75dc760`。Kaggle 2026-11-25 UTC 的提交名额已用完，此包尚未提交。 [v5 模型](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v5) · [Kaggle 最终包](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v5-task2-update)

### 候选 v6

用 `kaggle_task1_metric_update.py` 从完整训练集为 mice2 拟合精确 OCF 重排。嵌套编码、误差估计和 OCF 校准分别使用互不重叠的训练字，四个 session 共评分 5,951 个训练候选对。Kaggle GPU 用时约 276 秒。以 v5 最终包为输入，只重新选择 mice2 的 1,360 个 Task 1 字形，其中 971 个预测发生变化。已核对全部 30,200 个 ID、顺序、64×64 单通道 PNG 头和源包哈希。候选 SHA-256 为 `ea91c7ef73f5e6cbe02dc9aed7ecac3a6aab8a79360355d3ee210b14abb1f981`。本轮用于比较的 Task 1 留出样本每字有 4 次重复，正式预测使用提供的 20 次。v6 是本 UTC 日结束后待提交的候选。 [Kaggle v6 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v6)

## 2026-11-25 第四轮研究

主办方的 SOPHON 把计数、解剖坐标和脑区编码为神经单元 token，用区域内及全局注意力对齐 CLIP 图像特征。其公开结果来自 Allen 自然图像数据的图像级中心检索。本比赛的试验适配用竞赛单元索引和拟合字估计的功能信息，并以单次留出重复的 OCF 衡量。公共视觉核心实验另参考多鼠共享视觉特征和逐单元读取的前向建模。 [SOPHON](https://github.com/SiClink/neural-to-image-retrieval) · [小鼠视觉皮层共享核心研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11981942/) · [感受野双分支研究](https://www.nature.com/articles/s41593-026-02213-3)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 精确 OCF 重排 | 两次拟合、一次训练重排、一次外层评分。四个留出块共 384 张，v6 逐鼠策略 16.61，弱锚定重排 16.19，强锚定 16.82（+0.21，标准误 0.53）；轮廓与时间群体增益的固定融合为 17.50（+0.88，标准误 0.47）。在内层提名列表中用参考挑选的均分为 36.55/35.08，两鼠都有重排空间。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-exact-metric) |
| Task 1 多鼠共享视觉核心 | 用两鼠同一拟合字的平均试次训练公共卷积核心，各单位有独立空间和时间读取。四读出核心的 320 已知图候选 Top-1 在四块均提高。公共字库四块 256 张中，单鼠核心为 19.72，公共四读出为 20.32，按 60%/40% 融合为 20.36（+0.65，按字标准误 0.30），原始两读出的公共核心为 19.90。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-shared-visual-core) |
| Task 1 跨录制双线索匹配 | 嵌套训练字校准 GCCA 与字库后验。在独立打乱的 320 个留出字中，同鼠两两首位正确率从 12.66% 提升到 15.55%，跨鼠从 4.26% 提升到 5.23%。正式预设精度阈值下没有匹配通过；覆盖探查得到 38 组同鼠匹配，其中 21 组正确。保持 v6 的独立解码。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-dual-alignment) |
| Task 3 群体增益改用种子 419 的留出试次 | 每鼠 512 张成对图。mice2 的 v6 轮廓 OCF 16.32，轮廓与时间群体增益等权融合后为 17.31（+0.99，标准误 0.38），四个录制块分别为 +0.16、+0.60、+1.49、+1.69。mice1 的原轮廓加 Poisson 为 18.34，相同替换为 18.08（−0.26）。因此改动只影响 mice2 的 Task 3。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-gain-repeat419) |
| Task 3 重复配对去噪 | mice2 每个拟合字符的其他重复和其他天响应构成互不重叠的去噪目标。192 张成对图中，v6 为 16.11，稳定子空间岭去噪为 14.94，残差 MLP 为 15.10，加到 v6 的固定融合为 15.76（−0.35，标准误 0.34）。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-paired-denoise-mice2) · [配对带噪观测去噪](https://proceedings.mlr.press/v80/lehtinen18a.html) |
| Task 3 SOPHON 风格单元注意力 | 两鼠各 384 张。实际单元索引将匿名注意力的 14.59/14.50 提高到 16.27/16.41，训练字功能足迹为 16.08/16.63。现有逐鼠策略为 18.51/16.09。固定 30% 单元注意力融合相对现有策略变化 −0.13/+0.08，标准误 0.44/0.17。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-unit-attention) |
| Task 1 共享视觉核心的同宽复核 | 换用种子 887 与 session2/3，每块 96 张。单鼠两读出为 21.15，单鼠四读出 21.09，共享四读出 21.45。将共享四读出加到单鼠模型为 21.42（+0.27，按字标准误 0.32）。增益较第一次小，候选采用 v6 的 Task 1。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-shared-core-recheck) |

### 候选 v7

以经过核验的 v6 包为输入，用 `kaggle_task3_gain_update.py` 在 Kaggle GPU 为 mice2 的 Task 3 拟合前向模型和轮廓模型。时间群体增益在 50–100 ms 和 100–250 ms 两段从全部可用训练重复估计，分别为 0.09254 和 0.04324。两种评分等权融合，重新选取该鼠四个录制的 10,992 个字形，其中 8,331 个预测相较 v6 改变。Kaggle 用时约 31 秒，核对了 30,200 个官方 ID、顺序、64×64 单通道 PNG 头和输入哈希。SHA-256 为 `32c8933f0b3afc886c9d1de24beb64c3f9c320d7bde925a090641357825441d9`。Kaggle 2026-11-25 UTC 名额已用，此包待下一个 UTC 日提交。 [v7 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v7)

## 2026-11-25 第五轮研究

Task 1 的候选分数增加从群体放电中回归的多尺度方向轮廓。训练时使用已对齐的字库渲染，验证按字符整组留出，沿用 v4 的字库先验和最终渲染。使用 192 个稳定神经方向，逐查询标准化分数，以 60% v4 和 40% 轮廓融合。另用内层留出的拟合字估计逐维预测误差，比较按可靠度和残差方差加权的候选分数。 [小鼠 V1 自然图像解码研究](https://www.nature.com/articles/s41467-020-14645-x) · [空间频率通道研究](https://www.nature.com/articles/s41467-022-29656-z)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 方向轮廓字库检索 | 种子 887、两鼠的 session1/4、384 张配对留出图。v4 首位 OCF 20.22，方向轮廓融合 21.94（+1.72，按字标准误 0.64）。mice1 两块为 +1.09/+4.43，mice2 两块为 −0.04/+1.40。按预测误差逐维加权的融合为 20.24。另用 32 张图检查候选覆盖，前 32 取最佳为 37.93/40.15，增益标准误为 3.53。Kaggle GPU 用时 194 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-atlas-contour-likelihood) |
| Task 1 重复试次噪声似然 | 种子 509、两鼠的 session2/3、48 个留出字。每字用 4 次重复时 v4 为 22.98，固定对角、拆半逆方差和半组一致性惩罚相比分别 −0.02、−0.14、−0.10，三者的标准误均约 0.2。正式数据的两组 10 次重复首选一致率为 21.03%–21.25%，只衡量候选复现率。Kaggle GPU 用时 152 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-repeat-atlas) |
| Task 3 对齐后的笔画共识 | mice2 四个 session 共 192 张留出图。在 v7 的前 12 个候选上先对齐再改笔画。v7 硬字形 OCF 16.25，对齐后取代表字 16.13，像素投票 15.30，局部新增笔画 15.96，允许同时增删为 16.09。Kaggle CPU 用时 404 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-stroke-consensus-mice2) |
| Task 2 跨天来源模板混合 | 双鼠留出 session2/4，384 张成对 OCF。按查询的 4 次重复和全部 6 个两次重复子集计算边际预测证据，为源 session 估权。均匀混合为 21.51，证据加权为 21.91，与 v7 融合后为 23.98。v7 为 24.30，融合变化 −0.32，按字标准误 0.95。Kaggle GPU 用时 174 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-source-mixture) |
| Task 1 方向轮廓独立复核 | mice1 的 session2/3，种子 1411，256 张成对留出图。v4 为 25.03，方向轮廓融合 25.99（+0.96，按字标准误 1.26），两份录制分别为 +2.61/−0.69。按预测误差加权的融合为 26.02（+0.99，标准误 0.98），首轮同一方法只提高 +0.02。Kaggle GPU 用时 118 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-atlas-contour-mice1-recheck) |
| Task 1 方向轮廓扩大复核 | mice1 全部四个 session，种子 2209，每块 240 张，合计 960 张成对留出图。v4 为 23.09，方向轮廓融合为 24.37（+1.28，按字标准误 0.60）。四份录制分别为 +0.61、+2.46、+1.73、+0.32。按预测误差加权的融合为 23.92（+0.83，标准误 0.55）。Kaggle GPU 用时 136 秒。 [Kaggle 扩大复核](https://www.kaggle.com/code/ccjjllt/siclink-atlas-contour-mice1-expanded) |

### 候选 v8

用 `kaggle_task1_contour_update.py` 从全部 2,748 个训练字拟合 mice1 的方向轮廓读出。正式 Task 1 使用每字 20 次重复，并按 60% v4 字库分数与 40% 方向轮廓分数排序，保留 `usage_unseen` 先验和训练字渲染校准。在 v7 上重算 mice1 Task 1 的 1,360 条预测，共有 838 条改变。逐行核验其余 CSV 记录与 v7 相同，并核对全部 30,200 个官方 ID、顺序及 64×64 单通道 PNG 头。Kaggle GPU 用时 95 秒，候选 ZIP 的 SHA-256 为 `eea6a9f43c24fd7e3ba37cb8d16c7d273df5ef00e24befbcd4bc677802638eaa`。 [Kaggle v8 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v8)

扩大复核的四个录制块均提高，选 v8 作为下一次提交候选。留出评估每字 4 次重复，正式数据每字 20 次，榜分待验证。2026-11-25 UTC 的唯一名额已经使用，下一个窗口于北京时间 2026-11-26 08:00 开启。 [提交规则](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/rules)

## 2026-11-25 第六轮研究

Task 3 用同一字的训练重复估计 50–100 ms 和 100–250 ms 两个时间段的神经元间波动。低秩 Poisson 模型允许各单元对潜在增益有不同幅度和符号。协方差估计扣除 Poisson 方差，并用训练字拆半估计噪声底。单试次用二阶近似积分预测字与潜在增益的似然；打乱单元与增益的对应作为对照。另一条实验直接对字库的轮廓效用矩阵建低秩读出，保留硬字形作为候选。 [共享皮层变异](https://pmc.ncbi.nlm.nih.gov/articles/PMC4534383/) · [潜变量放电模型](https://proceedings.neurips.cc/paper/2011/hash/7143d7fbadfa4693b9eec507d9d37443-Abstract.html)

Task 2 跨天对照以整段训练录制作为外层留出。几何实验用四次重复拆半估计 crossnobis 距离，衡量同一组字跨天的放电几何能否帮助定位候选。监督实验在其他已标注录制之间模拟匹配，学习共享响应表示并保留完整图库的一对一评估。 [Crossnobis 距离](https://doi.org/10.1016/j.neuroimage.2015.12.012) · [情景式领域泛化训练](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf) · [可微置换学习](https://arxiv.org/abs/1802.08665)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 2 跨天几何与自动锚点 | 双鼠各四个录制作外层留出，八块距离排序 Spearman 相关均值 0.427。256 张配对 OCF 中，分层经验贝叶斯为 20.34，增加 15% 几何重排后为 20.19（−0.15，按字标准误 0.55），打乱锚点分数为 20.64。此轮基准是独立分层模型，v8 的正式 Task 2 还融合了逐鼠解码器。Kaggle CPU 用时 196 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-crossnobis) |
| Task 2 正确锚点覆盖诊断 | 使用外层留出训练标签随机指定 96 个正确配对，评分时排除这些查询行。融合几何后，其余查询的 Top-5 从 12.90% 变为 13.28%，Top-32 从 34.17% 变为 34.40%。这项诊断用于估计锚点质量提升后的收益。Kaggle CPU 用时 17 秒。 [Kaggle 诊断](https://www.kaggle.com/code/ccjjllt/siclink-task2-anchor-ceiling) |
| Task 2 监督跨录制原型 | 双鼠外层留出 session2/4，种子 211，384 张配对图。v7 Task 2 为 24.30，固定原型为 22.40，轮换源录制训练的匹配模型为 19.07，与 v7 融合后为 23.17（−1.13，按字标准误 1.07）。新表示的完整图库 Top-1 在四个留出块都低于固定原型。Kaggle GPU 用时 172 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-episodic-prototypes) |
| Task 3 期望轮廓效用回归 | 低秩读出用 128 个特征保留 97.55% 的中心化轮廓效用能量。留出种子 419，加入 35% 效用分数后，mice1 从 18.34 变为 17.83（−0.51，标准误 0.33，512 张），mice2 从 16.86 变为 16.61（−0.25，标准误 0.33，256 张）。和等权引入普通轮廓分数的对照相差 −0.05/+0.005。双 GPU 首次运行的 mice2 遇到并行初始化错误，修正共享分解的执行顺序后，mice2 在 Kaggle CPU 配对复核用时 404 秒。 [mice1 GPU 日志](https://www.kaggle.com/code/ccjjllt/siclink-expected-contour-utility) · [mice2 CPU 对照](https://www.kaggle.com/code/ccjjllt/siclink-expected-utility-mice2-cpu) |
| Task 3 低秩放电波动 | 留出种子 419，双鼠各 512 张配对图。与轮廓融合后，mice1 OCF 为 18.82（v8 18.70，变化 +0.12，按字标准误 0.37），mice2 为 17.53（v8 17.65，变化 −0.13，标准误 0.31）。相对时窗共享 Gamma，低秩模型的完整图库负对数损失分别由 8.60/8.43 降至 8.27/8.04，Top-5 由 8.24%/4.63% 变为 8.17%/4.45%。Kaggle 版本 1 在 T4 的批量矩阵求逆报错，版本 2 在 Notebook CPU 求逆，其余计算用 GPU，成功用时 66 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-latent-spikes) |

下一次提交候选选 v8。低秩放电模型改善了单试次的完整图库对数损失，候选召回和 OCF 的变化较小。正式最高分仍为 20.54607，下一个提交窗口为北京时间 2026-11-26 08:00。 [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第七轮研究

Task 1 参考 Sensorium-Viz，先用带标签的拟合字训练 spike 前向编码器，再给公共 Noto 字形生成训练用 Poisson 放电。每个录制的模拟响应通过拟合字的实测放电校准，轮廓读出同时测试纯实测和 35% 模拟数据加权，并分别比较线性与 Fourier 特征。待评估字整字留出，参考图仅用于官方 OCF 及前 16 候选的覆盖诊断。Sensorium-Viz 的结果来自自然图像和钙成像。 [Sensorium-Viz 论文](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202520220) · [开源代码](https://github.com/GuanLab/sensorium-viz)

Task 2 检验一种受稳定潜变量跨天解码启发的模型。先从已标注录制学习共享字形放电码和每个记录日的观测载荷，再用目标录制的无标签重复估计跨日观测差异。只允许沿已见录制的载荷变化调整，留出整块评估官方 OCF 和完整图库排名。 [NoMAD 跨日神经解码](https://www.nature.com/articles/s41467-025-59652-y)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 模拟放电增强 | mice1 四个录制、种子 961、512 张留出图。现有 v8 为 25.80，纯实测 Fourier 为 26.20（+0.40，按字标准误 0.34）。在读出中加入 35% 的模拟数据后，线性和 Fourier 分别为 25.02、25.49。另在 32 张图用参考图检查 v8 的前 16 个候选，最佳候选均值为 55.33。Kaggle GPU 用时 116 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-synthetic-atlas) |
| Task 2 共享刺激与跨天载荷 | 双鼠各四个录制分别留出，种子 211，768 张配对图。v8 为 22.75，共享固定模板为 18.16，仅更新目标噪声为 19.09，适配目标载荷为 18.78，与 v8 融合后为 22.55（−0.21，按字标准误 0.63）。Kaggle GPU 用时 169 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-shared-stimulus) |
| Task 1 词汇与形态的字库覆盖 | 两组各 320 个留出字，只用参考图评估候选池的形状覆盖。只保留前 1,000 个字形时，现有先验两组覆盖均为 86.56%，形态与 Unihan 密度先验为 95.94%/97.50%。加入 SUBTLEX-CH 实际频率为 95.63%/95.63%，频率打乱对照为 94.69%/97.19%。前 3,000 时现有先验已达 98.75%/100%。覆盖定义是候选的图像描述符相似度距视觉最优不超过 0.035。Kaggle CPU 用时 50 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-lexical-support) · [SUBTLEX-CH 论文与数据](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0010729) |
| Task 1 模拟器校准的独立验证 | 2,048 个拟合字训练编码器，192 个独立字拟合仿射校准，188 个独立字在 mice1 四个录制作封闭图库识别。原始模拟、只校准均值、仿射校准的 Top-1 分别为 24.87%、25.13%、24.73%。仿射校准降低了校准集误差，但仅一个留出录制的误差下降；加入模型误差协方差后，Top-1 较仅用重复噪声的版本下降 2.39 个百分点。该诊断的候选字形来自留出训练图，评估的是模拟放电质量。Kaggle GPU 用时 39 秒。 [Kaggle 诊断](https://www.kaggle.com/code/ccjjllt/siclink-simulator-transfer-diagnostics) |
| Task 1 以 v8 为基准的精确 OCF 重排 | mice1 四个录制，种子 961，128 个留出字共 512 张图。v8 为 25.92，重排加 2 分锚定为 32.21（+6.30，按字标准误 1.35），加 4 分锚定为 32.43（+6.52，标准误 1.28），四个录制分别 +6.69/+7.65/+3.18/+8.55。用校准字的 8,111 个候选和参考图配对训练重排器，校准字与前向模型拟合字分开。候选池合并 v8 前 16 和三种其他评分的前 6。另在 32 张留出图中用参考图选最佳，候选并集为 60.91，v8 前 16 为 55.33。Kaggle GPU 用时 436 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-v8-metric-mice1) |
| Task 1 精确重排的互斥划分复核 | 种子 2209，排除首轮全部 320 个留出字后重新划分。mice1 四个录制各 160 张，640 张配对结果显示 v8 为 25.80，重排加 4 分锚定为 29.63（+3.82，按字标准误 0.90），加 2 分锚定为 29.64（+3.83，标准误 0.97）。权重 4 的四个录制变化为 +5.05/−0.42/+2.48/+8.18。两轮合计 1,152 张配对图、288 个不重合的留出字，八个块有七个提高。Kaggle GPU 用时 534 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-v8-metric-recheck) |
| Task 1 形态先验的真实放电验证 | mice1 两个不重合的字符划分，各 96 个字，每字四个录制，合计 768 张配对 OCF。全字库 v8 为 24.680/26.903，连续形态先验为 24.733/26.915，按字标准误 0.588/0.509。限制同一 v8 排名到 1,000 候选，现有先验为 24.125/26.347，形态先验为 24.601/26.970。两种 3,000 候选设置的结果与全库 v8 相同。Kaggle GPU 用时 187 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-lexical-ocf-mice1) |

### 候选 v9

两组不重合的留出字支持将精确 OCF 重排用于 mice1 Task 1。用 `kaggle_task1_v8_metric_update.py` 在全量训练集内划分 96 个 OCF 校准字和 384 个噪声校准字，前向模型与方向轮廓读出在另一组字上拟合；重排器用 7,855 对候选和参考图训练。预测阶段用完整拟合的 v8 作锚点，锚定权重为 4。重算锚点的 1,360 条预测中，1,346 条的编码与原 v8 包相同；首位选择不变的查询直接使用原 v8 包中的记录。实际更新 932 个 mice1 Task 1 预测，另外 28,840 个任务和鼠的记录逐字节核验。全部 30,200 个官方 ID、顺序、64×64 单通道 PNG 头和哈希均已检查。Kaggle GPU 用时 346 秒，产物 SHA-256 为 `b5817a9cffb29246422452f524cdff941a4990722681c5c3046da514eef5f8ee`。 [Kaggle v9 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v9)

下一次提交选 v9。留出验证每字用 4 次重复，正式预测用 20 次，正式分数待验证。2026-11-25 UTC 的提交名额已用完，下一个窗口为北京时间 2026-11-26 08:00。 [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第八轮研究

Task 1 测试将方向轮廓候选加入 mice2 已在用的精确 OCF 重排，对照其 v9 中的实际排序器。Task 2 留出整份录制作评估，在另一份带标签录制上拟合重排器。Task 2 保留 v9 全局指派的所有配对，使新的候选边仍可求完整的一对一指派，另外记录逐行选择作为诊断。

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 单试次的 5 ms 共放电 | mice1 用拟合试次选 28 个重复响应较可靠的单元，在每个 25 ms 窗扣除其均值后统计单元对的同步放电特征。留出种子 419，四个录制各 40 张，共 160 张。25 ms 计数轮廓基准 17.81，加入真实共放电为 17.85（+0.03，按字标准误 0.53），独立循环扰动时序并保留 25 ms 计数的对照为 16.76（−1.06）。Kaggle CPU 用时 158 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-spike-synchrony) · [小鼠 V1 单元对相关研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC4153809/) · [条件时序扰动](https://pubmed.ncbi.nlm.nih.gov/22031767/) |
| Task 3 字形边缘软化 | 32 个校准字和 48 个留出字，预测字形由精确匹配和不同距离的视觉邻居构成。字形软化保持 PNG 在阈值 128 下的像素形状。三种软化在精准匹配时约损失 2.57–14.44 OCF，在最近视觉邻居上损失 0.13–2.28，较远邻居的最高变化约 +0.08。校准选择二值输出。Kaggle CPU 用时 405 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-antialias) |
| Task 1 向 mice2 迁移重排 | 种子 2209，640 张配对图，160 个未见字。v9 的原 mice2 排序器为 22.72，加轮廓候选后用 2 分锚定为 24.05（+1.33，按字标准误 0.79），4 分锚定为 24.01（+1.28，标准误 0.80）。2 分锚定四个录制分别 +4.04/−1.88/+0.07/+3.08，只用轮廓融合排序为 21.25。Kaggle GPU 用时 733 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-v9-metric-mice2) |
| Task 2 全局指派与精确 OCF | 两鼠分别留出 session2/4，共 384 张配对图。v9 指派为 24.30，4 分锚定的精确 OCF 指派为 25.44（+1.14，按字标准误 0.84），8 分为 25.07。逐行直接重排仅 22.87。mice1 的 4 分方案为 31.38 对 29.01（+2.37，标准误 1.42，录制变化 +1.44/+3.31），mice2 为 19.49 对 19.59。首版的 SciPy 索引类型与 PyTorch 不一致，修正版 Kaggle GPU 用时 319 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-exact-metric) |
| Task 1 迁移重排的独立复核 | 种子 613 排除 961 和 2209 的外层留出字，mice2 又评分 640 张配对图。v9 为 23.28，2 分锚定为 23.55（+0.28，按字标准误 0.63），4 分为 23.81（+0.53，标准误 0.64）。4 分方案在 session1 两轮分别 +4.61/+3.10（各 160 个互不重合的字，标准误 1.58/1.55）。另三个录制的结果不一致或变化很小，因此只为 session1 使用此方案。Kaggle GPU 用时 676 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-metric-recheck) |
| Task 2 指派的独立录制复核 | mice1 改留出 session1/3，种子 509，320 张配对图。v9 指派为 23.54，4 分锚定为 23.61（+0.07，按字标准误 0.89），8 分锚定为 24.61（+1.07，标准误 0.82）。8 分方案在 mice1 的首轮为 +1.35（标准误 1.35），四份录制各为 +0.88/+0.93/+1.26/+1.77。正式候选只在 mice1 Task 2 使用 8 分锚定。Kaggle GPU 用时 167 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task2-mice1-metric-recheck) |

### 候选 v10

用 `kaggle_mice2_metric_update.py` 从全部 2,748 个训练字拟合重排，并仅对 mice2 session1 的 Task 1 使用 4 分锚定。该录制的 340 个查询各用 20 次重复，共更新 v9 的 253 条预测。Kaggle GPU 用时 332 秒。已检查 30,200 个官方 ID、顺序、64×64 灰度 PNG 和候选源包哈希。产物 SHA-256 为 `014ad8940a5afeaa6e6c55ee89e24efe066789ef7773e5572c51c545525fba26`。 [Kaggle v10 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v10)

### 候选 v11

用 `kaggle_task2_metric_update.py` 在四份有标签的 mice1 录制上轮流留出一份校准，其余三份拟合神经分数。汇合 512 个查询的 8,605 对精确 OCF 标签训练重排器。用全部四份训练录制拟合 `session5` 的神经分数，在候选边上用 8 分锚定求全局指派。2,748 个查询各有唯一字形，未触发原指派回退。v9 中的 378 条 mice1 Task 2 预测发生变化。Task 2 组件在 Kaggle GPU 用时 230 秒，SHA-256 为 `6a98a2614879d66a35dbd6d0cb6aa8bba266fbd528defde54ace7930c19e9663`。 [Kaggle Task 2 组件](https://www.kaggle.com/code/ccjjllt/siclink-task2-mice1-metric-update)

Kaggle CPU 将 v10 与已核验的 Task 2 组件合并为 v11。相较 v9，实际更新 253 条 mice2 session1 Task 1 预测和 378 条 mice1 Task 2 预测。逐条核对两份来源、30,200 个官方 ID、顺序和 64×64 灰度 PNG。v11 的 SHA-256 为 `20e4dd1f80f63f5635f0bc5cf16369500bb4d15b2d2d3ea9971a8925b07f21e8`，选为下一次提交候选。正式最高分仍为 20.54607，2026-11-25 UTC 名额已使用，下一个窗口于北京时间 2026-11-26 08:00 开启。 [Kaggle v11 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v11) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第九轮研究

Task 3 分开测量单试次神经信息和字形选择的损失。图库效用从精确 OCF 标注的视觉近邻与随机图对学习，用不重合的图库字形核对预测误差，再固定同一份放电后验比较旧轮廓代理与新效用。局部轮廓实验在同一候选池上比较标量和空间方向特征，学习相对 v11 首选的 OCF 差。两个实验均先留出独立放电试次，再做内部校准。 [结构化输出的隐式损失嵌入](https://www.jmlr.org/beta/papers/v21/20-097.html)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 局部方向轮廓重排 | 两鼠各 512 张配对图。v11 的 OCF 为 19.56/18.45，4 分锚定的标量重排为 19.50/18.20，加局部方向证据为 19.19/18.48（相对 v11 −0.37/+0.03，按字标准误 0.67/0.32）。8 分锚定下局部重排变化为 −0.34/−0.07。用已知参考图挑选，校准候选池的首位与最佳均分在两鼠分别为 18.22→35.15 和 18.51→33.62。Kaggle GPU 用时 291 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-local-rank) |
| Task 3 放电对墨量的增量信息 | 逐鼠四个录制各留出一次试次，单试次预测对数墨量的相关为 0.66/0.73（每鼠 10,992 次），单用总放电计数为 0.24/0.07。在轮廓模型固定的前 8 个候选上，两鼠各 64 张配对图的基准为 17.21/14.86，加入预测墨量和粗空间分布变化 −0.02/+0.35，按计数分层打乱为 +0.26/+0.32。用参考图挑前 8 候选的诊断在两鼠各 24 张图上可提升 12.16/8.72。Kaggle CPU 用时 242 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-density-diagnostic) · [小鼠 V1 亮度和方向研究](https://www.eneuro.org/content/13/2/ENEURO.0281-25.2025.full) |
| Task 3 同墨量困难负样本 | 两鼠各 512 张配对图。对比训练等量抽取 36 个负样本，难样本版本用 24 个墨量和轮廓能量接近的字符加 12 个广抽样；对照用 36 个广抽样。以 20% 权重和 v11 融合后，困难样本相对随机抽样的 OCF 变化为 −0.08/+0.07，按字标准误 0.31/0.36。困难样本版本在同墨量小池略提高识别率，单独用于全库时，两鼠 Top-5 均下降。Kaggle GPU 用时 124 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-hard-negatives) |
| Task 3 精确图库效用与放电后验 | 用 6,307 个图库图对拟合 OCF，在另 320 个字的 1,003 对图上验证。预测 MAE 为 2.77，拟合值与精确 OCF 的相关为 0.874，旧轮廓代理的相关为 0.798。每鼠 384 张配对留出图，v11 为 17.29/17.55，只用新效用的后验选择为 17.28/17.06；加入 20% 去除均匀图库先验后的效用为 17.54/17.79（+0.26/+0.24，按字标准误 0.50/0.39）。前 32 字形只占平均约 15%/11% 后验质量。Kaggle GPU 用时 240 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-posterior-utility) |

本轮仍选择已核验的 v11 作为下一次提交候选。精确图对拟合能改善图库的视觉效用估计，四项 Task 3 实验均未取得可复现的留出 OCF 增益。 [Kaggle v11 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v11)

## 2026-11-25 第十轮研究

Task 1 分别测量录制内的候选占用和同鼠录制间的对应关系。录制内允许预测字形重复，只对多个查询争用同一公开字形的概率加软惩罚，并留出字库外候选的概率。跨录制实验只使用共有字集已知的训练留出块，在独立打乱的四份录制上比较两两匹配、经第三个录制传递的证据和全局一致的联合匹配。 [GCCA](https://arxiv.org/abs/1605.09459) · [软对应配准](https://www.cise.ufl.edu/~anand/students/chui/tps-rpm.html)

Task 3 将一对一用于约束隐含字符身份，先用独立训练重复校准首选预测的正确率，再对其他查询占用的同一候选做有限降权。另一项实验从放电预测左右和上下的局部笔画形态，测量其能否在同等候选数量下补充全字轮廓。 [字形组件的零样本解码](https://www.ijcai.org/proceedings/2021/85)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 四录制交叉匹配 | 两个不重合的留出字集，每个 320 个字。mice1 的两两指派身份准确率 14.90%/15.00%，引入两跳证据后为 17.19%/17.92%，强制四录制一致则为 12.76%/15.42%。mice2 的两跳准确率为 9.27%/8.85%，高于直接指派的 8.59%/8.33%。mice1 两个中间录制都支持的边覆盖 1.77%，其中 58.82%/70.59% 正确；拆重复后的稳定匹配准确率约 46%–48%。本项仅使用完整训练留出集测量身份，Kaggle CPU 用时 20 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-cycle-alignment) |
| Task 3 可信首选的软排他 | 两鼠各 512 张与 v11 配对的留出图。完整 2,748 条的录制块中只有 2–18 条达到置信阈值，首选实际正确率约 7%。相对同一图库效用的独立后验，软排他变化 +0.00/+0.10 OCF；mice2 只改变 1 张留出预测。完整 Sinkhorn 降低了全部录制的身份对数损失，OCF 相对 v11 变化为 −0.11/+0.50（按字标准误 0.52/0.45）。Kaggle GPU 用时 274 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-soft-exclusion) |
| Task 1 共享字库的软容量 | mice1 四录制独立打乱，每个留出块 320 个字，按 v9 配对评分 160 个。候选的潜在身份总概率超过 1 时降权，最终输出仍可重复。将候选效用修正限制在 ±3 OCF 时，四录制从 31.282 提至 31.919，差值 +0.638（按字标准误 0.317），逐录制 +1.425/+0.588/+0.260/+0.277。效用修正限制在 ±1 OCF 或直接软约束输出时，增益较小。Kaggle GPU 用时 395 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-latent-repertoire) |
| Task 3 局部笔画候选先导 | 两鼠各取录制 2 和 4，每块配对评估 64 张。用局部笔画模型的候选替换原候选集最后四个，top-12 真实字形召回从 4.99%/4.33%/2.73%/3.09% 提至 6.04%/4.69%/3.49%/3.42%，保持全字形态分组的打乱对照均低于原候选集。256 张的首选 OCF 从 17.01 提至 17.53（差值标准误 0.36），32 张的 top-12 精确上限从 30.92 降至 27.31（差值标准误 4.48）。Kaggle CPU 用时 253 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-part-discovery) |
| Task 1 跨录制高斯似然 | 同一组双鼠外层划分，配对 CCA 只使用拟合字。mice1 逐查询的同字似然准确率为 11.98%/10.73%，同一组坐标上的余弦为 5.65%/5.34%。两跳指派为 13.39%/13.12%，低于原 GCCA 的 17.19%/17.92%。mice2 的两跳结果为 8.54%/9.17%，GCCA 为 9.27%/8.85%。Kaggle CPU 用时 28 秒。 [Kaggle 版本 2](https://www.kaggle.com/code/ccjjllt/siclink-task1-cycle-alignment) · [概率 CCA](https://statistics.berkeley.edu/tech-reports/688) |
| Task 3 字形位移与放电变异 | mice1 用 2,108 个字拟合前向放电编码器，320 个校准字拟合位移先验，另留出 320 个字、5,120 次试次。两个各含 67 个单元的子群轮流预测对方。共享位移相对零位移、各半独立位移和保留幅度的字形打乱对照，条件对数似然每半分别改善 +0.349、+0.077、+0.124 nats（按字标准误 0.014、0.007、0.014），四个录制均同方向。此项用已知留出字形条件化放电似然，尚无 OCF 结果。Kaggle CPU 用时 42 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-retinal-shift) · [小鼠视皮层位移建模](https://pmc.ncbi.nlm.nih.gov/articles/PMC9742335/) |
| Task 1 软容量的独立复核 | 另一组 320 个不重合的字符中取 192 个字，在四个录制上配对，共 768 张。使用首轮的 ±3 OCF 修正，v9 的 30.318 变为 30.682（差值 +0.364，按字标准误 0.398），逐录制 −0.053/+1.502/−0.260/+0.267。两轮共 1,408 张图和 352 个留出字，加权差值 +0.49（标准误 0.26）。留出每字 4 次重复，正式每字 20 次。Kaggle GPU 用时 467 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-latent-repertoire-recheck) |
| Task 3 局部笔画候选复核 | 双鼠各四个录制，合计 1,024 张配对图。实际 v11 策略 OCF 18.18，加入 15%/25% 局部笔画分数后为 18.28/18.17（差值 +0.11/−0.01，按字标准误 0.27/0.38），相对使用同一分数在原前 16 个候选内重排，额外变化 +0.02/−0.00。192 张图由参考图从候选中选出的上限提高 +1.75（标准误 1.32），相对打乱对照提高 +0.85（标准误 1.37）。Kaggle GPU 用时 173 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-part-v11-recheck) |
| Task 3 字形位移重排 | mice1 四个录制，每块 128 张独立留出试次。冻结在内层校准得到的位移先验，在 v11 原前 16 个候选内重排。v11 为 18.823，位移修正权重 0.5/1.0 为 18.661/17.575（变化 −0.162/−1.249，按字标准误 0.261/0.505）。Top-1 身份准确率从 2.54% 变为 2.15%/1.17%。Kaggle CPU 用时 482 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-retinal-ranking) |

本轮配对验证完毕，保留 v11 作为下一次提交候选。软容量的优势不够稳定，Task 3 两种重排均未验证出 OCF 增益。实验在 Kaggle CPU 或 T4 上完成。 [Kaggle v11 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v11)

## 2026-11-25 第十一轮研究

Task 1 将字形位移从单次放电扩展到多次重复。编码器、位移先验和验证字符分三组拟合或留出。闭集有 320 个从编码器整字排除的候选，以相同的群体增益模型比较零位移、边际均值、逐次独立位移、组内共享位移，以及保留幅度的方向打乱。灵长类视网膜研究提供了从放电推断未知位移的先例，本实验使用的是小鼠皮层的静态字形与有限的位移网格。 [位移与贝叶斯重建](https://www.nature.com/articles/s41467-024-52304-7) · [开放实现](https://github.com/wueric/wu-nature-comms-2024)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 字形放电率的 Gamma 后验 | 两鼠各有 384 张配对图，并用每鼠 10,992 条查询衡量全字库身份。相同后验均值下，对潜在放电率积分与直接用 Poisson 均值相比，前 16 候选 OCF 差值为 +0.026/+0.043（按字标准误 0.224/0.029）。对照 v11 的 18.597/18.513，Gamma 前 16 重排为 19.284/18.234（差值 +0.687/−0.279，标准误 0.736/0.431）。全字库等量融合略降身份对数损失，直接代入相同的收缩均值就已得到其中大部分增益。Kaggle GPU 用时 88 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-gamma-template) |
| Task 1 多重复位移闭集诊断 | 2,172 字拟合、256 字校准、320 字留出。两鼠各 256 张配对 OCF，四个录制每字 4 次重复。逐次独立位移的 OCF 相对零位移，mice1 43.74→42.80（−0.94，标准误 1.41），mice2 31.19→33.39（+2.20，标准误 1.26）。mice2 的边际均值、共用位移和打乱对照分别变化 +1.03/+1.74/−0.24。mice2 的闭集 Top-1 20.55%→20.23%，四块身份 NLL 均下降，均值 5.757→5.446。使用留出训练图作已知候选的闭集有别于正式公开字体检索。Kaggle CPU 用时 223 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-shift-repeats) |
| Task 3 匹配放电残差增强 | 保持原型锚点和分类器架构固定，以拟合试次估计的同录制、相近活动量的残差构造增强样本。八块合计 512 张配对图，真实试次分类器为 15.12，匹配增强为 15.41（差值 +0.29，标准误 0.25）。v11 为 17.11，以 20%/35% 融合增强分类器后为 16.95/16.34（差值 −0.16/−0.77，标准误 0.35/0.39）。35% 融合提高全部八块全字库 Top-5 召回，首选图的 OCF 下降。Kaggle CPU 用时 727 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-repeat-density-ratio) |
| Task 3 残差分类器的后验和候选覆盖 | 新的试次划分，八块合计 512 张配对图。内层留出校准身份后验温度，外层 v11 为 17.57。沿用同一个视觉效用选择器，用 v11 或融合分类器的后验分别得 17.71/17.70（分类器相对 v11 后验 −0.01，按字标准误 0.09）。全字库 Top-5 从 4.06% 到 5.92%，等预算 top-16 召回从 9.73% 到 11.40%。64 张图用参考图选出的 top-16 OCF 上限从 43.91 到 46.93（+3.02，标准误 2.98），新增候选对已评分图的实际选图没有影响。Kaggle CPU 用时 810 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-repeat-posterior-utility) |
| Task 2 按字形估计的跨天方差 | 双鼠外层留出 session2/4，共 512 张配对 OCF。用其他录制的轮换验证拟合时窗权重，保留全字库一对一指派。按字形估计方差相对同样学习权重的全局方差，在两鼠上分别 −0.56/−0.28 OCF（标准误 0.47/0.41）。全局方差的时窗融合使四块前 32 候选真字召回提高 2.11–6.40 个百分点，mice2 的 OCF 在 session2/4 分别变化 −1.51/+3.87。对照为 v9，mice1 以后加入 v11 的精确 OCF 指派未参与此轮。Kaggle GPU 用时 113 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-conditional-precision) |
| Task 1 开放字库位移复核 | mice2 另留出 320 个不重合的训练字，四个录制各取 128 张配对图。在实际 v11 的固定候选和渲染上加入位移边际似然。原 v11 为 20.4765，两种修正为 20.4895 与 20.5337（+0.013 ± 0.056 与 +0.057 ± 0.062，按字标准误）。较强修正改变 23/512 个首选，逐录制分别 +0.199/−0.078/+0.113/−0.005。每字 4 次试次，正式 Task 1 每字 20 次。Kaggle GPU 用时 517 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-atlas-shift) |

本轮六项实验均已完成。Task 1 的位移闭集收益在公开字体复核中缩小至 +0.057 OCF。Task 2 和 Task 3 得到新的候选来源，留出 OCF 证据仍不足以更新提交包。保留已核验的 v11，当前正式分数 20.54607，下一个提交窗口为北京时间 2026-11-26 08:00。免费 Colab 未成功分配会话，六项实验分别在 Kaggle CPU 或 T4 完成。 [Kaggle v11 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v11) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第十二轮研究

Task 2 保留完整可行的一对一方案，比较把跨天时窗模型提出的字形加入同等大小的边集。Task 3 用内层配对 OCF 学习候选相对 v11 首选的差值，并在相同大小的原候选与混合候选中评估。同期用 5 ms 原始时间窗预测字形，条件在现有 25 ms 放电总数上，单独衡量更精细时间信息。小鼠 V1 的首发放电实验和早期视觉系统的自然刺激研究提供了时间信息的机制依据。 [首发放电实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC5902162/) · [自然视觉时序](https://www.nature.com/articles/nature06105)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 候选相对 OCF 的直接学习 | 双鼠八块，每块 64 张配对图。内层按字交叉拟合的重排阈值最终均选择原 v11。固定正差规则在各块选取 6%–27% 新候选，OCF 从 18.72 到 18.44（差值 −0.27，标准误 0.42），比同评分器在等预算原候选上的 18.46 低 0.01（标准误 0.32）。96 张图用参考图选出的同预算 7 候选上限从 30.08 到 34.90（+4.82，标准误 2.28）。仅按融合身份后验重排 v11 原前 7 个候选时，mice1 从 19.38 到 21.12（+1.73，标准误 1.04），mice2 变化 +0.08。Kaggle GPU 用时 219 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-density-ocf-selector) |
| Task 1 公开字体渲染覆盖 | 在拟合字上分别校准三种字体的几何和风格，用 48 个留出参考图在统一字形距离下查找候选。使用参考图从每种字体的 6 个候选中选最佳的 OCF 为 Noto Sans SC 80.56、SimXiHei 71.26、文泉驿正黑 54.09。每种字体取 2 个后再选最佳为 80.86，较 Noto 单独 +0.30（标准误 0.17），48 张有 5 张改善。所有首选码位均在三种字体的共有支持内。这是字体覆盖诊断，未使用放电。Kaggle CPU 用时 375 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-font-family-coverage) · [SimXiHei](https://github.com/lxgw/SimXiZhi/releases/tag/v1.103) · [文泉驿正黑](https://sourceforge.net/projects/wqy/files/wqy-zenhei/0.9.45%20%28Fighting-state%20RC1%29/) |
| Task 2 跨天时窗候选与原边集先验 | 两鼠各留出 session2/4，共 512 张配对图。时窗提名提高四个候选池真字覆盖 1.64–3.49 个百分点，但以同一评分器指派后，相对同量 v9 尾部提名的 OCF 差值为 +0.0001（标准误 0.284）。保留原候选，仅加入时窗分数，mice1 相对同折 v11 为 +1.129（标准误 0.492，两个录制分别 +1.046/+1.213）；mice2 相对同一稀疏 OCF 评分器为 +0.093（标准误 0.375）。所有方法都给出 2,748 个唯一匹配。Kaggle GPU 用时 764 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-temporal-nominations) |
| Task 3 给定 25 ms 总数的 5 ms 时序 | 双鼠八块各 512 条独立试次做全字库身份对照，各取 64 条配对 OCF，共 512 张。两鼠在内层都选最大有限平滑强度，外层字形特异模板的条件对数似然低于不区分字形的时序模型。较高权重的真实时序融合相对 v11 变化 +0.464/+0.082 OCF（标准误 0.385/0.415），打乱字形时序的对应变化 +0.152/−0.218，Top-5 在八块均未上升。Kaggle CPU 用时 259 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-conditional-timing) |
| Task 3 前七名后验独立复核 | mice1 排除早前相关实验的 723 个评分或校准字形，从其他字抽 192 个，四块合计 768 张配对留出图。预先指定沿用 65% v11 与 35% 残差分类分数，在 v11 前 7 个候选中取首选。OCF 从 18.073 到 19.040（+0.967，标准误 0.486），四块变化 +0.288/+2.370/+0.395/+0.814。全字库混合为 18.974（+0.900，标准误 0.510），其中一块下降 0.995。两轮前 7 策略在八块有七块提高，正式候选仅对 mice1 生效。Kaggle GPU 用时 59 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-density-top7-recheck) |
| Task 2 原候选图时窗先验独立复核 | mice1 留出 session1/3，用 192 个互不重叠的评分字，合计 384 张。固定原 v11 图和先验权重时，OCF 从 20.8782 到 20.8704（−0.008，标准误 0.316），两个录制分别 +0.158/−0.173。两种方案均完成 2,748 个一对一匹配，先前的 +1.129 未复现。Kaggle GPU 用时 350 秒，先验不进入提交包。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task2-temporal-prior-recheck) |

### 候选 v12

用全量四次训练重复拟合 mice1 Task 3 的 35% 残差分类融合，仅在 v11 原前七个候选重排。v11 对照重建的 10,992 张预测均与原包逐字节一致。Kaggle GPU 只更新其中 7,422 行，四块分别 1,888/1,843/1,892/1,799 条。已核对 30,200 个官方 ID、顺序、PNG 完整性和 64×64 单通道模式。Kaggle 用时 40 秒，产物 SHA-256 为 `c4b3b9e89719cb9e7db8687be343d8653b54c533f979a1cc00e755315c756039`。选用 v12 作为北京时间 2026-11-26 08:00 开启的下一提交窗口候选。当前正式分数为 20.54607。 [Kaggle v12 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-25 第十三轮研究

Task 1 的正式诊断利用已有的同字试次分组，将每个查询的 20 次重复分为互不重叠的两组。带标签的重复次数对照则从独立留出训练字按已知标签合并多个录制，估计同字跨天对应已知时的闭集识别和 OCF。神经表征的重复性与图像重建品质分别由实测放电和留出参考图检验。 [小鼠 V1 的自然图像解码](https://www.nature.com/articles/s41467-020-14645-x)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 字形距离场读出 | mice1 新的外层留出字，session2/4 各 96 张配对图。相同字库和渲染下，方向轮廓加 Poisson 的 CPU 对照为 26.34。加入 25% 距离场分数在无权重和协方差加权时分别为 24.02 和 24.17（差值 −2.32/−2.17，按字标准误 1.19/1.26）。在 16 张图上使用参考图选前 16 候选的最佳 OCF 从 60.60 到 64.31（+3.71，标准误 3.42）。Kaggle CPU 用时 261 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-distance-geometry) |
| Task 1 正式 20 次重复与实际排名稳定性 | mice1 四个录制，共 1,360 组正式查询，每组做两个固定种子的 10/10 拆分。仅凭放电找回同组的 Top-1 为 66.80%，两半全部候选的平均 Jaccard 为 0.349。v8 首选一致率 29.30%，v11/v12 重排后为 21.80%。完整 20 次选中的字同时出现在两半候选池的比例，从 v8 的 97.24% 到 v11/v12 的 76.25%。两半共有候选的排序相关反而从 0.424 到 0.507。只使用官方分组和放电，指标衡量两半一致性，正式测试参考图未知。Kaggle GPU 用时 349 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-v11-formal-stability) |
| Task 1 字形特异前向残差校正 | mice1 用 2,044 字拟合 CPU 版 v8、384 个独立训练字拟合放电残差，并留出 320 个字。四块各 64 张配对图，v8 为 24.95，仅校正均值为 23.57（−1.38，标准误 0.81），加入共享字形残差为 23.50（−1.45，标准误 0.82），打乱字形残差为 23.72。四块的闭集身份 Top-1 在字形残差版本中均低于原前向模型。Kaggle CPU 用时 1,152 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-forward-residual) |
| Task 1 重复数与神经似然权重 | mice1 同一批 96 个留出字，依据已知对应，在四个训练录制中各取 1/2/4 次，合并为 4/8/16 次。逐查询标准化前乘以重复数的结果完全一致。将标准化后的神经似然相对权重提高 40%，配对 OCF 增量为 +1.70/+1.91/+1.72，16 次相对 4 次的额外增量为 +0.018（按字标准误 1.573）。三个重复数的 CPU 基础对照分别为 23.00/27.74/28.35。此轮未使用正式测试或精确重排器。Kaggle CPU 用时 247 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-repeat-weight) |
| Task 1 固定神经权重独立复核 | 同样的方法，每个重复数用另一批 128 个留出字并保持评分字相同。提高 40% 相对神经权重后的配对差值为 +0.510/+0.247/+1.812，按字标准误 0.926/0.580/1.019。16 次相对 4 次的额外增量 +1.303（标准误 1.385）。两批的 16 次增量约 1.7–1.8，但重复次数与增量的交互证据弱。16 次由已知同字的四个训练录制组成。Kaggle CPU 用时 250 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-repeat-weight-recheck) |
| Task 1 实测 4/8/16 次的闭集参照 | 两个不重合的 320 字留出组，各鼠在固定 320 张真实候选图内评估。每个查询按已知同字对应，从四个训练录制分别取 1/2/4 次试次。时窗前向匹配在 4/8/16 次的 Top-1 为 mice1 30.0%/43.4%/67.0%、mice2 18.6%/31.25%/40.3%。首选 OCF 约 41.06/53.81/70.96 和 31.96/42.22/49.42。16 次时前 32 名身份召回为 99.4%/93.9%，32 张图用参考图选出的最佳 OCF 约 100.00/93.62。使用跨天已知对应和闭集真实字图库，正式查询未参与。Kaggle GPU 用时 209 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-repeat-ceiling) |
| Task 1 公开常用字表覆盖 | 用拟合训练图像在全字体库做不带字表偏好的稀疏一对一匹配，以独立公开字体审核置信筛选，接受的 877 个匹配中 844 个码位正确。通用规范汉字表一级在 Noto 有 3,499 个候选，移除与拟合字高置信匹配的 2,081 个码位后保留 1,418 个。320 张新留出训练图中 317 张保留全库最近字形、319 张保留近似字形。309 个高置信匹配中覆盖 308 个。相同大小的词频与形态对照达到相同覆盖，按形态分层打乱的对照只覆盖 149 个高置信匹配。此项只衡量参考图的视觉查找。Kaggle CPU 用时 113 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-repertoire-coverage) · [教育部一级字表](https://www.moe.gov.cn/jyb_xwfb/moe_2082/siju/yywzxxgls/ztzl_yywzgfbz/201306/t20130601_186002.html) · [Unicode 字段](https://www.unicode.org/reports/tr38/#kTGH) |
| Task 1 常用字表的放电解码复核 | mice1 新留出 320 个字，四块各 128 张配对 OCF。在相同 CPU 轮廓加 Poisson 读出中，通用规范一级去除高置信已见字后有 1,177 个候选，现有词频等规模字库与其重合 1,171 个。全库及未去除已见字的一级字表均为 20.8007。硬限制、软限制及等规模词频残余库各改 4/512 个首选，均为 20.7821（变化 −0.0186，按字标准误 0.0173）。Kaggle CPU 用时 208 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-repertoire-ocf) |
| Task 1 拆半提名与加深全量候选 | mice1 在独立整字留出中每块评分 128 张，合计 512 张，四次重复分成 2/2。冻结实际 v11 排序器，用完整四次的特征给原候选与新增候选评分。两种扩展各新增 2,047 个候选，逐行预算相同。原 v11 为 31.033，拆半提名为 31.488（+0.456，标准误 0.592），加深全量候选为 31.766（+0.734，标准误 0.569）。加深候选在四块均提高，两种扩展的差为 −0.278（标准误 0.555）。40 张图用参考图选出的最佳候选 OCF 为原池 52.86、拆半 60.90、全量加深 59.65。原池加拆半分歧惩罚时首选变化 −0.137。Kaggle GPU 用时 375 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-half-proposal) |

本轮完成九项实验。两个固定闭集的留出划分都显示，增加实测试次提高了匹配水平。公开字库内，限制为常用字几乎不改变现有首选，扩充原候选池则扩大了参考图挑选时的余量。现有 v11 重排在完整试次加深候选的首轮对照中改善 +0.734 OCF，仍需跨字集验证。候选选用 v12，等待北京时间 2026-11-26 08:00 的提交窗口。 [Kaggle v12 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3)

## 2026-11-25 第十四轮研究

同一个 Task 1 录制模型在两种候选图上预测时，固定放电试次和日间分数校准。320 个真实留出训练字与公开字体做一对一视觉配对，再给两种候选图加入同样的近形字。全字库另从所有固定公开候选中直接用放电排序。试验使用已知训练字对应汇合四个录制，在同一批字上核对 4/8/16 次实测试次的结果。视觉匹配和重建评估的目标不同，这里用官方 OCF 衡量选中输出的质量。 [视觉皮层重建研究](https://elifesciences.org/articles/105081)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 真实图、同字渲染和近形候选 | mice1 320 个新留出字、96 个配对 OCF 字。同一前向模型在 16 次时，320 张真实候选首选为 76.96 OCF。保持该首选，只用匹配的 Noto 渲染输出，得 62.85（−14.11，按字标准误 1.11）。用 Noto 候选图计算放电似然并重新选字时，320 候选为 62.05，增至 1,344/4,416 后分别为 51.41/44.96，完整公开字库为 32.67。全库对应渲染的 Top-1 为 15.63%，320 张真实候选为 70.00%。近形候选按参考图构造，只用于受控诊断。Kaggle CPU 用时 355 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-gallery-gap) |
| Task 1 加深全量候选独立复核 | mice1 两组互不重叠的留出字，每组 160 字在四个录制使用每字 4 次重复，合计 1,280 张配对 OCF。原 v11 在两组上分别为 31.686 和 30.738，每行多加入 4 个候选后分别为 31.748 和 31.167。合并变化为 +0.245，按字标准误 0.253，八块中五块提高，session1 在两组均下降。使用原拆半预算或仅使用完整放电构造扩展的首选完全一致。Kaggle GPU 用时 646 秒，提交候选沿用 v12。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-deeper-replication) |
| Task 1 轮廓候选的笔画像素改写 | mice1 整字留出，四个录制各 128 张配对 OCF。固定候选、局部注册和 8% 的像素翻转预算，原候选 OCF 为 24.696，字库改写为 22.122，放电引导改写为 22.491，跨查询打乱放电空间图为 22.447。真实相对打乱的差值为 +0.043，按字标准误 0.078。改写降低原始 OCF。Kaggle CPU 用时 787 秒，含 1,979 个独立精确图对。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-stroke-edits) |
| Task 1 近形候选的放电率校正 | mice1 四个录制共 512 张配对留出图，固定 v8 轮廓与先验。未校正为 25.162，Poisson 生成式校正为 25.154（−0.008，按字标准误 0.199），OCF 配对训练为 25.317（+0.155，标准误 0.300）。独立校准字的两个放电半组均给出正确偏好的比例分别为 56.60%/57.55%/56.77%。配对校准用了 3,484 个精确图对，选中的图对平均差 38.5 OCF，对极近形字的证据有限。Kaggle GPU 用时 218 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-rendered-pair-rates) |

四项实验均到达 Kaggle 完成状态。受控对照量化了公开字体渲染与近形竞争的各自损失，两项重排和一项像素改写都未提供足以更新提交包的增益。保留已核验的 v12，当前正式 OCF 为 20.54607。下一提交窗口于北京时间 2026-11-26 08:00 开启。 [Kaggle v12 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第十五轮研究

Task 1 在同一批校准字、同样的每字配对数和同一个前向校正器上，比较较窄（4–18 OCF）和较宽（至少 25 OCF）的候选差值。候选来自图像轮廓近邻和冻结的 v8 放电排名。校准使用训练字的参考图，外层留出从完整公开字库仅凭放电提名。窄差值标签打乱是拟合对照。 [难负例学习](https://openaccess.thecvf.com/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) · [成对排序](https://papers.neurips.cc/paper_files/paper/2006/hash/af44c4c56f385c43f2529f9b1b018f6a-Abstract.html)

Task 3 的细时标同步对照保留每单元 25 ms 计数以及整群体每毫秒的计数，同时按拟合的单元放电时间分布重新随机化单位标签。未入选单元用来估计共同潜伏期。32 个单元对从拟合字确定，在外层 320 个留出字上做 2/2 拆半。 [条件时序扰动](https://pubmed.ncbi.nlm.nih.gov/22031767/) · [小鼠 V1 同步实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC4153809/)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 给定计数边际的跨单元同步 | mice1 四录制各 320 个留出字。相对条件随机化，错开试次校正后的同字复现增量为 +0.00935（按字标准误 0.00584），四块为 +0.00289/+0.01078/+0.02036/+0.00336。校正前为 +0.01390（标准误 0.00574）。保边际样本实际改标 88.3%–89.1%，符合条件的交换接受率 97.1%–97.4%。Kaggle CPU 用时 18.3 秒，未使用字图或计算 OCF。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-conditional-synchrony) |
| Task 3 多尺度时窗计数的条件精度 | mice1 四录制、192 个新留出字，共 768 张配对图。给定父窗总数及 v12 正向均值，在 v12 前七名加固定、字形条件或按均值及父窗计数分层打乱的精度修正。v12 为 18.507，三组为 18.786/18.893/18.934。字形条件相对固定为 +0.107，标准误 0.125，相对打乱为 −0.041，标准误 0.191。已知正确字的条件计数损失相对固定每试次降 0.017–0.019 nats，但全库 Top-32 在四块均低于 v12。Kaggle GPU 用时 49.6 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-allocation-precision) |
| Task 2 以字形估计目标日放电残差 | mice1 留出 session1/3，各 128 张配对 OCF。目标日四次重复分为 2/2，四折排除被预测查询的两半放电，以匿名软对应拟合字形到目标日模板的修正。固定 v11 候选边及 OCF 排序后，v11 为 20.350，形态条件为 20.121（−0.229，按字标准误 0.365），按墨量和轮廓强度打乱为 20.157（−0.192，标准误 0.166）。三种方案都给出 2,748 个唯一匹配。两次留出的拆半首选一致率为 2.26%/1.46%。形态修正的边效用有 60.5%/25.7% 触及限值，两个录制的完整字库 Top-5 同时下降。Kaggle GPU 用时 357 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-morphology-adapter) |
| Task 2 等试次的同日／跨日匹配 | 两鼠各 2,236 个留出字、两套 2/2 拆分及 12 个有向录制对。固定同一查询、图库大小与测距，仅切换同日或跨日的两试次模板。mice1 Top-32 为 15.122%/11.511%（相差 3.611 个百分点，按字标准误 0.329），用 512 对已知字拟合低秩旋转仅恢复 0.626 个百分点，打乱配对为 −0.220。mice2 同日／跨日为 8.637%/8.646%，已知配对旋转增益 0.186 个百分点。mice2 session4 两试次噪声约为其他天的 3.7 倍。Kaggle CPU 用时 13.0 秒，未计算 OCF，已知字映射只用于衡量可恢复差距。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-repeat-day-gap) |
| Task 1 窄差值和宽差值的成对排序 | 317 个校准字的两组各有 1,856 对，平均 OCF 差分别为 9.96 和 42.98。在 320 个整字留出中抽 128 字，四录制合计 512 张外层配对图。全字库 v8 为 25.432，窄差值训练为 25.336（−0.096，按字标准误 0.539），宽差值为 25.607（+0.176，标准误 0.573），打乱窄差值标签为 23.954（−1.478，标准误 0.483）。窄差值相对宽差值为 −0.272（标准误 0.601），相同 24 候选预算下用参考图选最优比纯加深候选低 2.330（标准误 1.102，24 字）。候选间视觉距离及两端绝对 OCF 可随分组变化。Kaggle GPU 用时 442 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-pair-difficulty) |

五项实验均已完成。目标日已知的低秩对齐只恢复 mice1 等试次同日／跨日差距的约 17%，近形配对和单次放电精度修正的首选 OCF 增益不足以更新提交。v12 仍是已核验的候选；当前 UTC 日名额已用完，下一次可正式提交为北京时间 2026-11-26 08:00。 [Kaggle v12 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第十六轮研究

Task 1 的重复权重在训练字拟合的 64 维白化放电空间内估计。试次与本组其余试次的典型距离超过组内中位距离时，用固定衰减降权，未归一化权重下限为 0.4。组内随机置换权重保留有效试次数。正式 20 次数据只用官方组内对应关系，对两个固定的 10/10 划分做拆半识别。字图 OCF 单独用训练留出字评估，以此检验放电更可复现是否改善了图像。 [稳健位置估计](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full) · [小鼠视皮层群体放电](https://pmc.ncbi.nlm.nih.gov/articles/PMC6948932/)

同轮 Task 1 直接用官方 SpikeDenoiser 做匹配训练对照。两臂在四个录制合并的数据上使用相同原始试次、起始权重和扩散噪声，混合臂从同字同日无放回补齐 1、2 或 4 次重复并取平均。取同一批整字留出的真实放电，固定 64 步采样和 0.5 二值化，同时保留同样输入的全字库检索。现代鼠视觉扩散研究支持探索这一方向，实测对照以本比赛的放电和官方 OCF 为准。 [官方 DDPM](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data) · [Sensorium-Viz](https://github.com/GuanLab/sensorium-viz)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 固定字形混合的图像重建 | 官方训练图库中构造 28 个近形双峰或三峰、8 个远形双峰和 8 个单峰。每个近形组合提名 2–10 项真实差分笔画，编辑总预算为原墨量 8%。可微轮廓代理驱动的贪心在 28 个近形组合中均保留硬代表字，精确混合期望 OCF 为 72.325，IoU 0.776。像素投票及旧稀疏共识的 OCF 差分别为 −5.163 和 −1.026（标准误 1.315/0.443）。8 个单峰均得 100。此对照只支持判断当前小预算代理贪心，混合权重由实验预设。Kaggle CPU 用时 177 秒，共 793 个精确图对。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-preimage-geometry) |
| Task 1 对异常重复降权 | 正式数据的 10/10 拆半中，mice1/mice2 的余弦同组首选率由 56.76%/27.68% 到 57.21%/28.24%，打乱同一组权重为 56.65%/27.72%。高斯打分的首选分别变化 +0.29/−0.33 个百分点，仍使用等权 10 次的噪声标尺，两组拆分复用原始试次。训练留出每字 4 次共 448 张图，组内有效次数约 3.98，开放字库的 OCF 由 24.669 到 24.369（−0.300，按字标准误 0.225），打乱为 24.676。已知同字的四日 16 次融合在 112 张图上为 29.31/30.16/30.31。正式诊断衡量组内放电再识别。Kaggle CPU 用时 284 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-robust-repeats) |
| Task 1 二值字形的局部字体校正 | 使用 768 个高置信自动配对训练约 2,400 参数的共享边缘模型，固定原有全局渲染。代理目标从 0.42345 降到 0.41507，二值化后的 128 个冻结视觉近邻及 384 张冻结放电首选均没有像素变化，OCF 分别保持 80.659 和 22.823。结果覆盖当前阈值，未比较真正发生编辑时的改动质量。Kaggle CPU 用时 351 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-local-font-style) · [局部风格生成](https://github.com/clovaai/lffont) |
| Task 1 局部字体校正的小预算复核 | 64 个独立校准字，强制修改原墨量 0.5%/1%/2% 的边缘像素，平均改 3.55/7.56/15.56 像素。原渲染为 80.233，三种编辑相对原渲染变化为 −0.175/−0.340/−1.369 OCF（按字标准误 0.120/0.151/0.233）。相对同增删数的形态控制则分别为 +0.581/+1.440/+1.952。校准选零编辑，冻结放电首选的复核未执行。Kaggle CPU 用时 193 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-local-font-small-edits) |
| Task 3 跨录制实测度量 | mice1 四录制各 192 张留出单试次。度量拟合排除目标录制与评分字，跨日和同日使用同一结构、负例数与相同实测支持库。固定 v12 前七名融合 25% 时，原 v12 为 18.463，冻结／同日／跨日的变化为 −0.049/+0.004/+0.116 OCF（按字标准误 0.286/0.278/0.284）。跨日相对同日变化 +0.112±0.177，独立内层选择 15% 的跨日版本在外层变化 −0.084±0.261。跨日融合的全库前 32 名在四块均上升，冻结融合在四块均更高。Kaggle GPU 用时 120 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-measured-metric) |
| Task 1 跨录制共享字集与列背景校正 | mice1 四个录制，384 个独立背景字、320 个整字留出。每个目标只用其他三块，分别筛选固定 320 列，在相同 v11 候选边上加分。640 张配对图，v11 为 27.188，逐字形背景校正为 27.173（−0.014±0.011），全库共享背景为 26.970，分层打乱为 27.134，公开先验为 27.200。事后用真实训练图求字库匹配代理，校正、原始、打乱、公开先验的支持召回为 2.03%/3.36%/1.25%/20.23%。近似字库代理和每字 4 次重复限制对正式 20 次的外推。Kaggle GPU 用时 342 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-shared-support) |
| Task 1 官方 DDPM 的重复数匹配训练 | mice1 正式模型架构，四录制各 96 张整字留出图，两个训练臂均完成 3,200 次更新。给定 4 次重复，单试次训练和混合 1/2/4 次训练的二值 OCF 为 11.018/11.034（+0.015±0.062），灰度为 11.356/11.269。共用放电的全 Noto 检索在 1/4 次时为 19.610/23.998。每录制 24 字的 4 次混合臂真实／打乱条件为 10.948/10.426（差 +0.522±0.196）。T4 训练 242 秒，全实验 387 秒。训练轮数和同日 20 次的局限仍在，这次比较支持在已测范围内保留字库检索。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-official-ddpm-repeat) |

七项 Kaggle 实验均已完成。用实测放电辅助缩小字库、跨录制度量、局部字体校正和短训 DDPM 均未提供足够 OCF 增益来更新 v12。当前正式分数为 20.54607，下一次可正式提交为北京时间 2026-11-26 08:00。 [Kaggle v12 候选](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第十七轮研究

Task 2 将潜在字形身份和输出图像的选择放在同一份放电评分上比较。按查询归一化的后验与整块均衡的后验共用温度，后者根据每个真实字在录制中只出现一次的设计作校正。固定 v11 的候选边，在原来的分数上分别加入身份概率和预期方向轮廓相似度，并核对输出保留或放开一对一约束的效果。 [Sinkhorn](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) · [结构化决策的期望损失](https://www.jmlr.org/beta/papers/v21/20-097.html)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 冻结编码器的局部逆向形变 | mice1 四录制各 128 个新留出字。前两次放电选字并优化局部平滑流，后两次核验，再反向交换。真实放电优化相对原图与打乱分别提高独立似然 0.324±0.049 和 0.384±0.065 nats/试次。使用全部四次的同一 v11 首选测 512 张配对 OCF，原图 29.995、真实形变 28.294（−1.701±0.139）、打乱 28.020。真实峰值位移 1.028 像素、绝对墨量改 0.867%，二值轮廓平均改 122 像素。Kaggle GPU 用时 414 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-inverse-warp) · [输入优化研究](https://elifesciences.org/articles/105081) |
| Task 1 公开字形锚定的残差 DDPM | mice1 四录制各 96 张整字留出图。训练残差限制在原边缘 3 像素及原墨量 12% 内，内层放电首选与真图平均相差 831 像素，这一约束平均保留 96.9 个训练编辑。外层冻结同一个开放字库首选。原锚 OCF 为 22.222，只看字形、同日打乱放电与真实放电为 19.188/19.196/19.202。真实相对原图 −3.020±0.340，相对打乱 +0.005±0.024。384 张真实条件均编辑，平均 85.64 像素，21 张在截断前超预算，无空图。Kaggle T4 完成 6,400 更新，用时 372 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-anchored-ddpm) · [残差扩散研究](https://arxiv.org/abs/2308.13712) |
| Task 1 局部形变只用于排序 | mice1 另一组 320 个整字留出，四录制 512 张配对图。每行固定 v11 的八个候选，双向拆半拟合和评估形变，使用给定群体计数的空间似然，二值字形保持同一墨量，输出使用原图。v11 为 32.567，真实形变重排 32.291（−0.276±0.280），打乱 32.210。真实相对打乱 +0.081±0.271。形变相对零位移提高所有候选的独立似然 0.466±0.048 nats/试次，原首选只提高 0.139±0.050。候选提名使用全部四次，形变增量才使用拆半。Kaggle GPU 用时 400 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-warp-scoring) |
| Task 3 多重复教师的形态邻域蒸馏 | mice1 四录制，共 768 张配对真实单试次图。源训练的单试次学生、四重复教师和候选模板使用互斥放电，学生拟合排除目标录制。原 v12 为 18.401，固定前七内硬标签、纯形态软标签、放电调整的软标签分别为 18.187/18.288/18.304。多重复教师相对纯形态 +0.016±0.024，相对 v12 −0.097±0.305。25 选 1 的源诊断中四重复为 69.2%–71.4%，同模板单次为 42.0%–44.2%。前两版的单试次轴引起广播和内存错误，第三版修复并完成全部四日，Kaggle T4 用时 174 秒。 [Kaggle v3](https://www.kaggle.com/code/ccjjllt/siclink-task3-distilled-teacher) · [知识蒸馏](https://arxiv.org/abs/1503.02531) |
| Task 2 均衡身份后验与轮廓效用指派 | mice1 两个完整留出录制，使用相同的 v11 边与 128 个评分字。整块均衡使全库 Top-32 从 53.35%/56.73% 提高到 54.66%/59.10%，身份负对数损失也下降。256 张图上，v11 为 34.757，按行轮廓期望、均衡身份、均衡轮廓期望分别为 33.076/34.536/33.517。均衡轮廓相对 v11 为 −1.241±0.922，相对按行轮廓为 +0.441±0.634。取消输出一对一后，固定 v11 分数降至 30.345。Kaggle CPU 用时 3,429 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-posterior-assignment) |

五项研究对照均取得完整结果。形变改善放电似然，均衡后验改善真字召回，蒸馏教师在源录制可靠识别真字，这些信号均未在本轮转为稳定的选图 OCF 增益。保留 v12，当前正式分数为 20.54607。北京时间 2026-11-26 08:00 为下一次可提交时间。 [Kaggle v12](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第十八轮研究

Task 2 将两只鼠的目标录制分别打乱，使用其他已标注天的同字训练共享响应坐标。由放电推断两鼠查询行的软对应，将 mice1 的独立字库证据传给 mice2，再保持一对一输出。同行打乱是配对对照，已知对应的上限只通过训练留出标签诊断。跨神经群体的共享响应空间提供这一建模思路，模型准确度由本赛题留出的录制检验。 [多视图表示学习](https://arxiv.org/abs/1702.02519) · [视觉刺激的神经群体表征](https://pmc.ncbi.nlm.nih.gov/articles/PMC10823231/)

Task 1 用同一组正式策略候选核对精确 OCF 校准字的规模，比较原 96 字头、互不重叠的新 96 字以及嵌套的 192/384 字。Task 3 在固定的 v12 前七候选内学习整行选择，将候选的期望 OCF 按潜在参考后验名次分组。两项都以未参与拟合的图像配对效果为主要依据。 [结构化决策的期望损失](https://www.jmlr.org/beta/papers/v21/20-097.html)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 2 给定放电计数边际的试次聚集 | mice1 留出 session2/4，每块全库 2,748 字。给定单元窗计数和群体逐试次计数后，真实聚集残差相对近计数错配和固定边际随机化使合并身份 NLL 各低约 0.023，Top-32 高 0.855/0.946 个百分点。相对原 v9，合并 NLL 增加 0.00787±0.00381 nats，Top-32 下降 1.328±0.351 个百分点。未进一步测 OCF。首版修复单元点积广播后，第 2 版 Kaggle CPU 用时 69 秒。 [Kaggle v2](https://www.kaggle.com/code/ccjjllt/siclink-task2-repeat-collision) · [试次放电变异研究](https://journals.physiology.org/doi/full/10.1152/jn.00382.2025) |
| Task 3 后验分秩与精确 OCF 选择 | mice1 四录制共 768 张配对真实单次图。用 13,814 个图库精确图对拟合匹配效用，890 个独立图对核验，再用 6,709 个真实单试次候选图对校准。在同一 v12 前七名内，v12 为 18.770，单温度效用为 18.887，固定 20% 的分秩效用为 19.106（+0.336±0.268），逐录制 +0.306/+0.826/−0.388/+0.601。学习分秩选择为 18.859（+0.090±0.291），相对同模型用轮廓代理为 +0.238±0.107。真图从当前七个候选选优为 28.925，四块有 45%–61% 的真实字排在第 129 名及之后。Kaggle GPU 用时 650 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-rankband-ocf) |
| Task 2 跨鼠软对应与一对一选字 | 两鼠独立留出 session1/3，目标为 mice2。正式策略 v9 在 320 张配对图上为 20.866，放电推断跨鼠对应后为 20.799（−0.067±0.662），打乱为 17.418。真实软对应相对打乱为 +3.381±1.226。使用留出标签指定正确跨鼠对应的诊断为 23.728（+2.862±1.069），可用软对应的身份首选只有 0.44%/0.47%，虽然后者使正式字库 Top-32 在两块提高 3.75/4.84 个百分点。Kaggle CPU 用时 542 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-peer-mouse) |
| Task 1 精确 OCF 排序的标注规模 | mice1 四录制 768 张新留出配对图，全部输出在同一开放字库候选图上选取。匹配编码折及 640 次更新时，新 96/192/384 字的 OCF 为 31.507/32.159/32.738，384 相对匹配的 96 为 +1.231±0.491。完整原 v11 为 32.344，384 相对原 v11 为 +0.394±0.578，逐录制 −1.446/+1.258/−0.073/+1.836，去除 session4 后平均 −0.087。共享校准 480 字、四录制，共 29,720 个去重后的官方精确图对。Kaggle T4 用时 1,054 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-metric-scale) |

四项对照均完成。扩大精确 OCF 标注在匹配实验内提高了排序，完整原 v11 对照的平均优势偏小且随录制变化。Task 2 跨鼠证据和 Task 3 分秩效用尚不足以更新候选。保留 v12，正式分数为 20.54607。北京时间 2026-11-26 08:00 开启下一次提交名额。 [Kaggle v12](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v12-task3) · [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第十九轮研究

Task 1 的交叉拟合校准分四折收集每折 96 字的精确图对，各折保留原 v11 的前向训练字数与同一噪声校准。另用训练中已知同字的四录制独立证据，拟合两种固定神经权重的 OCF 差，比较只使用重复数、加入半组一致性及打乱一致性的选图。开放字库的正式 20 次输入未参与这两个对照。

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 3 固定分秩效用的独立复核 | mice1 四录制共 1,024 张新的真实单次配对图，评分字与上轮互斥，新旧内外留出试次逐字核对无重合。保留 v12 前七候选和固定 20% 权重。v12 为 18.639，分秩为 18.894（+0.255±0.210），同权重单温度效用为 18.946（+0.306±0.212），分秩相对单温度 −0.052±0.045。两次分秩共 448 个评分字、1,792 张配对图，按字加权约 +0.290，合并标准误约 0.166，录制仍是同四天。Kaggle T4 用时 364 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-rankband-recheck) |
| Task 2 跨鼠提名的等候选预算复核 | mice2 新留出 session2/4，各 160 张配对图。每行保留原 v9 解的边并给四个新槽。完整 v9 为 19.099，同预算本鼠深挖为 17.942，真实跨鼠提名为 19.585（相对 v9 +0.486±0.480），分层打乱为 19.920。真实相对打乱 −0.334±0.748。稀疏裸目标与原 v9 两天逐行完全一致，所有方案均 2,748 个唯一输出。困难负例过采样后，校准只保留了字库重叠项，原始 GCCA 特征系数为零，结果仅覆盖本次组合。Kaggle CPU 用时 332 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task2-peer-nomination) |
| Task 1 跨录制一致性与重复数组合 | mice1 用 144 字拟合选择，另 144 个新字做配对 OCF。保持字库、图像和权重备选固定，Poisson 信号乘 1.4 在合并 4/8/16 次实测重复时分别使 OCF 变化 −0.448±0.774、+2.527±1.021、+3.401±1.144。在 8 和 16 次时，按重复数、真实一致性及打乱一致性均逐行选择较高权重。16 次的原方案为 31.588，较高权重为 34.990，参考图从两者选优为 35.092。此为已知同字的四天合并，不等同正式单日 20 次。Kaggle CPU 用时 407 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-repeat-agreement) |
| Task 1 四折精确 OCF 校准复核 | mice1 另一组 320 字，四录制共 768 张配对图。每折用 1,948 字拟合前向读出，保留同一 384 字噪声校准，四个不重叠的 96 字折训练校准头，同为 640 次更新。共享同一外层候选、渲染与锚权重 4 时，原 v11 28.896，四折 30.684（+1.788±0.661）。相对另一预指定单折 +2.033±0.622。逐录制增益 +3.369/+0.550/+1.275/+1.957。共有 23,117 个去重后精确图对。四次放电留出验证未使用正式输入。Kaggle GPU 用时 896 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-crossfit-metric-scale) |
| Task 1 较高 Poisson 权重迁移到 v11 | 复用 CPU 试验同一批 144 字和重复，冻结原精确排序。单日 4 次、576 张配对图的 v11 为 33.207，固定候选与同预算重提名的变化为 +0.389±0.442 和 +0.623±0.504。跨日已知同字的 16 次代理在 144 张图上为 47.584，固定和同预算重提名提高 +1.202±1.047、+1.629±1.147，只重提名变化 −0.024±0.024。16 次和四次的重提名增益差为 +1.005±1.212。此为同字模型迁移，未采用到 v13。Kaggle GPU 用时 337 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-v11-repeat-weight-transfer) |

### v13 提交候选

用全量 2,748 个训练字构造四折 OCF 排序头，并以 v12 为来源计算 mice1 的正式 Task 1。v13 首包的 30,200 个预测完整，非目标行 28,840 条均与 v12 逐字节一致。384 条变化中有 54 条的旧输出未能从重跑的旧策略逐像素复现，因此首包只保留作研究对照。首次重跑在评分前已有 7,858 条校准候选边，旧版有 7,855 条。第二版直接调用旧校准函数并保持其精确评分顺序，只应用到已复现旧像素且改变首选的查询。 [Kaggle 生成记录](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v13-task1-crossfit)

第二版在 Kaggle T4 用时 1,162 秒。旧策略的 PNG 逐字节复现 1,259/1,360 条，未复现的 101 条全部继承 v12，包含原本建议改变的 62 条。更新 371 条 mice1 Task 1，四块分别 88/84/101/98 条。剩余 29,829 条完全一致，30,200 个官方 ID 顺序、ZIP CRC、64×64 单通道 PNG 及更新 ID 清单均已复查。第二版 SHA-256 为 `f478a5c0e1a68b1c0dfd5e016432861ada3b0f2c1433d0a6fb3b0e761e8ce540`，作为下一次提交候选。北京时间 2026-11-26 08:00 前正式名额仍未开放，官方分数为 20.54607。 [Kaggle v13 第二版](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v13-task1-crossfit?scriptVersionId=2) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第二十轮研究

Task 1 在新整字留出上检验两个问题。mice1 将 384 字交叉拟合扩大到 768 字，冻结校准头更新次数、外层候选图、渲染与分数锚定。mice2 在同一批外层字上比较现代候选池中的单折和四折，并与当前包逐录制采用的实际策略作配对。

Task 2 在 mice2 的两个完整录制上，从冻结的 v9 后验出发，用其他查询的软字形对应更新一对一指派。分别用来源和目标的真实拆半放电估计相似关系，去掉自连接。按墨量和放电响应幅度分层打乱图库对应，并保留只用于诊断的真实邻居对应。这里借鉴融合 Gromov–Wasserstein 中同时利用单点分数和成对关系的思想，只使用一次固定权重更新。 [结构化最优传输](https://proceedings.mlr.press/v97/titouan19a.html) · [交叉验证表征几何](https://elifesciences.org/articles/82566)

Task 3 在新留出字上保留 v12 的七个候选，分别用固定九个空位比较 v11 加深、v12 残差排名和精确轮廓效用的边际覆盖。排序直接使用冻结 v12 分数或同一个固定权重的单温度效用。另对预先指定的子集使用参考图，从每个固定候选池选出 OCF 最高图，衡量扩充候选增加的余量。

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 2 软对应传播跨天放电几何 | mice2 留出 session1/3，384 张配对 OCF，192 个独立字。v9 为 21.218，真实几何为 21.770（+0.552±0.565），逐录制 −0.089/+1.192。分层打乱为 21.418，真实相对打乱 +0.352±0.505。以真实邻居对应替代软对应时为 21.502。所有方法在每块输出 2,748 个唯一字形。Kaggle CPU 用时 393 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-relational-geometry) |
| Task 3 残差候选与精确效用的边际覆盖 | mice1 四录制共 768 张新留出单试次图。冻结 v12 排序时，原七候选为 18.933。加深 v11 得 19.106，同残差库按 v12、单候选期望效用及边际覆盖提名分别为 19.220、19.643、19.509。单候选效用相对原七为 +0.710±0.352。320 张用真图选优的同库 v12／边际上限为 38.512／38.550，差 +0.038±0.809。边际覆盖相对同库 v12 的实际选图增益 +0.289±0.155，共用 20% 效用二级分数后为 +0.052±0.060。Kaggle T4 用时 656 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-utility-coverage) |
| Task 2 逐单元跨日稳定时序成分 | mice2 留出 session2/4，320 张配对图。沿用早期跨日白化方法，每单元只保留一个轴并以固定 10% 权重与 v9 融合。v9 为 20.961，按每次试次整窗计数去率后为 20.695（−0.265±0.822）。同结构保留计数时为 22.185（+1.224±0.784），两录制分别 +1.985/+0.463，Top-32 各增 1.20/0.69 个百分点。首版修复了基线的梯度作用域，第二版 Kaggle CPU 用时 414 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-temporal-components?scriptVersionId=2) · [跨天时序编码研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12322180/) |
| Task 1 四折与八折同预算校准 | mice1 四录制各 320 个新整字，共 1,280 张配对图。原单折 32.715，四折 33.536（+0.821±0.379），八折 33.128（相对四折 −0.408±0.285）。八折相对四折的逐录制变化为 −0.882/−0.970/+0.537/−0.318，删除任一录制后的合并估计仍为负。八折 47,301 个去重精确校准图对，前四折、候选、渲染及 640 次更新与四折一致。Kaggle T4 用时 1,582 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-eight-fold-metric-scale) |
| Task 2 保留计数时序成分的独立复核 | mice2 改留 session1/3，各 256 张新图。原 v9 为 20.776，保留计数的一轴为 20.743（−0.033±0.405），逐日 +0.068/−0.135。Top-32 分别提高 1.71/2.04 个百分点，身份 NLL 降低 0.0529/0.0872 nats。两轮合计 832 张图，按 416 个评分字加权的 OCF 变化为 +0.450±0.392。此轮未测匹配结构的仅总计数参照。Kaggle CPU 用时 628 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task2-temporal-components-recheck) |
| Task 1 mice2 四折与现有逐录制策略 | 320 个独立留出字、四录制、1,280 张配对图。现行方案 session1 用现代单折，session2–4 用旧 v6。现行 23.151，现代四折 24.754（+1.603±0.437），逐录制 +1.362/+2.910/+1.384/+0.756。保持现代模型、候选、渲染及锚定一致，四折相对现代单折 +0.681±0.391。只看旧 v6 的三块，四折相对现行 +1.683±0.555。Kaggle T4 用时 1,292 秒。各图使用四次真实重复，正式输入每组有 20 次。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-incumbent-crossfit) |
| Task 3 单候选精确效用提名复核 | mice1 四录制、320 个未做过 OCF 评分的字、1,280 个新外层单试次。冻结 v12 选图，原七候选 18.559、同一残差库按 v12 排序补九个为 18.621、按候选期望效用补九个为 18.572。两种等预算扩池差 −0.050±0.153，期望效用提名较原七 +0.013±0.170，前轮实际增益未复现。192 张用真图选优为 27.971/33.476/33.862。新外层试次与前三轮评分试次互斥，其中 649 个曾仅参与早轮温度校准。Kaggle T4 用时 437 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-utility-nomination-recheck) |
| Task 2 一对一指派的差分循环 | mice2 session1/3 的固定 v9 与 10% 保计数时序指派差完全复现 588/654 条，均有 2,748 个不同输出。变化只分解成 7/9 个循环，最大循环各包含 342/562 条（58.2%/85.9%），按循环长度的有效个数为 2.36/1.34。同长度随机选择只能改变少量小环，无法构造有代表性的等规模对照。此次测量止于结构，未计算 OCF。Kaggle CPU 用时 170 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-temporal-cycles) |
| Task 1 在 v13 四折上复核较高 Poisson 权重 | mice1 另 320 个整字，冻结生产四折结构、渲染和锚。等预算重提名并将 Poisson 权重乘 1.4 时，单日四次相对 32.629 提高 +0.464±0.240，四日各取一次相对 31.608 提高 +0.768±0.268，四日合并 16 次相对 50.845 提高 +1.364±0.513。16 次只重提名为 +1.025±0.451，固定候选为 +0.339±0.249。保持同样四日覆盖时，16 减四次的额外增益为 +0.595±0.578。代理依赖已知跨日同字，未使用正式同日 20 次。Kaggle T4 用时 1,118 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-v13-repeat-weight-recheck) |
| Task 1 独立拆半放电与选图 OCF | mice1 320 个新整字在四个录制双向做 2/2 拆半，固定同预算 v13 与 1.4× 权重。2,560 个选择中改变 233 个、涉及 171 个字。独立感受野的真图正控相对轮廓近邻为 +5.935±0.323 nats/试次。按相近计数错配扣除背景后，独立放电差与 OCF 差的符号一致率为 53.07%±3.50 个百分点。全样本 OCF 变化 +0.056±0.104。拆半指标未提供足够证据用于选包。Kaggle T4 用时 1,158 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-poisson-bridge) |

### v14 提交候选

用全部 2,748 个训练字对 mice2 的四个录制拟合四折排序器，正式推理使用每组 20 次实测放电。从 v13 第二版出发，重新生成旧策略后逐像素复现 1,153/1,360 条。无法复现的 207 条保留原图，其中 129 条本来会更换首选。实际更新 637 条 mice2 Task 1，逐录制 158/171/152/156 条。其余 29,563 条逐字节一致。全部 30,200 条的官方顺序、ZIP CRC、64×64 单通道 PNG 与更新清单均核对。Kaggle T4 用时 1,241 秒。选用 SHA-256 为 `1da6c1c7f6ba4c2dbf388921569baf9fb061a55ff78a822e4823079e5570c1c6` 的包，保留 v13 的 mice1 四折输出。 [Kaggle v14](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v14-mice2-crossfit)

2026-11-26 08:49（北京时间）以 `ccjjllt` 提交 v14，编号 `58918204`。09:06 核对评分完成，公开 OCF 为 **25.45723**，较上一正式提交 20.54607 增加 **4.91116**，较第一份提交 15.04410 增加 **10.41313**。公开榜当时 76 支队伍，`ccjjllt` 位列第 35。增量覆盖 v2 到 v14 的多项更新，Kaggle 没有提供本次各任务的单独分数。当前 UTC 日已用一次提交，下一次窗口为北京时间 2026-11-27 08:00。 [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions) · [公开榜](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/leaderboard)

## 2026-11-26 第二十一轮研究

Task 1 用带标签灰度图估计公开字体的像素生成过程，比较字重、原生栅格与过采样、下采样核和亚像素位移。在新留出参考图上固定候选码位，单独测生成过程可能带来的视觉改善。另把 1.4× Poisson 等预算方案打包作正式输入对照，从 v14 保留无法重现的旧预测。 [Noto Sans CJK 字体资产](https://github.com/notofonts/noto-cjk/tree/Sans2.004/Sans) · [此前权重复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-v13-repeat-weight-recheck)

Task 2 用早、中、晚时间窗的多个独立重复协方差约束跨天坐标旋转，再用另一半重复考察其迁移。对照包括同汇总矩白化、时间窗错配，以及只用于诊断的带标签映射。 [二阶统计盲分离](https://doi.org/10.1109/78.554307)

Task 3 把训练图库按轮廓聚类，把每个录制中各形态组的总后验质量约束在已知图库容量上。比较同天估计、共享跨天校正、同容量打乱、整字级平衡和原方案。在同一输出候选内使用官方 OCF 数据拟合的图像效用完成选择。 [后验约束](https://www.jmlr.org/papers/v11/ganchev10a.html)

追加两项放电对照。Task 1 在冻结四折候选上，用一半试次估计软跨录制对应，另一半独立传递 Poisson 证据，再交换两半。以相近计数的打乱对应和已知同字对应为诊断，并在同一独立留出上固定原选字、单独换用新栅格。Task 2 在 mice2 比较单日 96 字与跨三日共 384 字的精确 OCF 校准，归一化锚、候选边和更新次数保持一致。 [多视图匹配研究](https://proceedings.mlr.press/v54/tang17a.html)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 2 多时窗矩的无标签跨天对齐 | mice1 留出 session1/3。与同维 CORAL 相比，早、中、晚窗及跨窗的六个算子同时对齐，四个独立验证半组的矩误差均下降，Top‑32 真字召回增加 0.54/1.07 个百分点。256 张配对图上，原 v9 神经基线为 20.369，联合对齐为 19.726（−0.642±0.509），同维 CORAL 为 20.209，时间窗反序为 20.170。此处基线未包含当前精确 OCF 重排。所有指派都保留 2,748 个唯一字形。Kaggle CPU 用时 206 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-moment-alignment) |
| Task 3 跨录制轮廓组容量 | 双鼠八块各 256 个配对单试次，共 2,048 图。原 v14 为 17.906，固定 20% 单查询精确效用为 18.549（+0.644±0.156），再加跨录制共享轮廓组容量为 18.558。容量相对同一独立效用的变化 +0.008±0.120，相对同容量轮廓打乱为 −0.001±0.112。单查询效用在 mice1 为 +0.375±0.158，在 mice2 为 +0.913±0.262，mice2 四块变化 +0.942/+0.571/+1.337/+0.801。Kaggle T4 用时 403 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-shape-capacity) |
| Task 1 灰度栅格生成 | 342 个独立字体一致的训练码位用于拟合，从 24 种绘图组合中选出 Noto 原生 64 像素、字重 400、字号 50 的方案。在两个各 80 字的外层中保持相同六个候选码位，用真图挑选时 OCF 为 79.441→82.053（+2.612±0.586）及 80.702→82.379（+1.677±0.599）。两个同码位子集分别提升 2.708 和 1.584，均无逐像素完全匹配。此处只用图像测覆盖，未以放电选图。Kaggle CPU 用时 529 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-raster-generation) |
| Task 3 mice2 单查询精确效用复核 | 四个录制各 384 个新字，1,536 个配对单试次。原 v14 为 17.944，固定 20% 的校温精确效用为 18.041（+0.096±0.203），固定温度 1 为 18.044（+0.100±0.190），同温度轮廓代理为 17.921（−0.023±0.186）。首轮 +0.913 的增益在独立评分字上未复现，两轮按 640 个评分字加权为 +0.423，且覆盖相同四个录制。Kaggle T4 用时 373 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-mice2-ocf-recheck) |
| Task 2 mice2 精确 OCF 校准规模 | 四个留出录制各 256 个新字。保持候选图、160 次更新与 8 分锚一致，v14 的全库 v9 为 20.053，单日 96 字为 20.557（+0.504±0.375），三日共 384 字为 20.293（+0.240±0.328），三日相对单日 −0.264±0.434。所有方法均一对一匹配 2,748 个字形，各录制均未回退。在每块 32 个字上逐行用真图选优，完整候选图上限均值为 43.91，只用 v9 前八候选为 32.40，均未受完整指派约束。Kaggle T4 用时 730 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-mice2-ranker-scale) |
| Task 1 拆半跨录制转移和固定选字渲染 | mice1 四个录制各 320 个新整字，共 1,280张配对图。在同一四折候选图上，原策略为 30.871，软对应传递另一半放电为 30.899（+0.028±0.103），相对计数分层打乱 +0.019±0.104。使用留出标签指定真实同字对应的诊断为 31.919（+1.048±0.213）。保持原策略选字并只换用此前锁定的原生栅格时，OCF 为 30.940（+0.069±0.087），四个录制变化 +0.116/+0.189/+0.029/−0.058。评分字没有参与栅格拟合或前轮覆盖评估。Kaggle T4 用时 1,172 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-peer-transport) |

### v15 Poisson 对照包

在正式 mice1 Task 1 的 20 次试次中应用 1.4× Poisson 系数，保留四折精确重排器和逐行候选数。1,360 个查询中重跑逐像素复现 v14 的 1,242 个；未复现的 118 个全部保留原样，其中 31 个本来建议换字。实际替换 77 张，逐录制 27/13/18/19 张，其余 30,123 行与 v14 逐字节一致。ZIP、PNG、30,200 个 ID 及改动清单已复核。Kaggle T4 用时 1,135 秒，包 SHA-256 为 `7bfd0a6b7d11d8711299968fd19609432e2d295fec8d7658fb2c8bec4f7184a5`，尚未正式提交。 [Kaggle 打包](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v15-poisson)

## 2026-11-26 第二十二轮研究

本轮从全字库提名着手。Task 1 用公开汉字结构序列监督放电对字形布局和位置有关的部件，使用独立带标签整字测等预算候选。另比较匹配模型参数的感受野结构，隔离固定低频相位与局部高频能量，验证对真实近形字放电差的预测。 [汉字开集部件识别](https://openaccess.thecvf.com/content/ICCV2025/html/Li_MSA2_Multi-task_Framework_with_Structure-aware_and_Style-adaptive_Character_Representation_for_ICCV_2025_paper.html) · [公开 IDS 词典和许可](https://github.com/skishore/makemeahanzi/blob/bddc96d41bef78427ed0e034e9f7e31d71fd1b92/COPYING) · [V1 双子场研究](https://www.nature.com/articles/s41593-026-02213-3)

Task 2 用来源日同字重复的噪声关系划分神经元群，比较按群计数条件化的检索与同容量乱组、整群模型和 Poisson 检索。Task 3 在候选条件下用同试次早窗的放电偏差更新晚窗预测，并以同一字的另一试次早窗替换和只用群体计数作对照。 [视觉皮层乘性变异](https://pmc.ncbi.nlm.nih.gov/articles/PMC4534383/) · [小鼠视觉短时适应](https://pmc.ncbi.nlm.nih.gov/articles/PMC7474461/)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 2 按局部神经群增益条件化提名 | mice2 四源中的两个独立留出日，全库每行 16 条候选边。v9 等预算深挖的真字覆盖为 21.98%/21.98%，局部条件化为 22.78%/22.23%，活动率分层打乱为 23.14%/22.02%，整群条件化为 23.51%/21.98%。局部条件化只在第二块超过所有对照，净增覆盖 7 个字。来源日拟合的组内放电残差相关在两天复现，实验止于候选覆盖，未计算 OCF。Kaggle CPU 用时约 199 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task2-local-gain-nomination) |
| Task 3 同试次早窗条件化的晚窗似然 | 双鼠八个录制，4,096 个全字库查询及 2,048 个官方 OCF 配对单次。真早窗配对相对同字打乱使正确字的对数似然在八块均提高，幅度 0.48–7.29 nats/试次。Top‑32 从 23.24%/12.84% 增到 25.05%/14.55%，同字打乱已为 25.00%/14.36%。相对 v14 的成对实际 OCF 为 −0.067±0.295 和 −0.422±0.317，等预算扩候选亦未改善。Kaggle T4 用时 136 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task3-early-late-innovation) |
| Task 1 公开汉字部件提名 | 公开 IDS 匹配到 2,300 个拟合字，学习 177 项布局和部件。mice1 在 320 个新留出整字及 1,280 个四试次配对 OCF 上，v14 为 32.547，等预算补四个深度候选为 32.624，IDS 等预算提名为 32.987，相对 v14 为 +0.440±0.158，四录制均提高。相对同预算深挖为 +0.363±0.177。再加固定 3 分的部件排序项为 33.263（+0.716±0.281），改善集中在录制 2 和 3。同样的打乱标签提名及加分得 32.177。305 个高置信图像近邻仅用于估计公开码位召回，实际 OCF 使用留出真图。Kaggle T4 用时 1,168 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-ids-open-atlas) |
| Task 1 双子场近形放电建模 | 320 个新留出整字构成 149 对独立近形对，四录制共 512 张配对 OCF。三臂同为 45,148 参数。双子场真实响应二选一准确率比混合真实高 1.342±0.706 个百分点，比双子场响应方向打乱高 4.362±0.798 个百分点。全库 v14 为 31.664，双子场等预算为 31.108（−0.557±0.642），混合真实为 32.660（+0.996±0.601），但后者全库 Top‑32 召回四块均下降。双子场未替换入正式包。Kaggle T4 用时 1,166 秒。 [Kaggle 实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-bipartite-glyphs) |
| Task 1 部件提名与 1.4× Poisson 独立复核 | mice1 新 320 个整字与此前十个队列均无重叠，四日四试次共 1,280 个配对。v14 为 31.489，v15 Poisson 为 32.080（+0.591±0.253），四日变化为 +0.895/+0.916/+0.497/+0.057。单用 IDS 的 31.425 比 v14 低 0.064±0.088，比等预算深挖低 0.110±0.111。联合为 31.976，比 Poisson 低 0.105±0.102，比同一 Poisson 的等预算深挖低 0.183±0.099。联合比 Poisson 改选四日分别为 6/7/4/2 张，逐日 OCF 差为 −0.182/−0.043/−0.221/+0.029。已知同字跨四日汇总 16 次时，联合为 50.786，比 Poisson 高 0.903±0.511，该代理使用已知跨日配对。Kaggle T4 用时 1,124 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-ids-poisson-recheck) |

### MICrONS 预训练视觉核心迁移

Wang 等人用小鼠观看自然视频时的钙成像记录训练视觉模型，再迁移到其他小鼠。本实验冻结 v1 归档中的共享 `params_core.pt`，以 30 Hz 输入三帧黑屏和七帧同一黑底字，在固定时间点提取空间特征。对照使用相同架构随机权重与既有轮廓核。各臂使用相同的核回归和留出整字，先测真字图库内的检索、独立重复的近形字放电差和官方 OCF。图库由已知的 320 个留出字组成，因此分数只描述已知图库内的排序。 [论文](https://doi.org/10.1038/s41586-025-08829-y) · [权重发布说明](https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/foundation_model/readme_v3.md)

FNN 代码固定为 `v2.1.0`（提交 `8f34c6753bd818c8fcd1daa91499a057252bea7e`），采用 MIT。发行方 BossDB 的 MICrONS Minnie 项目清单列出权重目录，并标注 CC BY 4.0。归档可免费匿名访问，项目级声明是权重使用的许可依据。归档记录了官方 MD5，Kaggle 脚本校验后记录 SHA-256。 [代码许可](https://github.com/cajal/fnn/blob/v2.1.0/LICENSE) · [BossDB 项目及许可](https://bossdb.org/project/microns-minnie) · [比赛规则](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/rules)

mice1 的 320 个新留出整字、四录制上，固定核回归的已知图库 Top‑1 为轮廓核 19.45%、随机核心 19.69%、已发布核心 21.56%。151 对独立近形字的二选一分别为 83.77%、84.69%、85.35%。384 张配对图的官方 OCF 分别为 31.772、32.640、35.533。预训练相对轮廓核为 +3.762±1.996，相对同架构随机核心为 +2.894±1.717。将预训练与轮廓核各取 50% 时为 32.587。同一留出真图图库随各臂固定，测试只测图库内的排序。第二版用共同 0.02 黑位和 FP32，修正第一版随机权重在全零前缀的零范数。有效版本在 Kaggle T4 用时 149 秒，归档 SHA-256 为 `c378e8fbf1e7a9e44f07d5d85fba972b946d73adc096fbf58c69cc524052e1e1`。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-microns-transfer) · [结果](../outputs/siclink-task1-microns-transfer/task1_microns_transfer.json)

Task 3 用冻结核心为 2,748 字生成四折图像先验。每折排除校准和评分字，每个字自己的放电也不进入其先验。三次同日实测更新 Gamma 后验，另一次作单试次查询。mice1 四日 1,280 次独立查询、512 张配对 OCF。v14 为 20.101，预训练固定原七个候选为 20.119（+0.018±0.494），等预算提名为 +0.002±0.579。固定候选相对轮廓核为 +0.163±0.312，相对同架构随机为 −0.345±0.361。预训练融合的 Top‑16/32 由 17.42%/24.06% 到 17.66%/25.31%，Top‑32 与同架构随机相同，轮廓核为 25.78%。同均值的 Gamma 预测改善真字放电对数似然约 5.57 nats/试次，固定候选的 OCF 变化为 +0.063±0.083。Kaggle T4 用时 187 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-microns-gamma-prior) · [结果](../outputs/siclink-task3-microns-gamma-prior/task3_microns_gamma_prior.json)

Task 1 开放字库复核以 27,550 张公开字形做候选。沿用 v14 四折重排和渲染，仅替换时序及汇总 Poisson 的预测放电。新 320 个整字与闭集对照、部件首轮和双子场队列均无交集，四录制各有四次实测放电，共 512 张配对 OCF。v14 为 29.190，已发布核心固定原候选时为 31.186（+1.996±0.686），等预算提名为 31.470（+2.280±0.782），四录制变化 +1.152/+0.610/+3.259/+4.099。同架构随机核心等预算为 30.945（+1.755±0.828），预训练相对随机为 +0.525±0.766。306 个高置信公开码位代理的 Top‑32 为 55.96%/58.01%/59.48%（v14/随机/预训练）。证据支持跟进新的字形到放电核读出，预训练权重独有的增益仍有待辨别。Kaggle 单 T4 用时 2,279 秒。 [Kaggle 全库实验](https://www.kaggle.com/code/ccjjllt/siclink-task1-microns-open-atlas) · [结果](../outputs/siclink-task1-microns-open-atlas/task1_microns_open_atlas.json)

在与十个既往队列均无交集的 320 个完整字上，冻结相同的四折精确重排及每行候选数，直接复核 v15 和两档公开核心。单日四次、共 1,280 个配对 OCF 中，v14 为 33.537、v15 为 33.604、MICrONS 1.0× 为 33.624、MICrONS 1.4× 为 33.981。1.4× 相对 v15 为 +0.377±0.536，四个录制分别为 +0.146/+2.217/−0.130/−0.725。已知同字跨四日汇合 16 次的 320 个配对中，四方案分别为 53.831/54.328/55.876/57.565。1.4× 相对 v15 为 +3.237±1.112。高置信公开码位代理 Top‑32 从 79.87% 到 84.56%。完全遮蔽直接 Poisson 后，两个条件均下降。跨日汇合使用已知同字对应，正式条件是同日 20 次，因而正式总分须通过榜单确定。Kaggle T4 用时 1,705 秒。 [Kaggle 重复复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-microns-repeat-recheck) · [结果](../outputs/siclink-task1-microns-repeat-recheck/task1_microns_repeat_recheck.json)

### v16 MICrONS 候选归档

用 mice1 的全部 2,748 个带标签字拟合一次共享核读出，冻结公开核心并沿用 v14 四折重排、逐行候选预算和 v8 继承渲染。正式输入每组使用 20 次实测放电。重跑逐像素复现 v14 的 1,254/1,360 条，无法复现的 106 条保留原输出。1.0× 和 1.4× 只改变 mice1 Task 1 的 402 和 407 条，分别与 v15 分歧 434 和 426 条，两档彼此分歧 100 条。独立核对每档 30,200 个 ID、顺序、ZIP CRC、64×64 灰度 PNG 和差异清单，两个候选均未提交。Kaggle T4 用时 1,665 秒。 [Kaggle 打包](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v16-microns) · [来源、许可证与哈希](../outputs/siclink-candidate-v16-microns/source_candidates.json) · [逐任务核对](../outputs/siclink-candidate-v16-microns/local_archive_verification.json)

下一提交窗口优先选 v16 MICrONS 1.4×，因为它在更接近正式重复数的留出对照中优于 v15，且首轮全字库开放对照的四个录制均受益。单日四次下的相对收益波动，v15 保留为对照。选定文件 `kaggle_submission_microns_poisson140.zip` 的 SHA-256 为 `a342f28c350b51b803cf53dc5369210d522706aa0fc96de57cbdb805ec12a20e`。北京时间 2026-11-27 08:00 开启新的提交窗口。当前已评分 v14 为 25.45723（编号 `58918204`），2026-11-26 UTC 的一次提交额度已经使用。 [候选 ZIP](../outputs/siclink-candidate-v16-microns/kaggle_submission_microns_poisson140.zip) · [正式提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第二十三轮研究

官方参赛包给出 Task 3 每鼠、每个训练录制的每个已见字各有一条未用于训练的试次。因此身份推断可以使用每字出现一次的约束。实验独立打乱各录制查询，用放电建立对应。Task 2 研究完整指派中的短循环，比较同一批两行和三行交换的逐边预测及联合收益。Task 3 研究固定覆盖率的真匹配诊断和匿名多日放电邻域，分别考察理想匹配和实际可获得证据。 [官方参赛包](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data) · [结构化目标的直接损失](https://proceedings.mlr.press/v139/indelman21a.html)

Task 1 依据 MICrONS 原论文的空间位置与特征权重分解，用冻结视觉核心重新拟合两只鼠的逐单元读出。先在互斥的两组训练字上估计去墨量后的放电关联，借其复现性定位局部空间池化，再在同参数读出中比较真实位置、按总活动率分层交换的位置和全图池化。独立留出已知图库提供早期验证，沿用此前全图核读出作为对照。 [MICrONS 论文](https://doi.org/10.1038/s41586-025-08829-y)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 冻结 MICrONS 逐单元空间池化 | 双鼠四录制各 320 个新留出字，已知图库每鼠 512 张配对 OCF。mice1 全图池化/错位置/全图核/真位置为 28.464/31.041/32.889/34.832，mice2 为 22.486/25.201/25.805/27.062。相对同维全图池化为 +6.368±1.460 和 +4.575±1.174，相对错位为 +3.791±1.681 和 +1.861±1.224，相对全图核为 +1.943±1.957 和 +1.256±1.531。真位置相对全图核在 Top‑16 提高 5.86/3.13 个百分点。第一版的 T4 MAGMA 批量解线性系统报错，第二版改在 Kaggle Notebook 的 CPU 解小矩阵，其余使用 T4，完整用时 139 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-microns-spatial-readout) |
| Task 2 监督短循环重排 | mice2 两个完整录制各 768 张新留出配对图。原 v14 为 20.385，整循环为 20.451（+0.065±0.211），同预算逐边为 20.498（+0.113±0.217），仅按 v9 损失选择为 20.343。整循环相对逐边为 −0.047±0.219。各对照在两块分别改动 352/360 行，均保持 2,748 个唯一字，改动分解为二行和三行互斥局部交换。Kaggle T4 用时 411 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-local-cycles) |
| Task 3 匿名多日放电关系 | mice1 四日各 320 张新留出配对图，四日查询独立打乱。v14 为 18.469，20% 单查询精确效用为 18.728，半平衡后为 18.881（+0.412±0.154，四日均提高）。加入其他日的匿名查询近邻为 18.774，相对半平衡 −0.107±0.120，比固定跨日图低 0.031±0.074，比匹配的分层打乱低 0.065±0.077。真实关系的全库 Top‑32 为 25.86%，半平衡为 27.50%，独立为 23.20%。独立重复的近邻重合率为 9.0%–10.7%，随机期望约 5.0%。Kaggle T4 用时 362 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-relational-consensus) |
| Task 3 诊断性一对一锚的覆盖上限 | 双鼠 session1/3，每块 128 个评分查询，在所有锚覆盖率下保持评分行不变且从不作为锚。无锚平衡的全库 Top‑32 为 21.09%，使用已知真字固定 5%/15%/30%/60% 的其他查询后依次为 21.88%/23.44%/25.78%/36.72%。30%/60% 自动锚则为 16.60%/13.87%，分别错删约 30.3%/63.1% 仍需预测的真字。在固定 v14 前七候选与同一轮廓效用下，真锚相对无锚平衡的 OCF 变化分别为 −0.035±0.063/+0.036±0.067/+0.159±0.238/+0.153±0.190。128 个真图选优的前七均分为 26.765。第二版修正首版的轮廓效用量纲，身份结果及已评分试次与首版一致，Kaggle CPU 用时 600 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-anchor-coverage) |
| Task 3 同候选、同效用的双鼠独立复核 | 每鼠四块、每块 256 个新评分字，各有 1,024 张配对单试次。mice1 的 v14、独立 20% 效用、半平衡、全平衡依次为 18.790/19.214/19.091/19.082。独立为 +0.424±0.137，四录制均提高，半平衡相对同一独立效用为 −0.124±0.132。mice2 四种为 16.918/17.001/17.094/17.405。全平衡相对 v14 为 +0.486±0.207，四录制均提高，相对独立为 +0.403±0.186。mice2 此次将输出候选固定为七个，旧复核允许全图库作为输出，不能合并估计。Kaggle T4 用时 459 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-half-balance-recheck) |
| Task 1 mice2 全库 MICrONS 空间读出 | 27,550 个公开候选，320 个新字各四录制，1,280 张配对 OCF。同日四次下，v14、原 Poisson 1.4×、MICrONS 全图核 1.4×、MICrONS 逐单元 1.4× 分别为 25.297/25.617/25.830/26.104。逐单元相比 v14 为 +0.807±0.351，四录制 +1.411/+0.098/+1.276/+0.441，相比全图核为 +0.273±0.329。已知同字跨四录制汇总 16 次时，对应 OCF 为 36.093/37.413/36.747/36.692。逐单元相对放大的原 Poisson 为 −0.721±0.879，全图核为 −0.666±0.889。16 次汇总使用已知对应关系，只作高重复代理。Kaggle T4 用时 1,693 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-microns-spatial-atlas) |
| Task 1 全字库证据参与跨天配对 | mice1 两录制独立打乱，320 个新整字，使用每录制两次放电建立一对一链接，另两次用 27,550 个公开字形选图，并交换两半。在同一 80 字上，单日四次、仅放电匹配、联合匹配、分层打乱、真链接诊断的 OCF 分别为 18.963/15.821/16.088/15.925/17.455。联合相对分层打乱为 +0.163±0.392，相对单日四次为 −2.875±0.995；两半联合链接正确率为 3.44%/0.94%。23.28% 的联合首选落在两个单天两次短名单之外，新覆盖的高置信字形代理为 2.46%。这说明当前设定下的新候选大多没有找回正确字形。修正初版依赖导入后，第二版在 Kaggle CPU 用时 240 秒。 [一致性多视图匹配](https://onlinelibrary.wiley.com/doi/10.1111/cgf.12184) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-joint-atlas-pairing) |

### v17 候选归档

以 v16 MICrONS 1.4× 为底，用 mice1 四次带标签重复重拟 Task 3，再按每组一条正式放电选择。严格复现 v14 的七个候选，固定 20% 权重加入独立后验的期望 OCF。各录制温度只用其他三个录制的带标签留出试次拟合。四块原 v14 图像 10,992 张全部逐像素复现后，改动 473/456/443/423 张，共 1,795 张。相对 v16，其余 28,405 个 PNG 载荷逐字节一致。独立核验完整 30,200 行、顺序、ZIP CRC、64×64 灰度 PNG 和差异清单。Kaggle T4 用时 327 秒。SHA-256 为 `61dc0a81e8ff1f3f5c323f516818947c24886f7a6708ad71b84b9ee38aa52ec5`。2026-11-26 UTC 的名额已使用，候选待北京时间 2026-11-27 08:00 之后提交并核对结果。 [Kaggle 打包](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v17-task3-utility) · [候选 ZIP](../outputs/siclink-candidate-v17-task3-utility/kaggle_submission.zip) · [独立核验](../outputs/siclink-candidate-v17-task3-utility/local_archive_verification.json)

## 2026-11-26 第二十四轮研究

Task 1 复核 mice2 的高重复 Poisson 权重。另用训练字的真实栅格和公开字库共同识别左右、上下部件，把训练中同部件的栅格差异迁移到未见码位。冻结留出字的公开候选码位，比较原渲染、固定强度的同部件差异、形似异部件差异和完整部件替换，用官方 OCF 检验真实字风格能否迁移。Task 3 在新留出试次上比较同样七候选、同样效用下的独立与平衡后验。 [局部字体风格研究](https://ojs.aaai.org/index.php/AAAI/article/view/16340) · [公开部件词典](https://github.com/skishore/makemeahanzi) · [平衡矩阵估计](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)

| Kaggle 对照 | 结果 |
| --- | --- |
| Task 1 真实字图的同部件栅格迁移 | 2,428 个拟合字、320 个新留出字。279 个由两个公开字体独立匹配到相同码位，216 个支持编辑，共 341 个部件位置。冻结相同码位后，原图/同部件差异/形似异部件差异/同部件直接替换的实际 OCF 为 80.866/79.375/78.648/68.926。同部件差异相对原图 −1.491±0.340，相对异部件差异 +0.727±0.353。两组供体的公开字形相似度分别为 0.952/0.824，后一个差异同时含形状相近程度。此实验只测同候选的渲染，未使用放电。Kaggle CPU 用时 267 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-component-raster) |
| Task 1 感受野周边预测误差 | mice1 四个录制、320 个整字留出，固定训练字定位的感受野。只用互斥的周边像素预测中心局部 DCT，留出 R² 为 0.651。相比加性读出，真实二次预测误差使每字每试次的诱发单元和时间窗总偏差增加 0.718±0.264，相比同维的分离平方特征增加 0.260±0.178。单日四次的配对 OCF 相对加性为 −0.219±0.634，跨日已知同字 16 次代理为 +1.183±1.022，后者已知图库 Top‑16 从 63.44% 到 61.87%。这里只测局部 DCT 读出与已知图库，未扩展至公开全集。64 张每条件，Kaggle CPU 用时 75 秒。 [小鼠视觉皮层空间预测研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12829946/) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-context-prediction) |
| Task 1 mice2 高重复 Poisson 独立复核 | 新 320 字与全库空间、闭空间、原四折实验互斥，各策略每行候选数完全相同。同一录制四次的 1,280 张配对中，v14 为 25.142，1.4× 为 25.260（+0.118±0.218，四日 +0.714/−0.467/+0.030/+0.194）。已知同字跨四日 16 次的 320 张中为 34.083→35.301（+1.218±0.529），复现先前的 +1.320±0.556。固定候选增量 +0.570±0.354，固定原评分仅换提名 +0.632±0.396。高置信代理 Top‑32 为 50.51%→55.22%。此处 16 次依赖已知跨日对应，正式输入为同日 20 次。Kaggle T4 用时 1,023 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-mice2-poisson-repeat-recheck) |
| Task 3 mice2 同七候选效用复核 | 每录制 256 个新评分字，1,024 张配对单试次。v14、独立、半平衡和全平衡的 OCF 为 18.009/18.419/18.479/18.282。独立相对 v14 为 +0.410±0.237，四块 +0.374/+0.780/+0.078/+0.406。半平衡相对独立为 +0.061±0.162。全平衡相对独立为 −0.136±0.203，四块均低于独立，虽然全库 Top‑32 为 20.31%，高于独立的 15.72%。14,336 对采样得到 13,506 对去重官方 OCF 标注。Kaggle T4 用时 372 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task3-mice2-full-balance-replication/versions/2) |
| Task 1 冻结单字 OCR 特征 | mice1 320 个新留出整字，固定 1,344 字 Noto 字库。三臂共用 128 维拟合字 PCA、岭回归、候选和输出渲染。轮廓、随机同架构、冻结单字 OCR 在已知同字跨日汇总 16 次时的 Top‑32 为 51.88%/46.88%/73.75%，OCF 为 22.130/19.325/31.322。冻结相对轮廓 +9.192±3.302，相对随机 +11.997±3.098。同日四次 Top‑32 为 24.53%/21.41%/34.77%，OCF 为 14.564/16.180/17.875。两种重复条件各用 80 张配对 OCF，同日条件平均分到四个录制。受控图库为每个留出字预留从参考栅格匹配出的 Noto 代理字；公开 OCR 的训练字表也可能重叠。首版精确评分网格超时，第二版完成 440 个去重图对，Kaggle CPU 用时 203 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-native-hanzi/versions/2) · [公开 OCR 权重](https://huggingface.co/Ismantic/Handwritten) |
| Task 1 冻结 OCR 的完整公开字库复核 | 双鼠 320 个新留出整字，27,550 个公共候选按拟合字固定，参考图只用于事后代理召回和 OCF。已知同字 16 次时，轮廓/随机同架构/冻结 OCR/等权轮廓 OCR 融合的 OCF 在 mice1 为 18.933/19.641/25.036/25.908，在 mice2 为 16.205/16.160/19.032/19.830。融合相对轮廓为 +6.975±1.995/+3.626±2.167，Top‑32 为 15.94%→41.88%/9.38%→19.69%。同日四次的融合相对轮廓为 +5.763±1.827/+3.352±1.466。每鼠每条件精确评分 64 个配对字，四次组各录制 16 个，共 903 个去重官方 OCF 图对。OCR 在八个录制的同日 Top‑32 均高于轮廓。此处比较独立读出，尚未测相对 v16 的增量。Kaggle CPU 总用时 436 秒，特征提取约 58 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-native-hanzi-open-atlas/versions/1) |
| Task 2 mice2 冻结 OCR 跨天读出 | 仅用 session1/3 放电拟合，留出 session2/4。各臂在同一 v9 完整打分矩阵上混入固定 10% 经全块 RMS 缩放的读出，仍对 2,748 个字做一对一匹配。两日 Top‑32 为 17.43%→20.27% 和 19.76%→20.78%，固定温度 NLL 也降低。每个留出日用互斥的 96 个新字配对评分，v9/轮廓/随机/冻结 OCR 的 OCF 为 17.314/17.254/16.777/16.745。冻结相对 v9 为 −0.569±0.536，相对随机为 −0.032±0.297。结果覆盖两来源日的 v9，未拟合 v14 的额外 OCF 排序器。Kaggle CPU 用时 199 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-native-hanzi/versions/1) |
| Task 1 mice1 冻结 OCR 与 v16 直接比较 | 320 个未见整字，与前两轮 OCR、三轮 MICrONS、部件与周边对照无重叠。重拟四折 v14 与 MICrONS，沿用 27,550 个候选、逐行候选数和同一输出渲染。只在 v16 的 v4/v8 中等权混入标准化 OCR 和原解码轮廓。v14/v16/v16 加 OCR 同日四次为 31.203/31.672/33.475，OCR 相对 v16 +1.803±0.391，四日 +2.180/+1.919/+1.163/+1.949。跨日已知同字 16 次为 50.348/52.030/55.424，增量 +3.394±0.930。固定原候选加 OCR 为 +0.804±0.606，仅用 OCR 提名为 +1.874±0.640。高置信 Top‑32 为 80.67%→91.67%，等预算候选覆盖 73.00%→85.67%。共 1,816 个去重官方 OCF 图对。原会话被回收，仅留下初始化信息。新免费 Colab T4 重跑 3,656 秒，四折均已归档，对照未读正式放电。 [实现](../scripts/kaggle_task1_microns_hanzi_recheck.py) · [结果](../outputs/siclink-task1-microns-hanzi-recheck-colab/task1_microns_hanzi_recheck.json) |

Task 3 的 mice2 全平衡复核首版在 Kaggle CPU 上计算精确 OCF 标注，达到 2,220 秒时限时只完成初始化，未给出得分。第二版改用 T4 计算官方 OCF，沿用原定的留出字、试次、视觉拟合队列和采样数。 [CPU 版本](https://www.kaggle.com/code/ccjjllt/siclink-task3-mice2-full-balance-replication/versions/1) · [T4 版本](https://www.kaggle.com/code/ccjjllt/siclink-task3-mice2-full-balance-replication/versions/2)

### mice2 Task 1 正式打包准备

打包入口从 v17b 复现四个 mice2 Task 1 录制上的原 v14 字图，只在逐像素复现且按等候选数改选时更新，用正式同日 20 次放电推理。复用原校准第一折拟合的噪声估计，免去重复拟合。归档日志显示正式四折加外层拟合约需 805 秒，另有启动、推理和压缩时间。参赛账号的 Kaggle GPU 剩余额度低于完整拟合所需时间，配额北京时间 11 月 28 日 08:00 刷新。用户恢复免费 Colab 登录后，已在现有 Tesla T4 会话的独立目录启动打包，用参赛账号在远端下载官方数据。 [打包入口](../scripts/kaggle_task1_mice2_poisson_candidate.py)

### v17b 候选归档

选用 mice2 Task 3 的独立效用。它在新复核四个录制均提高 OCF，全平衡相对独立的领先未复现。基于 v17，复用 T4 复核发布的效用表和源录制校准温度，沿用七候选与 20% 效用权重，并以四次带标签实测重拟正式读出。Kaggle T4 打包用时 27.6 秒。10,992 个 mice2 Task 3 原图全部逐像素复现，修改其中 2,954 个，四块 713/743/745/753；其余 27,246 个预测逐字节相同。已核对 30,200 个官方 ID、顺序、ZIP CRC、64×64 灰度 PNG 和修改清单。正式包 SHA-256 为 `469f6d042755a5ed585fe7c58709fc9774136d1a2227ed3632681cbb3bc0cfdf`。 [Kaggle 打包](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v17b-mice2-utility/versions/1) · [候选 ZIP](../outputs/siclink-candidate-v17b-mice2-utility/kaggle_submission.zip) · [打包清单](../outputs/siclink-candidate-v17b-mice2-utility/source_candidates.json)

### v18 候选归档

在已恢复的免费 Colab Tesla T4 上，以 v17b 为底，只比较 mice2 Task 1 的同日 20 次正式实测放电和等预算候选。原 v14 策略从归档复现了 886/1,360 张正式图。1.4× 方案有 145 个改选，其中 76 张未能逐像素复现旧图而保留，最终修改 69 张，逐录制 22/14/18/15。独立核验 30,200 行、原顺序、所有 PNG CRC 与 64×64 灰度格式，其余 30,131 张预测逐字节相同。拟合和打包约 958 秒。SHA-256 为 `c75f2802ca6c1b5a69cb6f1d60e701bb96a9017c90352473e89a915f048c26b6`。这是高重复代理复核后的实际输入候选；正式成绩待北京时间 2026-11-27 08:00 后的提交。 [v18 ZIP](../outputs/siclink-candidate-v18-mice2-poisson-colab/kaggle_submission.zip) · [独立核验](../outputs/siclink-candidate-v18-mice2-poisson-colab/local_archive_verification.json)

### v19 候选归档

用完整 2,748 个训练字和四个录制的带标签实测放电重拟 mice1 Task 1 的 OCR、MICrONS 与四折排序头。使用正式同日 20 次放电，等量提名，保留 v16 继承 v8 锚图的渲染规则。原 v16 选择从 v18 逐像素复现 1,082/1,360 张；OCR 共建议改选 313 张，77 张未复现原图并保留，最终更新 236 张，四块 71/52/57/56。独立核验 30,200 行、原顺序、所有 PNG CRC 与 64×64 灰度格式，其余 29,964 条预测逐字节相同。免费 Colab T4 用时 3,601 秒，四折断点已保存。SHA-256 为 `7f3f483be15f28d6be707783a2fcedaa65b49473eea8020a040edf6e5c768fed`。v19 为北京时间 2026-11-27 08:00 打开的提交窗口首选。mice2 的 OCR 对照入口已准备，备用免费 T4 未分配成功，当前无直接增量指标。 [v19 ZIP](../outputs/siclink-candidate-v19-mice1-hanzi-colab/kaggle_submission.zip) · [独立核验](../outputs/siclink-candidate-v19-mice1-hanzi-colab/local_archive_verification.json) · [mice2 对照入口](../scripts/kaggle_task1_mice2_hanzi_recheck.py)

Task 1 优先验证跨字体的字形表示。第十四轮在同一 96 字、16 次代理下，把同一 Noto 渲染候选从配对的 320 个扩大到 27,550 个，OCF 从 62.05 降到 32.67。这个受控诊断量化了候选混淆，数值只适用于该对照。本轮先用公开的单字 MobileNetV2 权重取 96×4×4 空间特征，并核验固定版本及 SHA-256。可进一步借鉴多字体对比预训练，用公开授权字体生成同字的不同字体、字重和栅格视图，用相近异字作困难负例。将来可用单字体和打乱跨字体对应的同架构对照，并以候选内真图选优量化覆盖。 [全库对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-gallery-gap) · [MSA²](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_MSA2_Multi-task_Framework_with_Structure-aware_and_Style-adaptive_Character_Representation_for_ICCV_2025_paper.pdf) · [HOMOGLYPH](https://aclanthology.org/2023.emnlp-main.863/)

## 2026-11-26 第二十五轮研究

| Kaggle 或 Colab 对照 | 结果 |
| --- | --- |
| Task 3 mice2 冻结 OCR 的两位重提名 | session1/3 使用每字一个新留出试次，原 v19 策略用另三个带标签重复重拟。固定原胜者、另四张候选以及效用表和温度，两位从原候选之外提名。全 2,748 字的七候选召回为 4.55%→5.68% 和 4.69%→5.75%。47 个新评分字×两日上，OCR 组和原组的选图逐张相同，官方 OCF 均为 16.990。每日报告 16 张图的候选内真图选优，OCR 相对原策略 +0.129±0.431。新增形态读出与 PCA 排除 47 个评分字，原策略按既定任务使用这些字的另外三个带标签试次。Kaggle CPU 用时 247 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-native-hanzi-nomination/versions/1) |
| Task 1 mice1 冻结 OCR 的匿名跨日配对 | 320 个新字，独立打乱各录制。使用两次实测放电的 64 候选后验重合做一对一匹配，另一半两次放电用固定 OCR 读出选图，两次交换拆分。六个录制对的真匹配率由轮廓的 1.28% 到 OCR 的 1.98%，Top‑16 由 11.41% 到 17.31%。64 个预选字 × 四录制 × 双拆分，轮廓配对/OCR 配对/同日两次/同日四次/真对应跨日两次加两次的官方 OCF 为 13.169/13.705/14.808/17.640/17.586。OCR 相对轮廓配对 +0.536±0.389，相对同日四次 −3.935±0.768。这项结果限定在独立 OCR 读出和同等实测试次数的跨日组合。Kaggle CPU 用时 747 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-hanzi-peer-links/versions/1) |
| Task 1 mice2 冻结 OCR 与 v18 直接比较 | 320 个新留出整字，与此前五个队列无重叠。四折 v14、27,550 候选、逐行候选数及渲染固定，在 v18 的 v4/v8 中等权混入标准化 OCR 和原解码轮廓。v18/加 OCR 同日四次为 26.209/26.852（+0.643±0.251，四日 +0.858/+0.099/+0.615/+1.000），已知同字跨日 16 次为 35.327/39.659（+4.332±0.919）。固定候选只改分 +1.304±0.579，只改提名 +1.896±0.644。高置信 Top‑32 为 60.47%→78.74%，等预算候选覆盖 46.84%→69.77%。正式数据为同日 20 次，本次只访问训练集。共 1,530 个去重图对，免费 Colab T4 用时 1,141 秒，四折均已归档。 [对照报告](../outputs/siclink-task1-mice2-hanzi-recheck-colab/task1_mice2_hanzi_recheck.json) |

### v20 候选归档

在同一免费 Colab T4 上复用官方数据，用完整 2,748 个训练字和四次实测重复重拟 mice2 的四折排序头与 OCR 读出。正式输入为同日 20 次放电。保持 v18 原始 Poisson 1.4×、原渲染和 v14 的逐行候选数，在 v4/v8 混入等权 OCR。原 v18 策略逐像素复现 v19 的 894/1,360 张目标图。OCR 共建议改选 339 张，138 张原图无法复现，最终应用 201 张，四录制 52/45/51/53。独立核验全部 30,200 行、顺序、PNG CRC 和 64×64 灰度格式，其余 29,999 个预测逐字节相同。打包用时 1,077 秒，四折断点已归档。SHA-256 为 `b098c3bc62aa96b289b72c512e977fa246fb37f3ddfdab32ffe786d3e7ba8281`。v20 为北京时间 2026-11-27 08:00 提交窗口的首选，正式验证仍待窗口开放。 [v20 ZIP](../outputs/siclink-candidate-v20-mice2-hanzi-colab/kaggle_submission.zip) · [核验清单](../outputs/siclink-candidate-v20-mice2-hanzi-colab/local_archive_verification.json)

## 2026-11-26 第二十六轮研究

Task 1 在冻结 OCR 的 128 维特征内校准逐轴预测幅度和误差。先用训练字拟合逆读出，再用单独一批校准字估计四次重复的失配，并比较同一预测、渲染、等权轮廓分支下的余弦、等方差、对角与收缩协方差打分。 [群体放电的贝叶斯校准](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063820/)

| 对照 | 结果 |
| --- | --- |
| Task 1 mice1 OCR 误差校准 | 2,044 个逆读出拟合字、384 个校准字和 320 个与五个旧队列互斥的留出整字。两个条件各评分同一批预选的 96 个字。余弦/等方差/对角/收缩协方差的同日四次 OCF 为 17.512/17.787/18.490/18.127。已知跨日对应 16 次为 26.478/29.051/29.085/29.158。对角相对余弦分别为 +0.978±1.006 和 +2.607±1.504，16 次的高置信代理 Top‑32 从 38.23% 到 42.66%。三种校准模型的 16 次 OCF 相近，当前对照无法确认完整协方差的增量。此为简化读出，未拟合 v19/v20 排序器。Kaggle CPU 用时 236 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-hanzi-latent-likelihood/versions/1) |
| Task 2 mice2 一对一约束后的候选上限 | session1/3 拟合，session2/4 留出。两个等预算图均含 12 条候选边，固定八条基础边和同一 0.15 反向排名干预。128 个新字的实际 v9/加深图/反向提名为 18.255/18.079/17.989。64 个随机评分字的逐行→完整一对一真图选优在加深图为 37.337→34.047，在反向提名图为 37.927→36.792。反向提名的实际 OCF 相对加深为 −0.089±0.226，32 个拥挤字的约束上限相差 −1.729±2.671。真图选优可重排 1,731–2,230 个未评分行，因此只衡量抽样行在所有 2,748 个输出仍唯一时的可达分数。1,399 个去重图对，Kaggle CPU 用时 556 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-constrained-ceiling/versions/1) |
| Task 3 mice2 v20 原七候选选图余量 | session1/3 各评分 48 个互斥字，同字三个支持试次与单次查询互斥，查询也是早期对照未使用的试次。实际 v20 为 19.607，原七候选真图选优为 29.386（+9.779±1.435），原排行榜前十六张真图选优 35.361（相对七张 +5.976±1.556）。混合十六张真图选优 33.375，沿用 v20 决策尺度后实际只改选 1/96 张，OCF 为 19.494（−0.112±0.112）。1,920 个官方图对，Kaggle CPU 用时 765 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-v20-candidate-headroom/versions/1) |
| Task 1 mice2 OCR 误差校准复核 | 保持拟合、校准、打分协议，换鼠及随机划分。320 个新留出字与六个旧队列均无交集，各条件评分同一批 160 个字。余弦/等方差/对角/收缩协方差同日四次为 15.620/16.286/16.275/15.868，已知跨日对应 16 次为 19.686/18.937/19.201/18.526。对角相对余弦分别 +0.656±0.793 与 −0.485±0.908。mice1 的 16 次 OCF 增益未能复现，v20 保留原 OCR 打分。663 个去重图对，Kaggle CPU 用时 357 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-hanzi-likelihood-mice2/versions/1) |
| Task 3 mice2 原七候选官方效用校正 | 固定候选、神经后验、80/20 权重及原归一化。32 个单次查询各从后验和均匀混合中抽 32 张图库图，以官方 OCF 与现有效用表的残差校正期望效用。v20 为 17.390，校正为 17.753（+0.362±0.711，两日 −0.295/+1.020），真图选优为 32.543。独立半组有 16 次至少一方改选，只在两次改选中选中同一张，前 32 名后验质量为 9.64%/11.05%。7,104 个用于校正的去重图对和 224 个配对端点，Kaggle T4 用时 173 秒。CPU 小样本每查询 8 张、12 个查询，和 GPU 组有三个同日字重叠，OCF 17.284→16.781（−0.503±0.968），用时 333 秒。两次证据均不足以更新 v20。 [T4 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-v20-exact-utility/versions/1) · [CPU 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-v20-exact-utility-cpu/versions/1) |

### Task 1 笔画敏感视觉特征

TextVerifier 的冻结全局特征和保留 4×4 位置的特征使用同一个字库和岭回归框架，与现有 OCR 并列比较。先用 256 个校准字检查近形字排序，再在另 320 个留出整字上用全部 27,550 张公开候选评估。视觉编码器可处理发布原型表以外的字形。 [论文](https://arxiv.org/abs/2610.19551) · [发布权重](https://huggingface.co/ATH-MaaS/TextVerifier-CJK-CLIP/commit/adc233a117cbb039842b245b5acf84e0143a5f11) · [对照入口](../scripts/kaggle_task1_textverifier.py)

Kaggle CPU 下载公版权重、核对 SHA-256 并读取 294 个视觉张量及 4 个投影张量，用时 18 秒。T4 试跑成功加载模型，在第 97 秒完成公开字库渲染并进入特征提取，随后触及 GPU 时间上限。CPU 另用 92 秒预渲染字库，保存 256 个校准字对应的公开字形与 115 对互斥近形字，与 T4 试跑的配对和字库顺序完全一致。在 Kaggle 挂载两项产物后可通过 `--checkpoint /kaggle/input --pilot-bank /kaggle/input` 复用。参赛账户 GPU 配额在北京时间 2026-11-28 08:00 刷新。 [权重校验](https://www.kaggle.com/code/ccjjllt/siclink-task1-textverifier-assets/versions/1) · [T4 试跑](https://www.kaggle.com/code/ccjjllt/siclink-task1-textverifier-pilot/versions/1) · [CPU 预渲染](https://www.kaggle.com/code/ccjjllt/siclink-task1-textverifier-pilot-bank/versions/1)

7897 的 Google 连接恢复后，免费 Colab 成功分配 Tesla T4。用 1,024 个拟合字和预渲染的 115 对互斥近形字比较四个录制。以同日四次平均，现有 OCR/全局特征/空间特征/OCR 加空间特征在公开字形下的二选一准确率为 76.09%/69.78%/76.74%/81.09%，融合相对 OCR +5.00±1.61 个百分点。使用同一批真实字图时为 75.65%/70.43%/76.74%/80.43%，融合 +4.78±1.82 个百分点。已知同字对应跨日 16 次的公开字形比较为 90.43%→93.04%（+2.61±2.89 个百分点）。Colab 用时 206 秒，实际评估范围为近形字辨别率。 [Colab 近形字对照](../outputs/siclink-task1-textverifier-pilot-colab/task1_textverifier.json)

同一台 Colab T4 用 2,172 个拟合字和 27,550 个公开候选跑完整字库。320 个外层字与七个早期 OCR 队列以及本轮 256 个校准字没有重叠。64 个预选字在四个录制各评分一次，再按字合并计算配对标准误。现有 OCR/全局特征/空间特征/OCR 加空间特征的同日四次首选 OCF 为 14.308/14.890/15.164/15.456，融合相对 OCR +1.148±0.502，四日分别 +0.951/+0.958/+0.802/+1.881，前四候选真图选优 20.432→21.741（+1.308±0.648）。296 个高置信代理字的同日 Top‑32 为 4.56%→6.25%。跨日已知对应 16 次的首选 OCF 为 18.655→19.301（+0.646±1.475），前四真图选优 +1.657±1.301。共 3,855 个去重官方 OCF 图对，Colab 用时 443 秒。此轮是同等岭回归读出对照，v20 的排序与渲染留待直接复核。 [Colab 全库对照](../outputs/siclink-task1-textverifier-full-colab/task1_textverifier_full.json)

直接比较 v20 排序与渲染。重拟四折与外层读出，严格排除 320 个评分字。它们与两轮 TextVerifier 队列及五个早期 OCR 队列均无重叠，其中 96 个来自早期 mice1 OCR 误差校准的留出字。同候选数、同排序器、同渲染，OCR 分支内等权融合空间特征。同日四次的 1,280 张图上，v20/融合/固定候选/只改提名为 25.110/25.552/25.437/25.201，融合差 +0.442±0.228，四日分别 +0.866/−0.488/+0.460/+0.930。已知对应的跨日 16 次为 40.341/38.926/39.072/40.153，融合差 −1.414±0.568。正式 Task 1 供给同日 20 次实测放电，当前两项代理证据的收益方向不一致，因此保留 v20 提交包。Colab 免费 T4 用时 1,707 秒、共 1,795 个去重官方 OCF 图对，四折断点已归档。 [直接复核报告](../outputs/siclink-task1-mice2-textverifier-v20-colab/task1_mice2_textverifier_v20.json)

## 2026-11-26 第二十七轮研究

| 对照 | 结果 |
| --- | --- |
| Task 3 mice2 同均值的单试次零放电似然 | 保持 50–100 ms 和 100–250 ms 的预期计数，比较 Poisson、负二项、共享零放电校准、跨日同字校准、按放电率分层打乱。固定 v20 前七候选、锚定分数与图库效用。96 条相同留出试次的 v20/同均值负二项/共享零放电/同字校正/打乱 OCF 为 19.607/19.703/19.134/19.226/19.383。真实校正相对打乱 −0.157±0.276 OCF；完整字库两录制的 Top‑32 分别 13.54%/14.96% 与 13.57%/14.99%。负二项 Top‑32 由 13.36%/14.56% 升至 14.34%/15.83%，实际 OCF 变化 +0.096±0.348。当前零截断 Poisson 对正计数的拟合弱于负二项。保持均值的相对误差小于 1e−9，111 个去重官方图对，Kaggle CPU 用时 89 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-count-occupancy/versions/1) |
| Task 2 mice2 逐单元的跨日边际校准 | 两组互补的带标签源日、四个目标日、112 个新评分字。保持同一个完整一对一指派，固定 25% 放电分数融合。v9/目标日线性/冻结源 CDF/目标日 CDF 为 17.175/18.309/17.830/18.386。目标 CDF 相对 v9 +1.210±0.702，逐日 +0.880/+2.009/−0.607/+2.559，较冻结源 CDF +0.556±0.195，较线性 +0.077±0.374。目标 CDF 的全库 Top‑32 为 20.82%→23.33%，真字负对数似然 6.684→6.501，四天两项均改善。744 个去重图对，Kaggle CPU 用时 430 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-rank-marginals/versions/1) |
| Task 1 mice1 发布的跨字体相似度 | 相似字典收录全库 27,550 个码位中的 7,461 个，包含 320 个留出字中的全部 290 个高置信字形代理。2,204 个可信拟合字，固定 384 个拟合字码位、PCA、岭回归及公开字库。每个录制对同一 64 个预选字评分。轮廓加 OCR 基础/仅收录加分/匹配像素几何/发布相似度，同日四次为 18.167/18.306/18.857/18.119，已知对应 16 次为 22.106/25.347/26.537/25.478。发布相似度 16 次相对基础 +3.373±1.698，较覆盖加分 +0.132±2.256，较匹配像素 −1.059±2.283。基础版未叠加 v20 的 Unihan 和已见字先验。678 个去重官方图对，Kaggle CPU 用时 428 秒。 [论文](https://aclanthology.org/2023.emnlp-main.863/) · [字典发布](https://github.com/dell-research-harvard/HomoglyphsCJKTraining) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-homoglyph-transfer/versions/1) |
| Task 3 mice2 单试次粗轮廓与 OCR 残差 | 用拟合字估计方向敏感的 4×4 感受野并训练 12,733 参数的配对卷积读出。四个录制共 28,368 条真实单试次，2,364 个拟合字、128 个独立可靠性校准字、256 个形态留出字。固定 v20 候选和效用，在 session1/3 的 96 条预选查询上，v20/粗轮廓/完整 OCR/条件 OCR 残差/同形分组打乱残差的 OCF 为 19.607/18.982/18.896/19.143/19.143。残差相对 v20 −0.463±0.926，和打乱对照的逐行 OCF 相同。独立校准的轮廓、残差和打乱残差 \(R^2\) 为 0.089/0.001/−0.001。124 个去重图对，Kaggle CPU 用时 109 秒。 [空间与身份分头研究](https://arxiv.org/abs/2611.08887) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-retinotopic-residual/versions/1) |
| Task 1 mice2 覆盖先验与已有 Unihan 复核 | 另选 320 个与首轮互斥的留出字。原 Unihan 词频与已见字排除下，分别给发布字典收录和按同一 Unihan 字频与等级分层抽样的码位加 0.25。两组各 7,461 个码位，相交 6,499，各覆盖 291 个高置信代理中的 288 个。固定 128 个配对字的四次 OCF 为 16.913，已知对应的 16 次为 25.158，两种额外覆盖先验的所有配对 OCF 均相同。保留已见字排除、移除原词频时为 16.136/21.763。已有 Unihan 的条件下未观察到额外覆盖收益。1,155 个去重图对，Kaggle CPU 用时 487 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task1-repertoire-prior-transfer/versions/1) |
| Task 2 mice2 三源日留出边际校准 | v20 中该分支的 2,748 张图与 v9 正式包逐行一致。对四日轮流用其余三日拟合，每种方案对相同 256 个新评分字计算 1,024 次官方 OCF。v9/目标日线性/目标日 CDF 为 20.458/21.137/21.408，配对增量 +0.679±0.538 和 +0.949±0.566。CDF 的四日增量为 +0.038/+1.058/+1.128/+1.573，较线性 +0.270±0.325。全库 Top‑32 为 31.99%/34.85%/35.33%，两种校准的 Top‑32 与真字负对数似然在四日均改善。采用与首轮相同的 25% 融合和全库一对一指派。1,612 个去重图对，Kaggle CPU 用时 942 秒。 [Kaggle 复核](https://www.kaggle.com/code/ccjjllt/siclink-task2-rank-marginals-recheck/versions/1) |
| Task 1 mice2 按重复数训练选图排序器 | 在移除外层字的同一个小规模前向模型上，保持 v20 的候选、渲染及锚。两种新排序头均用相同的 96 个拟合字、同容量及 640 次更新，分别从同日四次或已知跨日 16 次生成校准候选。对相同 128 个留出字，重拟 v20/四次头/十六次头在同日四次为 24.575/23.052/24.204，跨日八次 31.006/27.098/30.872，跨日 16 次 35.860/33.239/35.671。16 次头在 16 次条件下较四次头 +2.433±1.942，较 v20 −0.189±1.574。24 个预选字的固定候选池真图选优为 46.910，相对该子集 v20 +18.567±4.832。3,113 个校准图对、1,308 个评分图对，Kaggle CPU 用时 2,825 秒，断点已归档。 [任务损失排序](https://proceedings.mlr.press/v162/mandi22a.html) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task-1-matched-repeat-head-cpu/versions/1) |

### Task 2 分位数校准的 CPU 归档

在四个带标签源日拟合 25% CDF 分支，并用 mice2 的正式 session5 匿名放电做完整指派。v20 含完整的 2,748 个不同官方字形。CPU 重拟原 v9 后逐像素复现 869 个目标输出。原始 CDF 方案相对 v20 改动 1,918 个输出，分为 13 个置换环。仅有一个两元素的完整置换环全部复现原图，存档中的 v21 CPU 候选只交换两张图。已核验 30,200 行、官方顺序、ZIP 和各 PNG CRC、保留全量一对一关系，其他 30,198 行逐字节一致。Kaggle CPU 用时 112 秒。输出 SHA-256 为 `08a508dbaeab7e9ac51c52496f823bda8e0c48161070d243c056be9e0f818e77`，保留为配对打包对照。 [Kaggle 打包](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v21-mice2-task2-cdf/versions/1) · [独立核验](../outputs/siclink-candidate-v21-mice2-task2-cdf/local_archive_verification.json)

两次新的免费 T4 分配均返回 Colab 服务暂不可用。另用三源日留出对照，冻结 v9 指派，在目标日分位数和冻结源分位数的标准化分数中给原字加同一个固定分值。使用 +3.0 时，四个录制的 2,748 个指派每次都与 v9 相同，160 个新评分字的 640 张配对图 OCF 同为 18.809。637 个去重图对，Kaggle CPU 用时 639 秒。 [固定原指派对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-anchored-cdf/versions/1)

另取 160 个与上一轮及旧评分队列互斥的字，在四个目标日复核较低的保留分值。v9/目标 CDF +1/冻结源 CDF +1/目标 CDF +0.5 的 OCF 为 18.798/18.524/18.725/18.747。主比较目标 CDF +1 改派 1,275/10,992 条，配对变化 −0.274±0.176，四日分别 −0.097/−0.654/+0.168/−0.510。+0.5 改派 6,893 条，变化 −0.050±0.521。保持已验证的 v20 为下一正式窗口首选。1,067 个去重官方图对，Kaggle CPU 用时 704 秒。 [较低保留分值复核](https://www.kaggle.com/code/ccjjllt/siclink-task2-anchored-cdf-followup/versions/1)

## 2026-11-26 第二十八轮研究

| 对照 | 结果 |
| --- | --- |
| Task 2 mice2 留出日的共同空间位移 | 每个目标日使用其余三个带标签日拟合，两个目标日各用 2/2 重复交叉估计位移和预测。每个方案对 112 个新字在两日各评分一次，保留每天 2,748 个唯一字形。v9、25% CDF 为 20.351、21.765（+1.414±1.023，两日 +1.432/+1.395）。真实和匹配墨量及放电率的打乱对照在四个拆半方向均选零位移，全部 5,496 个指派与 CDF 相同。另用 96 个已知配对做诊断，仅 session3 的一半重复选择非零位移，另一半相对零位移的真实配对对数似然下降 0.348 nats，当日 OCF 下降 0.377。345 个去重官方图对，Kaggle CPU 用时 325 秒，使用带标签训练录制。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-day-shift/versions/1) · [实验报告](../outputs/siclink-task2-day-shift/task2_day_shift.json) |
| Task 3 mice2 原七候选的非线性两两比较 | 固定 v20 七候选。用 7,168 个单试次从固定图库效用初始化，以另 64 个字的 448 个官方配对微调，再对 128 个互斥留出字每字精确评分七项。三种比较器均为 10,080 参数。v20、真实放电、摘要特征、按活动匹配的打乱对照 OCF 为 18.595、18.033、18.115、17.270。真实相对 v20 −0.562±0.859，相对摘要 −0.082±0.505，相对打乱 +0.763±0.536。真图选优在两个录制中比 v20 高 9.919 和 7.171。Kaggle CPU 用时 524 秒，留出评分使用训练集中的独立放电。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-pair-discriminator/versions/1) · [实验报告](../outputs/siclink-task3-pair-discriminator/task3_pair_discriminator.json) |
| Task 1 mice2 双鼠训练共享视觉读出 | 用 2,428 个拟合字估计 96 维 GCCA 空间，三臂视觉映射都用同一组 mice2 的四个录制及同样的 OCR、轮廓坐标。每个查询使用自身录制。同库 27,550 个候选，在 320 个新留出字中预选 96 个，在四日各评分一次。逐日岭回归、融合本鼠共享、融合真实双鼠共享、融合打乱异鼠对应的四次 OCF 为 16.265、16.666、17.362、16.263。真实双鼠相对岭回归 +1.096±0.440，四日增量 +0.790/+1.090/+1.376/+1.130，相对打乱 +1.099±0.482。已知同字跨四日汇总 16 次的 OCF 为 23.986、22.758、23.579、22.925，真实双鼠相对岭回归 −0.407±1.060，Top‑32 字形代理为 58.82%→64.01%。基线使用简化读出，正式 v20 排序、渲染尚未在此对照中比较。737 个去重官方图对，Kaggle CPU 用时 496 秒。 [MAXVAR GCCA](https://arxiv.org/abs/1605.09459) · [多视图监督](https://arxiv.org/abs/1702.02519) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-common-visual-readout/versions/1) · [实验报告](../outputs/siclink-task1-common-visual-readout/task1_common_visual_readout.json) |
| Task 2 mice2 同分类架构的三种子集成 | session2 和 session4 各用其余三日拟合，固定其余 v9 证据与 25% CDF。三个独立分类器分别按查询归一化后均值融合，回到原分类器单位，种子分歧令分类分数的行内 RMS 缩小约 21%–22%。128 个新字各在两日评分，保留每组全部 2,748 个唯一输出。v9、单种子 CDF、三种子 CDF、去掉分类器的 CDF 为 22.853、22.999、23.164、22.128。集成相对单种子 CDF +0.165±1.125，两日 −0.268/+0.598，分别改派 1,305/1,345 张。两日 Top‑32 均小升，仍需更强的 OCF 证据才能更新提交。543 个去重官方图对，Kaggle CPU 用时 754 秒。 [模型集成研究](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) · [跨分布评估](https://papers.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-classifier-ensemble/versions/1) · [实验报告](../outputs/siclink-task2-classifier-ensemble/task2_classifier_ensemble.json) |
| Task 1 mice2 实测近邻图的 OCF 效用 | 固定 v20 候选及渲染。在 48 个新留出字的已知跨日 16 次代理上，以 2/2 拆半的噪声白化放电距离选两个训练近邻，同时用 v20 胜者的轮廓选相同数量的视觉近邻。v20、纯放电近邻、20% 放电近邻、纯视觉近邻、20% 视觉近邻的 OCF 为 31.572、17.605、32.663、30.235、31.783。20% 放电近邻改选 10 张，相对 v20 +1.091±1.468，相对 20% 视觉对照 +0.880±1.455。3,998 个去重候选与近邻图对，103 个留出端点图对。Kaggle CPU 用时 2,611 秒，使用带标签训练录制。 [噪声白化距离](https://diedrichsenlab.org/pubs/Walther_Neuroimage_2016.pdf) · [期望效用选图](https://aclanthology.org/2022.emnlp-main.754/) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-native-witness-utility/versions/1) · [实验报告](../outputs/siclink-task1-native-witness-utility/task1_native_witness_utility.json) |

### Task 2 原 GPU 路径和 Colab

v20 的 mice2 Task 2 沿用 v5 在 Kaggle 双 T4 的第二张卡上独立训练、计算得分并完成指派的输出。v5 仅保存候选清单和提交包。原 12 个轮次的第 1、5、10、12 轮分类损失为 7.0394、2.6052、0.3807、0.2500。已准备同一单卡 CUDA 路径，并在计算 CDF 之前单独记录归档的逐像素复现数量和 Torch/CUDA 版本。通过 7897 能访问 Colab，用户反馈恢复后，两个免费账号分别尝试 T4 分配，均在分配阶段收到 503，两账户均无活跃会话。间隔数分钟和北京时间约 22:30 再次尝试，分配阶段仍收到 503。原 CUDA 路径的复现仍待 T4。 [v5 执行版本](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v5/versions/1) · [CUDA 入口](../scripts/kaggle_task2_mice2_rank_marginals_candidate.py)

五项对照均在 Kaggle CPU 完成。Task 2 分位数校准保留为待复现分支，当前正式包首选 v20，已核对归档 SHA-256。`ccjjllt` 正式最佳为 v14 的 25.45723，下一次正式提交时间为北京时间 2026-11-27 08:00，Kaggle 训练 GPU 的周配额于 2026-11-28 08:00 刷新。 [v20 ZIP](../outputs/siclink-candidate-v20-mice2-hanzi-colab/kaggle_submission.zip) · [正式记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-26 第二十九轮研究

| 对照 | 结果 |
| --- | --- |
| Task 3 mice2 字形特异的单试次放电协方差 | 三条实测支持试次拟合，在 12 维共享残差空间比较对角、共享相关、字形相关与按放电率及墨量匹配的打乱字形相关。固定 v20 七候选，在 session2/4 的 128 个新字上比较官方 OCF。v20、字形方差、共享相关、字形相关、打乱相关为 18.802、19.359、19.292、19.381、19.417。保持边际方差时，字形相关较打乱 −0.037±0.145 OCF，真实试次的对数密度两日增加 0.170 和 0.178 nats，较共享相关则下降 0.488 和 0.489 nats。各录制间的同字非对角相关可重复性为 0.0025–0.0134。268 个去重官方图对，Kaggle CPU 用时 132 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-glyph-covariance/versions/1) · [实验报告](../outputs/siclink-task3-glyph-covariance/task3_glyph_covariance.json) |
| Task 2 mice2 基于独立重复估计逐查询噪声精度 | session1/4 各用其余三日拟合，以源日重复噪声定义八个固定方向组。目标日的两次重复估计精度，另两次评分并互换。对同一 128 个新字跨两日评分，保持全库唯一指派和 25% CDF 权重。v9、CDF、方向精度、仅整体精度、匹配打乱方向的 OCF 为 21.706、22.395、21.561、21.946、21.480。方向精度相对 CDF −0.835±0.553，两日 −1.481/−0.188，相对打乱 +0.081±0.240。独立拆半的精度形状余弦相似度仅 0.0023/0.0050，方向修正的留出方差拟合也下降。419 个去重官方图对，Kaggle CPU 用时 354 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-repeat-precision/versions/1) · [实验报告](../outputs/siclink-task2-repeat-precision/task2_repeat_precision.json) |
| Task 1 HSK 认读字表与拟合字的覆盖 | 2025 考试大纲字表恰有 3,088 个字，独立第二字体验证高置信字形匹配 859/890 正确。该字表覆盖 2,082 个高置信拟合字的 86.50%，覆盖 308 个新高置信留出字的 84.42%。2021 字表为 83.44%，现有通用规范一级为 100%。等预算 700 时，2025 字表相对现有 Unihan 同日四次 OCF −0.011±0.098，已知跨日 16 次 +0.025±0.025，后者与匹配收录对照相同。按拟合字学习的字表分组先验使 1,000 候选的代理覆盖从 74.35% 至 79.22%，但 700 候选的实际 OCF 相对 Unihan 为 −0.768±0.408 和 +0.183±1.665。473 个去重官方图对，Kaggle CPU 用时 467 秒，使用训练数据。 [考试字表提取](https://github.com/harukicoder/hsk30) · [教育部 2021 标准](https://www.moe.gov.cn/jyb_xwfb/gzdt_gzdt/s5987/202103/t20210329_523304.html) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-hsk-inventory/versions/1) · [实验报告](../outputs/siclink-task1-hsk-inventory/task1_hsk_inventory.json) |
| Task 2 跨分类器的 CDF 置换环迁移 | session3/4 各留出，使用 112 个新字做配对 OCF。冻结种子 211 的指派模拟原归档，以独立种子 587 和两次放电产生 CDF 专属完整置换环，用另外两次独立评估并互换。冻结源分位数使用所有六种二重复组合，保持源、目标试次数匹配。种子 211、已知 211 分数加 CDF、种子 587、587 加 CDF 的 OCF 为 22.283、23.035、21.210、22.842。587 加 CDF 相对 587 提高 +1.633±0.657，相对 211 的差值为 +0.559±1.436。两天只复现 797/781 个原指派。24 个迁移环中仅一个在独立半组有正的 CDF 校正收益，加入代理分数后无正收益，最终无改派。524 个去重官方图对，第二版在 Kaggle CPU 用时 500 秒，首版因引用别名退出。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-cdf-proxy-cycles/versions/2) · [实验报告](../outputs/siclink-task2-cdf-proxy-cycles/task2_cdf_proxy_cycles.json) |
| Task 1 mice2 共享读出的等预算候选复核 | 112 个新评分字，固定 v20 排序器特征和渲染。将原候选中最低四项替换为本鼠、双鼠或打乱异鼠配对提出的新字。已知对应跨日 16 次时，真实双鼠提名的 101 个可信代理覆盖为 46.53%→52.48%，24 字的候选内真图选优为 59.412→63.104（+3.692±2.610）。原排序器仅改选 2/112，实际 OCF 30.090→29.575（−0.516±0.554）。同日四次的实际选图均相同，OCF 为 20.011。两条件共 228 个首选图对和 1,416 个选优图对。首版在全库提取 OCR 特征时进程被终止，修正版关闭梯度记录，Kaggle CPU 用时 1,644 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-shared-nominations/versions/2) · [实验报告](../outputs/siclink-task1-shared-nominations/task1_shared_nominations.json) |

北京时间约 22:55 查询四个免费 Colab 账号，活跃会话数和付费计算单元余额均为零。通过 7897 申请 T4 均返回 503，四次均未获得会话。免费资源的单独限制由 Google 动态调整且不公开，付费计算单元余额无法判断免费资源是否耗尽。 [Google Colab 资源说明](https://research.google.com/colaboratory/faq.html#resource-limits)

五项 Kaggle CPU 对照已完成，未生成新提交。v20 为下一个正式窗口的首选，`ccjjllt` 当前最佳正式 OCF 为 25.45723。 [v20 ZIP](../outputs/siclink-candidate-v20-mice2-hanzi-colab/kaggle_submission.zip) · [Kaggle 提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

### GPU 恢复后的群体放电表征

三项任务等权，即使 Task 1 达到 65，Task 2 和 Task 3 仍需平均 50 才能达到 55。既有四日匹配、独立重复去噪、MICrONS 和局部 DDPM 结果显示，单项微调难以补足这个差距。保留 Task 2 原 CUDA 路径复现，并优先用同架构对照检验原始放电时序的群体表征。SPARKS 的公开实现用放电间隔形成因果多单元注意力，跨录制示例使用 Allen Neuropixels 自然影片。拟用官方训练放电，比较真实原始时序、25 ms 窗内打乱时序和同预算的 25 ms 计数。用互斥同字重复学习共享表征和被遮蔽单元的预测，并直接在留出单试次及整日上比较 Task 3、Task 2 全库排序与官方 OCF。影片结果的刺激时长和类型与比赛不同。 [SPARKS 论文](https://doi.org/10.1101/2024.08.13.607787) · [跨录制示例](https://sparks.crick.ac.uk/demo_allen_multisession.html) · [公开实现](https://github.com/Iacaruso-lab/sparks)

## 2026-11-26 第三十轮研究

| 对照 | 结果 |
| --- | --- |
| Task 3 mice2 因果群体时序表征 | 三臂采用同一 22,780 参数模型、1,728 个拟合字、六轮训练和互斥配对放电。真实与 25 ms 内随机化保留相同粗计数，第三组将粗计数均匀摊回 5 ms。因果历史窗口为 5、15、35 ms，用独立同字重复对比和掩蔽单元预测共同训练。在 session2/4 的 128 个新单试次上，固定七候选的 v20、25 ms、打乱 5 ms、真实 5 ms OCF 为 18.017、17.435、17.598、17.743。真实较 25 ms +0.308±0.169，较打乱 +0.145±0.163，较 v20 −0.274±0.224。等预算重提名的真实组得 17.674，全库 Top‑32 在两日的收益方向不一致。549 个去重官方图对，Kaggle CPU 用时 305 秒。本文实现使用 5 ms 输入与因果群体特征，CUDA 入口待验证。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-event-attention/versions/1) · [实验报告](../outputs/siclink-task3-event-attention/task3_event_attention.json) |
| Task 2 mice2 毫秒级有向群体交互 | 每个目标日由其余三个带标签日拟合六个稳定群体方向。用 3/10/28 ms 的因果衰减和两个响应时间区间计算 90 维有向特征，固定 10% 融入 v9 加 CDF。对照保持每个单元的 25 ms 计数和整个群体每毫秒总放电，打乱时有 92.5%–93.2% 的事件移动。session2/4 各评分同一 128 个新字，所有方案保持 2,748 个唯一字形。v9、CDF、真实时序、打乱、仅粗计数的 OCF 为 17.656、18.375、17.917、19.347、19.381。真实时序相对打乱 −1.430±0.857，两个留出日均下降，Top‑32 和身份似然也弱于 CDF。620 个去重官方图对，Kaggle CPU 用时 445 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-causal-motifs/versions/1) · [实验报告](../outputs/siclink-task2-causal-motifs/task2_causal_motifs.json) |
| Task 2 直接复核 25 ms 计数轨迹 | 用此前固定的 10% 权重，在另外两个留出日 session1/3 对 160 个互斥评分字复核。仅用八步解析公式，相对于旧的毫秒展开在 256 次直接检查的最大绝对误差为 1.4×10⁻⁹。v9、CDF、计数轨迹、窗口内打乱计数顺序的 OCF 为 20.864、22.074、21.892、21.947。计数轨迹相对 CDF −0.182±0.894，两日 −0.784/+0.420，相对打乱 −0.055±1.184。Top‑32 和固定温度身份似然两日均低于 CDF。679 个去重官方图对，Kaggle CPU 用时 394 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-count-trajectory-recheck/versions/1) · [实验报告](../outputs/siclink-task2-count-trajectory-recheck/task2_count_trajectory_recheck.json) |
| Task 1 固定码位的灰度渲染 | 在 1,536 对已匹配训练字形上以同样的 19,473 参数模型和 400 次更新训练真实灰度残差与墨量分层打乱对照。96 个未见字固定相同码位，原全局渲染、Noto 原生配方、真实残差、打乱残差的 OCF 为 81.167、82.245、82.618、82.251。独立校准后的残差相对 Noto 原生配方 +0.361±0.287。冻结放电解码器在 48 个字的选择后，残差相对原生配方的同日四次与已知跨日 16 次代理分别为 +0.093±0.083、+0.028±0.158。439 个视觉图对加 606 个放电选字图对，Kaggle CPU 用时 527 秒。 [条件字形转换研究](https://openaccess.thecvf.com/content/CVPR2021/html/Xie_DG-Font_Deformable_Generative_Networks_for_Unsupervised_Font_Generation_CVPR_2021_paper.html) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-grayscale-font-residual/versions/1) · [实验报告](../outputs/siclink-task1-grayscale-font-residual/task1_grayscale_font_residual.json) |
| Task 1 mice2 官方 PP-OCRv6 读出 | 固定 27,550 张公共 Noto 候选、简化岭回归、Unihan 先验和渲染，使用 2,428 个拟合字及 320 个新留出字，其中 112 个预选精确评分。冻结官方 Apache-2.0 小型识别模型，从 18,710 类 CTC 输出层之前提取 120 维识别特征。轮廓加既有单字 OCR、PP 空间融合、PP 全局融合的同日四次 OCF 为 15.849、20.017、20.395，全局增量 +4.546±1.518。已知同字跨日 16 次为 27.811、32.275、34.403，全局增量 +6.593±2.387。16 次高置信 Top‑32 从 66.55% 到全局 75.61%、空间 77.00%。505 个去重官方图对，Kaggle CPU 用时 799 秒。此轮比较同样的简化读出，直接的 v20 增量需要独立检验。 [PP-OCRv6 论文](https://arxiv.org/abs/2606.13108) · [官方模型固定版本](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/commit/b8f84f0b80c529de40b4fbb3544b84fa7233a513) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-ppocrv6-readout/versions/1) · [实验报告](../outputs/siclink-task1-ppocrv6-readout/task1_ppocrv6_readout.json) |
| Task 1 mice1 官方 PP-OCRv6 的跨鼠复核 | 同一全库、拟合字数、同日四次和已知跨日 16 次协议，换为另一只小鼠和 320 个互不重叠的新留出字。对 112 个预选字评分。轮廓加旧 OCR、PP 空间融合、PP 全局融合的同日 OCF 为 22.392、21.724、23.751，全局 +1.359±1.882。跨日条件为 37.370、45.681、43.688，空间 +8.310±2.794、全局 +6.318±2.792。跨日高置信 Top‑32 两种融合均由 82.35% 增到 87.89%。506 个去重官方图对，Kaggle CPU 用时 755 秒。比较的是相同简化读出，未包含 mice1 正式 MICrONS 分支。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-ppocrv6-mice1/versions/1) · [实验报告](../outputs/siclink-task1-ppocrv6-mice1/task1_ppocrv6_readout.json) |
| Task 1 mice2 PP-OCRv6 直接对比 v20 | 复用四折排序器和首个 CPU 拟合分支，在 128 个与 PP 初筛互斥的留出字上固定每行候选数量与渲染。既有 OCR 内以 50% 混入全局特征。v20、融合、仅改分、仅改提名的同日四次 OCF 为 25.767、24.884、24.851、25.817。融合相对 v20 −0.883±0.462，四日均低。在已知同字的跨日 16 次中分别为 33.938、35.130、34.830、34.049，融合 +1.192±1.415。115 个可信代理的跨日 Top‑32 从 62.61% 至 68.70%，等预算池覆盖从 50.43% 至 52.17%。32 个此前未用 CPU 评分的字在两条件下的融合增量分别为 +0.063±0.340、−0.559±4.311。183 个同日、202 个跨日去重官方图对，Kaggle CPU 用时 1,692 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-ppocrv6-v20/versions/1) · [实验报告](../outputs/siclink-task1-ppocrv6-v20/task1_mice2_ppocrv6_v20.json) |

七项远程 CPU 对照已完成。PP-OCRv6 在两鼠简化读出的 16 次代理中均改善，但目前直接 v20 对照的两种重复条件方向不一致。已准备复用 v20 全量四折的打包入口，完成静态编译和存档摘要核对，尚无正式打包结果。当前仍选 v20，正式最佳为 25.45723。北京时间 11 月 26 日约 23:50 免费 T4 复试仍返回 503。Kaggle 提交于 11 月 27 日 08:00 开放，GPU 周配额于 11 月 28 日 08:00 刷新。 [打包入口](../scripts/kaggle_task1_mice2_ppocrv6_candidate.py) · [Kaggle 提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-27 第三十一轮研究

| 对照 | 结果 |
| --- | --- |
| Task 3 mice2 跨日放电率转移 | 对新 144 个真实单试次保持 v20 的七候选。所有替代分支均使用三条同日支持试次和同样的 Poisson 观察，依次加入全库平均、视觉先验、跨日逐单元校正和 64 维跨单元校正。v20、全库平均、视觉、逐单元、跨单元、保留逐单元但打乱跨单元对应的 OCF 为 17.294、18.458、18.901、18.906、18.538、19.029。视觉相对 v20 +1.607±0.939，两日 +2.122/+1.092。逐单元相对 v20 +1.613±0.940。跨单元分支提高了身份排名，但相对打乱 −0.491±0.561 OCF，其单试次预测对数似然也弱于逐单元。364 个去重官方图对，Kaggle CPU 用时 183 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-peer-rate-transfer/versions/1) · [实验报告](../outputs/siclink-task3-peer-rate-transfer/task3_peer_rate_transfer.json) |
| Task 3 跨日放电率独立复核 | 另两个留出日 session2/4 各 96 个互斥字，保持同一 v20 七候选、三条支持试次、似然融合、渲染及效用。v20、全库平均、视觉、逐单元 OCF 为 17.258、17.155、16.685、16.675。视觉相对 v20 −0.573±0.481，两日 +0.007/−1.154。逐单元比视觉的预测对数似然提高 8.524/9.666 nats，实际 OCF −0.010±0.044。233 个去重官方图对，Kaggle CPU 用时 148 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-peer-rate-recheck/versions/1) · [实验报告](../outputs/siclink-task3-peer-rate-recheck/task3_peer_rate_recheck.json) |
| Task 2 mice2 早期群体状态的目标日迁移 | 每个目标日用两条重复的 0–25 ms 群体放电估计随后响应偏差，并在另一半上应用，按相同程序互换。源日冻结和字内错配为对照。session2/4 的 128 个新字，完整一对一指派下，v9、CDF、真实目标修正、源冻结、目标错配 OCF 为 19.003、19.440、18.915、19.809、19.005。真实目标比 CDF −0.524±0.291，两个留出日均下降。真实目标在另一半解释 1.90%–1.95% 和 3.13%–3.39% 的响应方差，优于源冻结和错配，收益没有转化为身份似然或 OCF。470 个去重官方图对，Kaggle CPU 用时 341 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-early-state-transport/versions/1) · [实验报告](../outputs/siclink-task2-early-state-transport/task2_early_state_transport.json) |
| Task 1 mice2 重复噪声白化与 PP-OCRv6 二阶读出 | 固定 27,550 个公开候选及已有视觉分数。2,428 个拟合字在各录制按互斥 2/2 重复估计 96 维可靠信号。在与两次 PP 初筛及三项旧实验互斥的新 320 字中，配对精确评分 112 字。PP 岭回归、白化线性、均值平方、互斥重复乘积的同日四次 OCF 为 19.328、19.522、20.648、20.177。已知同字跨日 16 次的 OCF 为 25.285、30.293、30.881、30.386，白化线性提高 +5.008±1.878，常规平方 +5.596±2.056，互斥乘积 +5.101±2.122，白化解释了大部分增量。跨日高置信 Top‑32 由 69.79% 提到 74.31%、75.35%、76.39%。421 个去重官方图对，Kaggle CPU 用时 1,397 秒。此处仍是简化读出。 [重复噪声协方差研究](https://diedrichsenlab.org/pubs/Walther_Neuroimage_2016.pdf) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-ppocrv6-repeat-kernel/versions/1) · [实验报告](../outputs/siclink-task1-ppocrv6-repeat-kernel/task1_ppocrv6_repeat_kernel.json) |
| Task 1 mice2 用官方 OCF 训练 PP 感知重排 | 固定 v20 候选池、渲染与 128 个直接对照留出字，用 80 个独立训练评估字的 1,386 个官方图对拟合三种等容量残差排序器。v20、无 PP、真实 PP、按查询置换 PP 的同日四次 OCF 为 25.767、23.878、24.662、25.097，真实相对 v20 −1.105±1.599。已知对应的跨日 16 次为 33.938、34.721、37.025、35.288，真实相对 v20 +3.087±1.531，相对无 PP +2.304±1.407，相对打乱 +1.737±1.297。32 个此前未用 CPU 评分的字，真实跨日相对 v20 −0.740±3.465。409 个端点图对，Kaggle CPU 用时 2,281 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task1-ppocrv6-ranker/versions/1) · [实验报告](../outputs/siclink-task1-ppocrv6-ranker/task1_mice2_ppocrv6_ranker.json) |

五项 Kaggle CPU 对照均已完成，临时监控已清理。v20 保留为北京时间 11 月 27 日 08:00 的提交首选，正式最佳为 25.45723。公开榜分数留待下一次实际提交确认。11 月 27 日约 00:41 免费 Colab T4 再次返回 503，Kaggle GPU 周配额仍于 11 月 28 日 08:00 刷新。 [v20 ZIP](../outputs/siclink-candidate-v20-mice2-hanzi-colab/kaggle_submission.zip) · [Kaggle 提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions)

## 2026-11-27 第三十二轮研究

| 对照 | 结果 |
| --- | --- |
| Task 3 mice2 一对一指派直接进入训练目标 | 同一 18,816 参数模型和 800 次更新，采样相同的 128 字小组，保留三条支持重复和一条真实查询。两臂的视觉代价与交叉熵固定，只比较在损失内逐行或完整一对一地找竞争项。192 个新留出单试次、固定七候选，v20、逐行训练、整组训练的 OCF 为 19.813、19.988、20.069，整组相对逐行 +0.081±0.080。两种模型推理再做平衡的 OCF 为 19.484、19.577。真实整组训练增加约 0.29 个百分点完整 Top‑32，独立 249 字上的 Top‑32 不升。261 个去重图对，Kaggle CPU 用时 197 秒。 [结构化损失论文](https://www.jmlr.org/papers/v6/tsochantaridis05a.html) · [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task3-assignment-objective/versions/1) · [实验报告](../outputs/siclink-task3-assignment-objective/task3_assignment_objective.json) |
| Task 2 mice2 整表期望 OCF 训练 | 四个内层跨日环节各用互斥的 40×8 个精确图对，比较逐行概率和完整 2,748×2,748 稀疏平衡概率训练同一四参数轮廓混合。外层 session1/4 对同一批 128 个新字，固定最终一对一指派。CDF 基线、固定轮廓、逐行训练、整表训练的 OCF 为 19.236、19.191、19.379、19.353。整表相对逐行 −0.026±0.457，相对 CDF +0.118±0.122，两日 +0.343/−0.108。整表训练大幅降低列质量误差，内层八候选的真图选优仍高于原决策。1,280 个训练、301 个外层去重官方图对，Kaggle CPU 用时 1,017 秒。 [Kaggle 对照](https://www.kaggle.com/code/ccjjllt/siclink-task2-structured-ocf/versions/1) · [实验报告](../outputs/siclink-task2-structured-ocf/task2_structured_ocf.json) |

### v20 正式提交

北京时间 2026-11-27 08:49:42 以参赛账号提交已核验 v20，回执 `58973821`。08:58:29 核对评分完成，公开 OCF 为 **26.53844**，较 v14 的 25.45723 增加 **1.08121**，较首次 15.04410 增加 **11.49434**。09:01 查询公开榜在 91 支队伍中位列第 42。相对正式 v14，Task 1／2／3 中改变的 PNG 载荷数为 743／0／4,749，24,708 条保持逐字节一致。Kaggle 返回总分，Task 2 的提交载荷保持一致，无法拆分 Task 1 和 Task 3 的贡献。次日提交窗口和 GPU 周配额均于北京时间 11 月 28 日 08:00 刷新。 [官方回执](../outputs/siclink-candidate-v20-mice2-hanzi-colab/official_submission_58973821.json) · [提交记录](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/submissions) · [公开榜](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/leaderboard)

北京时间约 09:08，通过 7897 查询的四个免费 Colab 账号均无活动会话，逐一申请 T4 均返回 HTTP 503。免费 GPU 使用限制和恢复时间仍未知。`rainer` 的只读查询显示一个活动 L4，余额 690.72 计算单元，当前速率 2.33 单元/小时。Kaggle v21 的现有版本已挂载正式比赛数据和私有存档 `ccjjllt/siclink-v20-immutable-base`，CUDA 复现继续使用这两项输入。 [Colab 资源说明](https://research.google.com/colaboratory/faq.html#resource-limits) · [Kaggle v21](https://www.kaggle.com/code/ccjjllt/siclink-candidate-v21-mice2-task2-cdf/versions/1)
