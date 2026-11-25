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
