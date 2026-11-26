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
