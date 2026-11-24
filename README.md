# SiClink 小鼠视觉重建

给定小鼠视觉皮层在刺激出现后 250 ms 内的 spike，重建它看到的 64×64 灰度汉字图像。按未见字符、跨天记录、单次试验分别研究。数据探索、训练、推理和评分都在远端运行。 [Kaggle 比赛](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge)

## 开始

Kaggle Notebook 添加比赛 Data 后，从 `/kaggle/input` 读取正式数据。探索和轻量基线用 Kaggle CPU，需要加速时用双 T4。只有 Kaggle 资源不足且实验需要 GPU 时再切到 Colab，账号、代理和断点操作见 [AGENTS.md](AGENTS.md)。

- `scripts/inspect_data.py` 汇总数据目录并检查指定鼠和 session。放到 Kaggle 环境后，通过 `--data` 传入包含 `catalog.json` 的目录。
- `scripts/kaggle_baseline.py` 可直接作为一个 Kaggle Python Script 版本运行。它用 25 ms spike counts，从训练图像拟合 ridge 和字形模板，然后将 `kaggle_submission.zip` 写到 `/kaggle/working`。启用双 T4 时，两只鼠分别在一张卡上运行。 [Kaggle 运行](https://www.kaggle.com/code/ccjjllt/siclink-spike-count-baseline)

工作区的 `data/raw/` 已解压全量数据和官方参赛者包，可查看官方格式和代码。官方 DDPM 与 OCF 的依赖见参赛者包的 `README.md`。

## 目录

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | Kaggle 原始数据、官方参赛者包和提交适配器 |
| `data/processed/` | 可重算的 spike 特征与本地划分清单 |
| `notebooks/` | 探索样本、时序活动、session 漂移与重建可视化 |
| `scripts/` | 可从命令行重复运行的数据工具，后续放训练和推理脚本 |
| `notes/` | 赛题资料、研究设计与实验记录 |
| `outputs/` | OCF 记录、预测样图、模型权重和提交包 |

用 [赛题笔记](notes/competition.md) 查官方规则与公开资料，用 [实验计划](notes/experiments.md) 记录划分、特征、模型和分任务成绩。
