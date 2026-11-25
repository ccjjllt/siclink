# 比赛速查

核对日期为 2026 年 11 月 24 日，时区为北京时间。Kaggle 显示这是 SiClink 主办的 Community 比赛，当时有 54 支队伍。总奖金为 10,000 美元，最终截止为 **2027 年 1 月 20 日 18:00**。每队最多 5 人，每个 UTC 日可提交 1 次。[赛题](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge) · [时间表](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/overview/timeline) · [规则](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/rules)

## 数据与任务

数据来自两只小鼠，各有 5 个 session。每次刺激展示 250 ms，spike 时间以 30 kHz 采集。同一只小鼠的 unit 索引跨 session 一致，两只鼠分别有 134 和 78 个 unit。共享训练集覆盖两只鼠的 `session1`–`session4`，包含 2,748 种汉字，每个鼠、session、字符组合有 4 次带标签的 trial，共 87,936 条。正式数据根目录为 `siclink_chinese_character_decoding_dataset/`，图像标签在 `train/pictures.npz`。[Data](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data)

| 任务 | 字符 | 记录条件 | 每张预测的 trial 数 | 预测图数 |
| --- | --- | --- | ---: | ---: |
| Task 1 | 340 个未见字 | `session1`–`session4` | 20 | 2,720 |
| Task 2 | 2,748 个已见字 | 跨天 `session5` | 4 | 5,496 |
| Task 3 | 2,748 个已见字 | `session1`–`session4` | 1 | 21,984 |

每个 `sample_id` 输出一张白字黑底的 64×64 单通道灰度 PNG。OCF（Oriented Contour Fidelity）对齐小幅几何变化后，比较多尺度、多方向的轮廓，范围 0–100，越高越好。每个任务分别取平均，三项均值等权。全部 30,200 张计入公开榜，主办方可复核成绩和复现材料。核对时公开榜最高分为 54.01486。[评估](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/overview/evaluation) · [榜单](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/leaderboard) · [规则](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/rules)

本地已核对 87 个解压文件（约 250 MB）和正式数据的 26 个记录块。块内的 `group_ptr` 将样本对应到 trial，`trial_ptr` 将 trial 对应到 spike 事件。训练块中的 `target` 是 `train/pictures.npz` 的 `images` 数组的零基索引。

## 官方与社区代码

- `participant_package_en/participant_package_en/` 提供 `src/dataloader.py`、`src/spike_counts.py`、CPU/CUDA OCF、条件 DDPM 和小演示集。默认 1 ms loader 返回 `[重复, unit, 时间 bin]` 二值占用矩阵。`kaggle_submission_adapter/kaggle_submission_adapter/convert_to_kaggle.py` 把 PNG 压缩包转换成 Kaggle 需要的 `kaggle_submission.zip`，内含 `sample_id,prediction` 两列的 CSV，图像为 base64 PNG。[Data](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data)
- 公开 Code 查询返回两个 notebook。Avik Das 的实现使用 10×25 ms 时间窗、session 归一化和方向轮廓损失，可参考时序特征与图像解码。Moran Zavdi 的 CPU notebook 适合参考 PSTH、session 漂移和 OCF 校准。[Code](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/code) · [Avik Das](https://www.kaggle.com/code/avikdas567/spatio-temporal-cortical-visual-bci-reconstruction) · [Moran Zavdi](https://www.kaggle.com/code/moranzavdi/siclink-ocf-audit-v1-drift-14-point-verifier)
- Discussions 查询返回一个 OCF 帖。发帖人用样本观察到，将抗锯齿灰度真值直接当成预测，与先阈值化再评分，结果差异明显。官方规则对真值使用 `>=128` 二值化，预测保留归一化后的灰度。可用 Kaggle 上的官方 OCF 对照灰度预测与不同阈值的二值预测，按任务报告结果。[OCF 讨论](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/discussion/760906) · [评估](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/overview/evaluation)

榜首 Bowen1314 的 54.01486 和第二名 Oleg Kurilov 的 51.11346 暂未找到对应的公开实现。Moran Zavdi 的 22.63617 能与公开榜成绩和预测产物对应，公开 Notebook 主要校验并重打包既有预测。Avik Das 的 Notebook 则公开了从训练到提交文件生成的实现。[公开榜](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/leaderboard) · [Moran Notebook](https://www.kaggle.com/code/moranzavdi/siclink-ocf-audit-v1-drift-14-point-verifier) · [Avik Notebook](https://www.kaggle.com/code/avikdas567/spatio-temporal-cortical-visual-bci-reconstruction)

2026-11-25 约 14:45（北京时间）重新查询，榜上有 71 支队伍，前三名为 Bowen1314 54.05689、Frozen Kiwi 52.91478、Oleg Kurilov 51.42620。公开 Code 仍返回上述两个 Notebook，Discussions 仍只有 OCF 帖。Moran 的 Notebook 可核验已打分的附件，榜首三个成绩暂无对应的公开实现。 [公开榜](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/leaderboard) · [Code](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/code)
