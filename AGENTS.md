# SiClink 研究工作约定

## 运行环境

- 本机用于编辑代码、管理文件和查看产物。数据探索、特征统计、训练、推理和 OCF 计算在远端运行。
- 优先用 Kaggle Notebook。轻量数据处理、线性基线、字图库检索和小批量 OCF 用 Kaggle CPU。深度模型及批量评估用 Kaggle GPU。Kaggle 的双 T4 能满足本轮实验时继续用 Kaggle。
- 给 Notebook 添加比赛 Data 输入，从 `/kaggle/input` 查找正式数据目录。把需要下载的日志、权重和提交包写入 `/kaggle/working`。用比赛参赛账号查询分数和提交记录。
- 以研究迭代为主。记录数据划分、实验参数、用时和分任务 OCF。沿用官方数据与评估代码，工程化工作按实验需要推进。

## Colab 备用

Kaggle GPU 的配额、可用性或性能不足时，使用 Colab。免费账号优先试 T4。`rainer` 是用户的 Google AI Ultra 账号，有 2,000 credits，可在确实需要更强 GPU 时使用。启动后以当前会话分配到的卡为准。

本机连接 Colab 使用 `127.0.0.1:7897`。用户在直连时遇到 TLS 重置，在 `7900` 上遇到 `SSLEOF`。`colab-accounts` 查看账号，`colab-use` 选择默认账号。批量操作时可用 `COLAB_ACCOUNT=<账号>` 指定单次命令的账号，保持会话创建、查看和停止使用同一账号。

```sh
export COLAB_PROXY='http://127.0.0.1:7897'
colab-accounts
colab-use ccjjllt0
colabx new -s gpu1 --gpu T4
```

`colabx` 会优先继承已设置的 `HTTP_PROXY` 和 `HTTPS_PROXY`。如果当前 shell 里有旧值，运行前将两者设为 `$COLAB_PROXY`。同时设置了小写的 `http_proxy` 或 `https_proxy` 时，也使用同一地址。

如果从 Drive 读取数据，可以先把 `data/raw/` 中需要的内容放在桌面版 Google Drive 的私人目录里，待同步完成后在 Colab Notebook 挂载。数据目录应当只有获准参与比赛的账号可访问。把实际目录传给脚本，避免依赖本机盘符。[比赛数据规则](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/rules)

```python
from google.colab import drive

drive.mount("/content/drive")
```

也可以在 Colab 直接从 Kaggle 拉取数据。当前 `colabx exec --env` 会把环境变量展开进执行代码，再把代码记入本地会话历史。需要 `KAGGLE_API_TOKEN` 时，可以将凭证文件上传到私人临时会话，待下载脚本读取后立即删除。Drive 里已备齐数据时可直接挂载。

```sh
colabx upload -s gpu1 "$HOME/.kaggle/access_token_ccjjllt_check" .siclink-kaggle-token
```

下载脚本可将凭证只传给 Kaggle 子进程，并在下载前移除会话中的文件。

```python
from pathlib import Path
import os
import subprocess

token_path = Path("/.siclink-kaggle-token")
try:
    kaggle_env = {**os.environ, "KAGGLE_API_TOKEN": token_path.read_text().strip()}
finally:
    token_path.unlink(missing_ok=True)

subprocess.run([
    "kaggle", "competitions", "download",
    "-c", "siclink-mouse-visual-reconstruction-bci-challenge",
    "-p", "/content/siclink-data",
], env=kaggle_env, check=True)
```

以下是用户已有项目的后台保活、产物回收和状态查看命令，应从脚本所属项目运行。

```sh
mkdir -p /tmp/campaign2
nohup ./scripts/colab_keeper.sh --loop 300 >> /tmp/campaign2/colab_keeper.log 2>&1 &
nohup ./scripts/colab_harvest_safe.sh --loop 1800 >> /tmp/campaign2/colab_harvest.out 2>&1 &
```

运行期间查看状态。

```sh
colabx exec -s gpu1 -f scripts/colab_status_remote.py --timeout 120
./scripts/watch_colab.sh -s gpu1 --once
```

产物和断点已同步后可结束会话。

```sh
colabx stop -s gpu1
```

用户观察到免费 T4 会话约 2–7 小时可能回收，运行时长取决于实际分配。启动长实验时定期保存 SiClink 的权重、训练进度、随机种子和划分，并同步到私人持久目录。会话回收后用相同配置和最近的断点继续。上面已有的 GA 项目使用 `work/runs/ga_state`，SiClink 实验的断点位置由各自的训练脚本决定。
