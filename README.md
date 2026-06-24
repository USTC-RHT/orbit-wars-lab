# 依赖安装与启动

## 方式一：使用已有 conda 环境

如果本机已经有 `orbit-wars-lab` conda 环境，依赖已经装在：

```text
/home/dministrator/miniconda/envs/orbit-wars-lab
```

直接启动前后端：

```bash
cd orbit-wars-lab
bash scripts/dev.sh --conda orbit-wars-lab
```

## 方式二：新机器用 conda 安装依赖

```bash
cd orbit-wars-lab
conda create -n orbit-wars-lab python=3.12 -y
conda activate orbit-wars-lab
pip install -r requirements.txt
pnpm install
bash scripts/dev.sh --conda orbit-wars-lab
```

Python 后端依赖来自：

```text
requirements.txt
pyproject.toml
```

前端依赖来自：

```text
package.json
viewer/package.json
pnpm-lock.yaml
```

## 方式三：新机器用 venv / make 安装依赖

机器上需要有 `python3.12`、`pnpm` 和 `make`。

```bash
cd orbit-wars-lab
make setup
make dev
```

`make setup` 会创建 `.venv`，安装 Python 依赖和 pnpm 依赖；`make dev` 会调用 `scripts/dev.sh` 启动前后端。

## 方式四：Docker 启动

Docker 会在镜像里安装依赖，本机不需要单独装 Python 包：

```bash
cd orbit-wars-lab
docker compose up --build
```

打开浏览器访问：

```text
http://localhost:6001
```

本地开发模式会同时启动：

- 后端：`http://localhost:8000`
- 前端：`http://localhost:6001`

如果本机 `8000` 或 `6001` 已被占用，可以临时换端口：

```bash
BACKEND_PORT=8001 VIEWER_PORT=6002 bash scripts/dev.sh --conda orbit-wars-lab
```

# Orbit Wars Lab

这是一个本地 Orbit Wars 评估平台：可以在浏览器里选择 agent、运行 2P/4P 对局、看回放、跑 tournament，并用 TrueSkill 记录本地评分。

当前仓库已经整理成偏“可运行评估平台”的形态：保留前后端、agent zoo、启动脚本和 Docker 配置；删除了训练数据下载脚本、历史 runs/replays/submissions/logs、测试目录以及多余的 `agents/mine` 实验版本。

## 当前保留的 Agent

Agent 统一放在 `agents/` 下，按来源分三类：

```text
agents/
  baselines/   Kaggle / 项目内基准 agent
  external/    外部公开策略 agent
  mine/        本项目保留的本地模型 agent
```

当前 `agents/mine` 只保留四个版本：

```text
bc-feature-row-step50000-onnx
bc-feature-row-step70000-onnx
fixed-pool-tgtmask-0526-ckpt32700-onnx-producer-hybrid
fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid
```

每个 agent 目录通常包含：

```text
agent.yaml      # agent 元信息：名字、标签、是否禁用等
main.py         # Kaggle agent 入口，必须提供 agent(observation, configuration)
models/         # ONNX / normalizer / manifest 等模型文件
```

后端启动后会扫描 `agents/{baselines,external,mine}`，凡是目录里有 `main.py` 的都会出现在页面的 agent picker 中。

## 使用流程

1. 启动平台。
2. 打开 `http://localhost:6001`。
3. 在 Quick Match 中选择 2 个或 4 个 agent。
4. 点击运行，平台会调用 Kaggle `orbit_wars` 环境执行对局。
5. 对局结束后可以查看 replay、比分和 agent 状态。
6. 在 Tournaments 中可以跑 round-robin 或 gauntlet，用于批量比较 agent。

运行过程中产生的数据会写到：

```text
runs/
  trueskill.json          # 本地 TrueSkill 评分
  <date-id>/              # 每次 tournament / quick match 的结果
    run.json
    config.json
    results.json
    trueskill.json
    replays/*.json
```

这个目录默认不需要提前存在，平台会按需创建。

## 总体架构

项目可以理解成四层：

```text
浏览器 UI
  ↓
FastAPI 后端 API
  ↓
Kaggle Environment 对局执行 / Tournament 调度
  ↓
agents/ 中的各个 main.py 策略
```

### 前端：`viewer/`

`viewer/` 是 Vite + TypeScript 单页应用，负责：

- agent 选择
- quick match 创建
- tournament 创建与进度展示
- replay 列表和回放展示
- ratings / TrueSkill 展示
- Kaggle 集成页面

开发模式下由 Vite 在 `6001` 端口提供页面，并把 `/api/*` 代理到后端 `8000`。

生产 / Docker 模式下，`viewer/dist` 会被 FastAPI 当作静态文件直接挂载到 `/`。

### Vendored Kaggle Web Core：`web/`

`web/core` 是前端回放展示相关的 vendored Kaggle core。`viewer` 依赖它来展示 Orbit Wars replay。

这个目录不是后端服务本身，但前端构建时会用到。

### 后端：`orbit_wars_app/`

后端是 Python 3.12 + FastAPI。核心文件：

```text
orbit_wars_app/main.py              # FastAPI app，挂载 API 和静态 viewer
orbit_wars_app/api.py               # /api 路由
orbit_wars_app/discovery.py         # 扫描 agents/，构建 agent 列表
orbit_wars_app/match.py             # 单局 match 执行
orbit_wars_app/tournament.py        # round-robin / gauntlet 调度
orbit_wars_app/replay_store.py      # replay 保存和读取
orbit_wars_app/trueskill_store.py   # TrueSkill 持久化
orbit_wars_app/agent_subprocess.py  # faithful 模式下的 agent 子进程
orbit_wars_app/kaggle_*.py          # Kaggle 登录、submission、replay 相关功能
```

主要 API：

```text
GET  /api/health              # 健康检查
GET  /api/agents              # agent 列表
GET  /api/ratings             # TrueSkill 排名
GET  /api/runs                # 本地运行记录
GET  /api/replays             # replay 列表
POST /api/tournaments         # 创建 tournament
```

Quick Match 也是通过 `POST /api/tournaments` 创建，只是前端会在配置里设置 `is_quick_match: true`。

### 对局执行：`match.py`

对局最终由 Kaggle `kaggle-environments` 的 `orbit_wars` 环境执行。

平台支持两种执行模式：

- `fast`：在当前 Python 进程内直接运行 agent，速度快，适合大量评估。
- `faithful`：每个 agent 单独起子进程，通过 HTTP 方式与 Kaggle env 通信，更接近 Kaggle 线上运行方式。

大多数本地评估默认用 `fast`。

### Tournament 调度：`tournament.py`

Tournament 支持两种形态：

- `round-robin`：所有 agent 两两对战。
- `gauntlet`：一个 challenger 对多个 opponent。

每场 match 结束后会：

1. 保存 replay。
2. 写入 results。
3. 更新 `runs/trueskill.json`。
4. 给前端返回进度。

并行评估时，后端会用 `ProcessPoolExecutor` 分发 match，避免单个 agent 或单场对局拖慢整个 tournament。

## Docker 架构

Dockerfile 是多阶段构建：

```text
Stage 1: node:20-alpine
  安装 pnpm 依赖
  构建 viewer/dist

Stage 2: python:3.12-slim
  预编译 Python wheel

Stage 3: python:3.12-slim
  安装运行依赖
  复制 orbit_wars_app/
  复制 agents/
  复制 viewer/dist
  启动 uvicorn
```

容器启动命令：

```bash
uvicorn orbit_wars_app.main:app --host 0.0.0.0 --port 8000
```

`docker-compose.yml` 会挂载：

```text
./agents:/app/agents
./runs:/app/runs
```

所以你在宿主机修改 agent 后，容器可以直接看到；评估结果也会落回宿主机的 `runs/`。

## Kaggle Token

不配置 Kaggle token 也能跑本地 quick match 和 tournament。

只有这些功能需要 token：

- 查看 Kaggle submission
- 从 Kaggle 拉取 replay
- 从 UI 上传 submission

推荐在 UI 的 Settings 页面配置。也可以在 Docker 中挂载：

```yaml
- ~/.kaggle:/home/app/.kaggle
```

## 如何添加自己的 Agent

复制一个最小 agent：

```bash
cp -r agents/baselines/starter agents/mine/my-agent
```

修改：

```text
agents/mine/my-agent/main.py
agents/mine/my-agent/agent.yaml
```

刷新页面后，新的 agent 会出现在选择器里。

最小入口形式：

```python
def agent(observation, configuration=None):
    return []
```

返回值是 Kaggle Orbit Wars 的 action list。

## 仓库清理状态

当前仓库刻意不保留：

- 训练数据下载脚本
- 历史 runs
- 历史 replay
- 历史 submissions
- tests 目录
- 多余的 mine agent 实验版本

这让仓库更适合作为一个轻量、可复现的评估平台上传到 Git 仓库。

## 常见问题

### 打开页面后没有 agent

检查：

```bash
find agents -maxdepth 3 -name main.py
```

每个 agent 目录必须有 `main.py`。

### Docker 权限不对

Linux 通常默认 UID/GID 为 1000。如果宿主机用户不是 1000，可以创建 `.env`：

```bash
echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env
```

然后重启：

```bash
docker compose down
docker compose up --build
```

### ONNX agent 在本地能跑，Kaggle submission 不能跑

Kaggle submission 必须把运行时依赖一起打包，尤其是：

```text
onnxruntime/
onnxruntime/capi/*.so
```

本仓库里的本地评估 agent 可以依赖当前环境；提交到 Kaggle 时要另外确认 tar 包是自包含的。
