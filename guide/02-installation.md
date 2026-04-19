# 02 - 安装与设置

## 环境要求

- **Python**: 3.10+
- **操作系统**: macOS, Linux
- **可选**: GPU (CUDA) 用于加速

## 安装方式

### 方式一：uv（推荐）

```bash
# 克隆仓库
git clone https://github.com/minihellboy/factorminer.git
cd factorminer

# 基础安装（开发工具）
uv sync --group dev

# 安装 LLM 支持（OpenAI, Anthropic 等）
uv sync --group dev --extra llm

# 完整安装
uv sync --group dev --all-extras
```

### 方式二：pip

```bash
# 基础安装
python3 -m pip install -e .

# LLM 支持
python3 -m pip install -e ".[llm]"

# 完整安装
python3 -m pip install -e ".[all]"
```

## 安装选项

| 选项 | 说明 | 命令 |
|------|------|------|
| `dev` | 开发工具 + 基础依赖 | `--group dev` |
| `llm` | LLM 提供商支持 | `--extra llm` |
| `gpu` | GPU 加速 (Linux only) | `--extra gpu` |
| `all` | 所有选项 | `--all-extras` |

## 目录结构

安装后，项目结构如下：

```
factorminer/
├── factorminer/      # 主包
├── guide/            # 本指南
├── code-wiki/        # 代码知识库
├── output/           # 输出目录（运行后生成）
├── configs/         # 配置文件
└── ...
```

## 配置 API Keys（可选）

如需使用真实 LLM，需要配置 API Key：

```bash
# 设置 OpenAI API Key
export OPENAI_API_KEY="sk-..."

# 或创建 .env 文件
echo "OPENAI_API_KEY=sk-..." > .env
```

## 快速验证安装

```bash
# 查看帮助
uv run factorminer --help

# 运行演示（无需 API Key）
uv run python run_demo.py
```

## Docker 方式（可选）

```bash
# 构建镜像
docker build -t factorminer .

# 运行
docker run --rm factorminer --help
```

## 常见问题

### Q: 提示 "PyTorch not installed"

如需 GPU 支持，确保正确安装：

```bash
# Linux with CUDA
uv sync --group dev --all-extras

# macOS (仅 CPU/MPS)
uv sync --group dev --extra llm
```

### Q: 演示运行报错 "circular import"

这是已知的循环导入 bug，不影响基本功能。使用 `--mock` 模式：

```bash
uv run factorminer --cpu mine --mock -n 1 -b 4 -t 5
```

### Q: GPU 不可用

FactorMiner 会自动回退到 CPU：

```
Device: cpu (CUDA not available)
```

## 下一步

- [核心概念](03-core-concepts.md) - 理解因子、IC、表达式树
- [架构概述](04-architecture.md) - 理解系统架构
