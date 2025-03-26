# BUPT 人工智能引论 聚类算法实验

## 运行

可使用 uv 运行或使用 venv 手动安装依赖运行。

### 使用 [uv](https://docs.astral.sh/uv/)

依照 https://docs.astral.sh/uv/getting-started/installation/ 安装

```bash
uv venv
source .venv/bin/activate
uv run main.py
```

### 使用 venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib scikit-learn
python main.py
```