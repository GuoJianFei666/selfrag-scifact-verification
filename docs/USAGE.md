# 使用说明

仓库现已公开应用、数据处理、训练和评测源码。安装、导入、建索引与网页启动见[README](../README.md)，训练与冻结评测见[实验流程](EXPERIMENTS.md)，实际执行范围见[验证说明](VALIDATION.md)。

所有命令从仓库根目录运行。Windows解释器位于`.venv/Scripts/python.exe`，运行文件位于`var/`，该目录不入库。`Start.ps1`默认启动真实检索；加入`-WithModel`后读取`configs/model.yaml`。那份配置是 FP32，8GB 卡加载不了 7B。这张卡用 [8GB 实验](experiments-8gb.md) 里的 NF4 配置，例如 `.\Start.ps1 -WithModel -ModelConfig configs/B0-4bit.yaml`。

`tools/inspect_public_data.py`保留此前公开的静态复核入口，独立于新应用。`data/`保留来源、哈希与历史统计，不包含完整数据、缓存或模型权重。
