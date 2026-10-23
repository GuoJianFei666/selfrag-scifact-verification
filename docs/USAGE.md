# 使用与阅读说明

README与架构说明对应处理流程；数据契约规定文档绑定、句子编号与原句回查；评价说明定义证据视图和指标口径。

`tools/inspect_public_data.py`为此前公开的独立数据复核工具，具体参数以文件中的argparse定义和`--help`为准。该入口仅复核公开数据；核验应用与训练源码暂不公开。

若复用方法，数据从[AllenAI SciFact](https://github.com/allenai/scifact)获取；模型及原始方法从[Self-RAG](https://github.com/AkariAsai/self-rag)和相应模型发布页获取，并分别检查其环境、使用限制和许可。`data/`保存来源、哈希与既有统计，不包含模型权重或完整摘要原文。
