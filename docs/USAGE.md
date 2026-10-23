# 使用与阅读说明

本公开包用于查看方法、接口和数据来源。先读README与架构，再对照数据契约理解文档/句号回查，最后读评价口径和已有公开数据复核记录。

`tools/inspect_public_data.py`为此前公开的独立数据复核工具，具体参数以文件中的argparse定义和`--help`为准；它不是Self-RAG训练或核验服务入口。当前包不提供“一条命令运行完整核验服务”的承诺。

若复用方法，数据从[AllenAI SciFact](https://github.com/allenai/scifact)获取；模型及原始方法从[Self-RAG](https://github.com/AkariAsai/self-rag)和相应模型发布页获取，并分别检查其环境、使用限制和许可。`data/`保存来源、哈希与既有统计，不包含模型权重或完整摘要原文。
