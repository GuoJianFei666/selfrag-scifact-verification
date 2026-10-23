# 上游版本与继承范围

|来源|固定版本|本实现使用的部分|
|---|---|---|
|[Self-RAG](https://github.com/AkariAsai/self-rag/tree/1fcdc420e48f50a7d7ab1ece5494221b93252e99)|1fcdc420e48f50a7d7ab1ece5494221b93252e99|任务模板和标记的参考；不执行原仓库训练或推理入口|
|[SelfRAG Llama2 7B](https://huggingface.co/selfrag/selfrag_llama2_7b/tree/190261383b0779ff66d2f95a73c7ad267d94b820)|190261383b0779ff66d2f95a73c7ad267d94b820|发布模型及原词表；实际 tokenizer 已核对，7B 权重未加载|
|[Contriever](https://huggingface.co/facebook/contriever/tree/2bd46a25019aeea091fd42d1f0fd4801675cf699)|2bd46a25019aeea091fd42d1f0fd4801675cf699|实际加载编码器，固定 mean pooling 与内积|
|[SciFact](https://github.com/allenai/scifact/tree/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e)|68b98a56d93e0f9da0d2aab4e6c3294699a0f72e|数据结构、关系枚举、官方指标定义|

应用、数据校验、视图构造、训练循环、候选关系评分、结果绑定及网页均为本轮独立实现。官方评测公式按固定上游实现重新编写，并使用手工计数算例校验；未运行旧项目代码。该应用不是完整上游自适应检索流程，也未重现 Self-RAG 论文的全部实验。

SciFact 数据来自 [官方原始包](https://scifact.s3.us-west-2.amazonaws.com/release/latest/data.tar.gz)，SHA256 为 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`。来源主张文件来自 [claims_with_citances.jsonl](https://scifact.s3.us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl)，SHA256 为 `a12af8640b5079dc6ceccbf1a31e755b28158643ddd1928f91b92caa0c388704`。文件结构见 [来源说明](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/doc/claims-with-citances.md)。

使用数据时保留 SciFact 的 [分项许可](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md)。Self-RAG 代码、Contriever 和 Llama2 衍生模型有各自许可；下载或分发模型前请阅读固定发布页中的许可及基础模型要求。新增代码的 MIT 不替代数据或权重许可。源码交付包不含上游模型权重、完整数据或旧项目源码。

API 与训练实现依据 [Transformers 4.48.3](https://huggingface.co/docs/transformers/v4.48.3/en/index)、[PEFT 0.14.0 LoRA](https://huggingface.co/docs/peft/v0.14.0/package_reference/lora)、[PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)及 [Contriever 官方实现](https://github.com/facebookresearch/contriever)。
