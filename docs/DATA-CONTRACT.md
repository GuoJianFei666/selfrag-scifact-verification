# 数据与输出契约

`POST /api/search`和`POST /api/verify`接收以下输入。`document_ids`可省略，省略时检索整个当前索引。

```json
{"claim":"An English scientific claim.","top_k":5,"document_ids":[13900610]}
```

核验报告包含`request_id`、`claim`、`document_scope`、`retrieval_version`、`model_version`和`results`。每篇文献的结果使用以下字段：

|字段|含义|
|---|---|
|`claim_id`、`doc_id`、`view_id`|系统绑定的主张、文档和视图标识|
|`relation_candidate`|SUPPORT、CONTRADICT、NOINFO；执行失败时为空|
|`view_sentence_ids`|模型选择的连续局部编号|
|`original_sentence_ids`、`evidence_sentences`|映射后的原始编号与系统回取原句|
|`status`|accepted、review_required、invalid_output、execution_failed|
|`confidence`、`label_scores`|关系候选概率和完整序列分数|
|`raw_output`、`generated_token_ids`|原始结构化输出及生成token|
|`source`|文档ID、语料哈希、文档哈希|
|`cost`、`elapsed_seconds`|实际前向计数、输入token和时间|

原始句子编号从0开始。同一主张的干预视图继承来源组和划分。NOINFO只说明当前摘要信息不足；结构合法不等于语义充分。无效输出和执行失败独立记录，保留在诊断分母中。

模型未加载时核验接口返回503，不生成默认关系。报告可从`GET /api/reports/{request_id}`导出。
