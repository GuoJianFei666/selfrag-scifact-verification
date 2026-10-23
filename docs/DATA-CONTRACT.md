# 数据与输出契约

接口字段及其语义如下。JSON中的文字为字段说明。

```json
{
  "claim_id": "主张原始编号",
  "document_id": "绑定的摘要编号",
  "relation": "SUPPORT | CONTRADICT | NOINFO",
  "view_sentence_ids": [],
  "original_sentence_ids": [],
  "evidence_sentences": [],
  "status": "accepted | review_required | invalid_output | execution_failed",
  "versions": {"corpus": "语料版本", "model": "模型版本"}
}
```

1. 同一主张的改写、干预视图沿用父主张分组与数据划分。
2. 文档和原始句子编号贯穿检索、选句、回查和报告，不能仅存生成式解释。
3. NOINFO只说明当前主张—摘要信息不足，不能解释为该主张为假或全库无依据。
4. 输出合法与证据充分是两个判断；句子编号存在不等于语义支持。
5. 无效输出和执行失败独立记录，并保留在评价分母中。
