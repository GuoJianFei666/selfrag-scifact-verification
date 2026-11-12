# SciFact dev 检索对比

本表由结果 JSON 生成。Recall@k、MRR@10、nDCG@10 是合格主张上的宏平均，范围 0 到 1。没有实际跑过的实验写 `not yet run`，不补 0。

硬件：Intel(R) Xeon(R) Processor，4 核，内存 15.63 GiB，GPU none。

生成时间：2026-11-12T08:09:14+00:00

- contriever: `facebook/contriever` @ `2bd46a25019aeea091fd42d1f0fd4801675cf699`
- embedding: `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- reranker: `BAAI/bge-reranker-base` @ `2cfc18c9415c912f9d8155881c133215df768a70`

- Gold documents are the keys of claim.evidence (SUPPORT or CONTRADICT). Claims whose evidence map is empty stay in the latency sample and out of Recall, MRR and nDCG.
- Recall@k, MRR@10 and nDCG@10 are macro averages over eligible claims. Relevance is binary. nDCG uses log base 2.
- Latency is wall-clock seconds for one claim after a single untimed warmup query. It includes query encoding and, for the reranker, cross-encoder scoring. Index build time starts after the weights are loaded from the local cache and ends when the index has been persisted; download time is not included.
- Chunk hits collapse to documents by maximum score. Equal scores break toward the smaller doc id.
- Milvus uses a FLAT index, so the vector search is exact. The Contriever baseline is FAISS IndexFlatIP with unnormalized mean pooling.
- The semantic chunker keeps adjacent sentences together while their cosine similarity stays at or above the configured threshold and the chunk is under the sentence cap. That threshold is a fixed preset and was not tuned on dev.

| Experiment | Status | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Mean latency (s) | Index build (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Contriever / FAISS exact | measured | 0.6392 | 0.8249 | 0.8716 | 0.9173 | 0.9516 | 0.7495 | 0.7866 | 0.023 | 697.146 |
| bge-small-en-v1.5 / Milvus whole abstract | measured | 0.7416 | 0.8703 | 0.8997 | 0.9606 | 0.9739 | 0.8335 | 0.8611 | 0.234 | 262.651 |
| bge-small-en-v1.5 / Milvus + bge-reranker-base | measured | 0.7442 | 0.8805 | 0.9020 | 0.9605 | 0.9766 | 0.8399 | 0.8637 | 7.640 | reused |
| bge-small-en-v1.5 + query rewrite | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| bge-small-en-v1.5 + HyDE | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run |
| bge-small-en-v1.5 / sentence chunks | measured | 0.7695 | 0.8823 | 0.9226 | 0.9606 | 0.9819 | 0.8523 | 0.8780 | 1.600 | 433.065 |
| bge-small-en-v1.5 / sliding window (2, stride 1) | measured | 0.7735 | 0.8918 | 0.9309 | 0.9755 | 0.9915 | 0.8588 | 0.8854 | 1.455 | 573.232 |
| bge-small-en-v1.5 / adjacent semantic chunks | measured | 0.7469 | 0.8825 | 0.9146 | 0.9686 | 0.9883 | 0.8424 | 0.8702 | 0.682 | 667.449 |

- `contriever_faiss`: Frozen Contriever, attention-mask mean pooling, unnormalized inner product, exact FAISS.
- `bge_milvus_whole`: One vector per abstract: title, newline, then the abstract sentences joined by spaces.
- `bge_milvus_rerank`: First-stage pool is the top 50 whole-abstract BGE hits. Metrics use the reranked top 20.
- `bge_query_rewrite`: not yet run
- `bge_hyde`: not yet run
- `bge_milvus_sentence`: One vector per sentence. Each chunk is the title, a newline, and that sentence.
- `bge_milvus_sliding`: Window 2, stride 1. Each chunk is the title plus that sentence span.
- `bge_milvus_semantic`: Adjacent sentences stay together while cosine similarity is at least 0.5 and the chunk has at most 3 sentences.
