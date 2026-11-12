"""Named experiments. LLM rows stay not-run unless a command explicitly enables them."""


EXPERIMENTS = (
    {"id": "contriever_faiss", "label": "Contriever / FAISS exact", "group": "baseline",
     "chunking": "whole_abstract", "requires_llm": False},
    {"id": "bge_milvus_whole", "label": "bge-small-en-v1.5 / Milvus whole abstract", "group": "embedding",
     "chunking": "whole_abstract", "requires_llm": False},
    {"id": "bge_milvus_rerank", "label": "bge-small-en-v1.5 / Milvus + bge-reranker-base", "group": "rerank",
     "chunking": "whole_abstract", "requires_llm": False},
    {"id": "bge_query_rewrite", "label": "bge-small-en-v1.5 + query rewrite", "group": "rewrite",
     "chunking": "whole_abstract", "requires_llm": True},
    {"id": "bge_hyde", "label": "bge-small-en-v1.5 + HyDE", "group": "hyde",
     "chunking": "whole_abstract", "requires_llm": True},
    {"id": "bge_milvus_sentence", "label": "bge-small-en-v1.5 / sentence chunks", "group": "chunking",
     "chunking": "sentence", "requires_llm": False},
    {"id": "bge_milvus_sliding", "label": "bge-small-en-v1.5 / sliding window (2, stride 1)", "group": "chunking",
     "chunking": "sliding", "requires_llm": False},
    {"id": "bge_milvus_semantic", "label": "bge-small-en-v1.5 / adjacent semantic chunks", "group": "chunking",
     "chunking": "semantic", "requires_llm": False},
)


def comparison_rows(include_llm=False):
    rows = []
    for spec in EXPERIMENTS:
        row = dict(spec)
        if spec["requires_llm"] and not include_llm:
            row["status"] = "not_run"
            row["reason"] = "not yet run"
        else:
            row["status"] = "pending"
        rows.append(row)
    return rows
