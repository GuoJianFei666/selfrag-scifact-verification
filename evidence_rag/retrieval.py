"""Frozen Contriever mean pooling and exact FAISS inner-product search."""
from dataclasses import asdict
from pathlib import Path
import re

from .common import digest, file_hash, read_json, write_json


class ContrieverEncoder:
    def __init__(self, model_id, revision, device="cpu", max_length=512, normalize=False, cache_dir=None):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Retriever revision must be a fixed full commit SHA")
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.max_length, self.normalize = device, max_length, normalize
        self.spec = {"model_id": model_id, "revision": revision, "max_length": max_length,
                     "normalize": normalize, "pooling": "attention-mask-mean", "similarity": "inner_product"}

    def encode(self, texts, batch_size=16):
        import numpy as np
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Encoder input must contain nonempty texts")
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                if self.normalize:
                    pooled = self.torch.nn.functional.normalize(pooled, dim=-1)
                vectors.append(pooled.float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)


class ExactIndex:
    def __init__(self, directory, corpus, encoder):
        import faiss
        self.directory, self.corpus, self.encoder = Path(directory), corpus, encoder
        manifest = read_json(self.directory / "manifest.json")
        if manifest["encoder"] != encoder.spec:
            raise ValueError("Index and query encoder configurations differ")
        if file_hash(self.directory / "index.faiss") != manifest["index_sha256"]:
            raise ValueError("FAISS index hash mismatch")
        self.doc_ids = manifest["doc_ids"]
        selected = [asdict(corpus[i]) for i in self.doc_ids]
        if digest(selected) != manifest["corpus_hash"]:
            raise ValueError("Indexed corpus has changed")
        self.index = faiss.read_index(str(self.directory / "index.faiss"))
        if self.index.ntotal != len(self.doc_ids):
            raise ValueError("Index-to-document mapping is incomplete")
        self.manifest = manifest

    @classmethod
    def build(cls, directory, documents, encoder, batch_size=16):
        import faiss
        import numpy as np
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Index directory must be empty")
        docs = sorted(documents, key=lambda d: d.doc_id)
        if not docs or len({d.doc_id for d in docs}) != len(docs):
            raise ValueError("Index requires unique documents")
        texts = [d.title + "\n" + " ".join(d.abstract) for d in docs]
        vectors = encoder.encode(texts, batch_size)
        if vectors.ndim != 2 or vectors.shape[0] != len(docs) or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned an invalid matrix")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(directory / "index.faiss"))
        manifest = {"encoder": encoder.spec, "corpus_hash": digest([asdict(d) for d in docs]),
                    "doc_ids": [d.doc_id for d in docs], "dimension": vectors.shape[1],
                    "index_sha256": file_hash(directory / "index.faiss")}
        write_json(directory / "manifest.json", manifest)
        return manifest

    def search(self, query, top_k=5, document_ids=None):
        if not isinstance(query, str) or not query.strip() or type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("Nonempty claim and top_k between 1 and 100 required")
        scope = set(self.doc_ids) if document_ids is None else set(document_ids)
        if not scope <= set(self.doc_ids):
            raise ValueError("Requested document scope is not in this index")
        return self._search_vectors(self.encoder.encode([query]), top_k, scope)[0]

    def search_many(self, queries, top_k=5, batch_size=16):
        if not queries or any(not isinstance(q, str) or not q.strip() for q in queries) or type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("Nonempty queries and top_k between 1 and 100 required")
        return self._search_vectors(self.encoder.encode(queries, batch_size), top_k, set(self.doc_ids))

    def _search_vectors(self, vectors, top_k, scope):
        scores, indices = self.index.search(vectors, len(self.doc_ids))
        results = []
        for row_scores, row_ids in zip(scores, indices):
            pairs = [(self.doc_ids[int(i)], float(s)) for i, s in zip(row_ids, row_scores) if i >= 0 and self.doc_ids[int(i)] in scope]
            pairs.sort(key=lambda x: (-x[1], x[0]))
            results.append([{"doc_id": did, "rank": rank + 1, "score": score} for rank, (did, score) in enumerate(pairs[:top_k])])
        return results
