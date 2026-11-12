"""Run the SciFact retrieval comparison and write only measured numbers."""
import gc
import logging
import os
from pathlib import Path
import time

from ..common import utcnow, write_json
from ..data import Dataset
from .aggregate import aggregate_documents
from .chunking import semantic_chunks, sentence_chunks, sliding_chunks, whole_abstract_chunks
from .config import load_config, milvus_mode
from .llm import gated_transform, hyde_passage, llm_settings, rewrite_query
from .metrics import score_claims
from .models import ensure_snapshot, load_bge, load_reranker
from .plan import EXPERIMENTS, comparison_rows
from .report import render_markdown
from .rerank import order_by_scores

log = logging.getLogger("evidence_rag.retrieval_engineering")

BGE_IGNORE = ["onnx/*", "pytorch_model.bin", "*.ot"]
CONTRIEVER_IGNORE = ["onnx/*", "*.ot"]
METRIC_NOTES = [
    "Gold documents are the keys of claim.evidence (SUPPORT or CONTRADICT). Claims whose evidence map is empty stay in the latency sample and out of Recall, MRR and nDCG.",
    "Recall@k, MRR@10 and nDCG@10 are macro averages over eligible claims. Relevance is binary. nDCG uses log base 2.",
    "Latency is wall-clock seconds for one claim after a single untimed warmup query. It includes query encoding and, for the reranker, cross-encoder scoring. Index build time starts after the weights are loaded from the local cache and ends when the index has been persisted; download time is not included.",
    "Chunk hits collapse to documents by maximum score. Equal scores break toward the smaller doc id.",
    "Milvus uses a FLAT index, so the vector search is exact. The Contriever baseline is FAISS IndexFlatIP with unnormalized mean pooling.",
    "The semantic chunker keeps adjacent sentences together while their cosine similarity stays at or above the configured threshold and the chunk is under the sentence cap. That threshold is a fixed preset and was not tuned on dev.",
]


def configure_threads():
    cores = str(os.cpu_count() or 1)
    os.environ.setdefault("OMP_NUM_THREADS", cores)
    os.environ.setdefault("MKL_NUM_THREADS", cores)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def hardware_info():
    cpu = "unknown"
    ram_kib = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = os.uname().machine
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                ram_kib = int(line.split()[1])
                break
    except OSError:
        ram_kib = None
    gpu = "none"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = "unavailable"
    return {
        "platform": os.uname().sysname + " " + os.uname().release,
        "cpu_model": cpu,
        "cpu_cores": os.cpu_count(),
        "ram_gib": None if ram_kib is None else round(ram_kib / 1024 / 1024, 2),
        "gpu": gpu,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_threads": _torch_threads(),
    }


def package_versions():
    import importlib.metadata as metadata
    names = (
        "torch", "transformers", "faiss-cpu", "llama-index-core",
        "llama-index-vector-stores-milvus", "pymilvus", "milvus-lite", "sentence-transformers",
    )
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def ensure_work(path, resume):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Work directory is not empty: {path}. Pass --resume or choose another --work.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_nodes(documents, kind, cfg, embedder=None):
    chunking = cfg["chunking"]
    nodes = []
    if kind == "whole_abstract":
        for document in documents:
            nodes.extend(whole_abstract_chunks(document))
    elif kind == "sentence":
        for document in documents:
            nodes.extend(sentence_chunks(document))
    elif kind == "sliding":
        window, stride = int(chunking["sliding_window"]), int(chunking["sliding_stride"])
        for document in documents:
            nodes.extend(sliding_chunks(document, window, stride))
    elif kind == "semantic":
        if embedder is None:
            raise ValueError("Semantic chunking requires an embedding model")
        nodes.extend(_semantic_nodes(documents, embedder, float(chunking["semantic_threshold"]), int(chunking["semantic_max_sentences"])))
    else:
        raise ValueError(f"Unknown chunking: {kind}")
    return nodes


def select_claims(data, split):
    if split == "test":
        raise ValueError("SciFact test has no public evidence gold")
    if split not in data.claims:
        raise ValueError(f"Unknown split: {split}")
    claims = data.claims[split]
    if not claims:
        raise ValueError("Claim set is empty")
    return claims


def passage_lookup(corpus):
    texts = {doc_id: whole_abstract_chunks(document)[0]["text"] for doc_id, document in corpus.items()}

    def lookup(doc_id):
        try:
            return texts[doc_id]
        except KeyError as exc:
            raise KeyError(f"Missing passage for document {doc_id}") from exc

    return lookup


class Comparison:
    def __init__(self, data_dir, config_path, cache, work, split="dev", resume=False):
        configure_threads()
        self.cfg = load_config(config_path)
        self.data = Dataset(data_dir)
        self.claims = select_claims(self.data, split)
        self.split = split
        self.cache = str(cache)
        self.work = ensure_work(work, resume)
        self.documents = [self.data.corpus[doc_id] for doc_id in sorted(self.data.corpus)]
        self.depth = int(self.cfg["evaluation"]["ranking_depth"])
        self.ks = tuple(int(value) for value in self.cfg["evaluation"]["ks"])
        self.checkpoint_path = self.work / "checkpoint.json"
        self.done = _read_checkpoint(self.checkpoint_path) if resume else {}
        self._expect_counts()

    def baseline(self):
        return self._cached("contriever_faiss", self._baseline)

    def bge(self, kind="whole_abstract"):
        experiment_id = _chunk_experiment_id(kind)
        return self._cached(experiment_id, lambda: self._bge(kind, experiment_id))

    def rerank(self):
        return self._cached("bge_milvus_rerank", self._rerank)

    def chunking(self, kinds=("whole_abstract", "sentence", "sliding", "semantic")):
        return [self.bge(kind) for kind in kinds]

    def rewrite(self, env=None):
        env = os.environ if env is None else env
        settings = llm_settings(self.cfg["llm"], env)
        rows = []
        for method, experiment_id in (("rewrite", "bge_query_rewrite"), ("hyde", "bge_hyde")):
            label = _spec(experiment_id)["label"]
            rows.append(gated_transform(settings, experiment_id, label, lambda method=method: self._transform(method, settings)))
        return rows

    def compare(self):
        """Non-LLM suite. Query rewrite and HyDE are recorded as not yet run."""
        try:
            self.baseline()
            self.bge("whole_abstract")
            self.rerank()
            self.bge("sentence")
            self.bge("sliding")
            self.bge("semantic")
        finally:
            self._drop_embedder()
        rows = []
        for spec in comparison_rows(include_llm=False):
            if spec["status"] == "not_run":
                rows.append({
                    "id": spec["id"], "label": spec["label"], "status": "not_run", "reason": "not yet run",
                    "group": spec["group"], "chunking": spec["chunking"], "requires_llm": True,
                })
            else:
                row = self.done[spec["id"]]
                if row.get("status") != "measured":
                    raise RuntimeError(f"Missing measured result for {spec['id']}")
                rows.append(row)
        return self.report(rows)

    def report(self, rows):
        claim_count, eligible, gold = _gold_counts(self.claims)
        return {
            "task": "scifact-document-retrieval",
            "generated_at": utcnow(),
            "split": self.split,
            "hardware": hardware_info(),
            "packages": package_versions(),
            "dataset": {
                "name": "scifact",
                "split": self.split,
                "corpus_documents": len(self.documents),
                "corpus_hash": self.data.corpus_hash,
                "claims": claim_count,
                "eligible_claims": eligible,
                "gold_documents": gold,
                "gold_definition": "documents present in claim.evidence (SUPPORT or CONTRADICT)",
            },
            "models": {
                "contriever": {"model_id": self.cfg["baseline"]["model_id"], "revision": self.cfg["baseline"]["revision"],
                               "pooling": "attention-mask-mean", "normalize": bool(self.cfg["baseline"]["normalize"])},
                "embedding": {"model_id": self.cfg["embedding"]["model_id"], "revision": self.cfg["embedding"]["revision"],
                              "pooling": self.cfg["embedding"]["pooling"], "normalize": True,
                              "query_prefix": self.cfg["embedding"]["query_prefix"], "dimension": self.cfg["embedding"]["dimension"]},
                "reranker": {"model_id": self.cfg["reranker"]["model_id"], "revision": self.cfg["reranker"]["revision"],
                             "max_length": self.cfg["reranker"]["max_length"], "candidates": self.cfg["reranker"]["candidates"]},
            },
            "metric_notes": METRIC_NOTES,
            "experiments": rows,
        }

    def _baseline(self):
        from ..retrieval import ContrieverEncoder, ExactIndex
        spec = self.cfg["baseline"]
        log.info("downloading Contriever if needed")
        ensure_snapshot(spec["model_id"], spec["revision"], self.cache, CONTRIEVER_IGNORE)
        started = time.perf_counter()
        encoder = ContrieverEncoder(
            spec["model_id"], spec["revision"], device="cpu", max_length=int(spec["max_length"]),
            normalize=bool(spec["normalize"]), cache_dir=self.cache,
        )
        load_seconds = time.perf_counter() - started
        directory = self.work / "contriever"
        if directory.exists():
            import shutil
            shutil.rmtree(directory)
        log.info("building Contriever FAISS index for %s documents", len(self.documents))
        started = time.perf_counter()
        ExactIndex.build(directory, self.documents, encoder, batch_size=int(spec["batch_size"]))
        build_seconds = time.perf_counter() - started
        index = ExactIndex(directory, self.data.corpus, encoder)
        depth = self.depth

        def search(text):
            return index.search(text, top_k=depth)

        metrics = self._score(search)
        del index, encoder
        _release()
        return self._row("contriever_faiss", metrics, {
            "framework": "transformers+faiss",
            "vector_store": "faiss.IndexFlatIP",
            "embedding_model": spec["model_id"],
            "embedding_revision": spec["revision"],
            "reranker": None,
            "query_transform": None,
            "candidate_k": depth,
            "index_chunks": len(self.documents),
            "model_load_seconds": load_seconds,
            "index_build_seconds": build_seconds,
            "milvus_mode": None,
            "notes": "Frozen Contriever, attention-mask mean pooling, unnormalized inner product, exact FAISS.",
        })

    def _bge(self, kind, experiment_id):
        embedder, load_seconds = self._embedder()
        if getattr(self, "_bge_load_reported", False):
            load_seconds = None
        else:
            self._bge_load_reported = True
        index, build_seconds, chunks = self._milvus(kind, embedder)
        depth = self.depth
        chunk_k = self._chunk_k(kind, depth)

        def search(text):
            return _documents(index, text, chunk_k, depth, as_query=True)

        try:
            metrics = self._score(search)
        finally:
            index.close()
        return self._row(experiment_id, metrics, {
            "framework": "llamaindex+milvus",
            "vector_store": "milvus-" + milvus_mode(str(self.cfg["milvus"]["uri"])),
            "embedding_model": self.cfg["embedding"]["model_id"],
            "embedding_revision": self.cfg["embedding"]["revision"],
            "reranker": None,
            "query_transform": None,
            "candidate_k": depth,
            "chunk_search_k": chunk_k,
            "index_chunks": chunks,
            "model_load_seconds": load_seconds,
            "index_build_seconds": build_seconds,
            "milvus_mode": milvus_mode(str(self.cfg["milvus"]["uri"])),
            "similarity": self.cfg["milvus"]["similarity"],
            "index_type": "FLAT",
            "chunking": kind,
            "notes": _chunk_note(kind, self.cfg),
        })

    def _rerank(self):
        embedder, _ = self._embedder()
        if self._index_ready("whole_abstract"):
            index = self._open_milvus("whole_abstract", embedder)
            build_seconds = None
            reused = "bge_milvus_whole"
        else:
            index, build_seconds, _ = self._milvus("whole_abstract", embedder)
            reused = None
        spec = self.cfg["reranker"]
        log.info("downloading bge-reranker if needed")
        ensure_snapshot(spec["model_id"], spec["revision"], self.cache, BGE_IGNORE)
        started = time.perf_counter()
        cross = load_reranker(spec, self.cache)
        load_seconds = time.perf_counter() - started
        candidates = int(spec["candidates"])
        depth = self.depth
        lookup = passage_lookup(self.data.corpus)
        batch_size = int(spec["batch_size"])

        def search(text):
            pooled = _documents(index, text, candidates, candidates, as_query=True)
            doc_ids = [row["doc_id"] for row in pooled]
            pairs = [(text, lookup(doc_id)) for doc_id in doc_ids]
            scores = cross.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            return order_by_scores(doc_ids, _as_floats(scores), depth)

        try:
            metrics = self._score(search)
        finally:
            index.close()
            del cross
            _release()
        return self._row("bge_milvus_rerank", metrics, {
            "framework": "llamaindex+milvus+cross-encoder",
            "vector_store": "milvus-" + milvus_mode(str(self.cfg["milvus"]["uri"])),
            "embedding_model": self.cfg["embedding"]["model_id"],
            "embedding_revision": self.cfg["embedding"]["revision"],
            "reranker": spec["model_id"],
            "reranker_revision": spec["revision"],
            "query_transform": None,
            "candidate_k": candidates,
            "index_chunks": len(self.documents),
            "model_load_seconds": load_seconds,
            "index_build_seconds": build_seconds,
            "index_reused_from": reused,
            "milvus_mode": milvus_mode(str(self.cfg["milvus"]["uri"])),
            "notes": f"First-stage pool is the top {candidates} whole-abstract BGE hits. Metrics use the reranked top {depth}.",
        })

    def _transform(self, method, settings):
        from .llm import chat_complete
        embedder, _ = self._embedder()
        index = self._open_milvus("whole_abstract", embedder) if self._index_ready("whole_abstract") else None
        if index is None:
            index, _, _ = self._milvus("whole_abstract", embedder)
        depth = self.depth
        chunk_k = depth

        def complete(prompt):
            return chat_complete(settings["base_url"], settings["api_key"], settings["model"], prompt)

        def search(text):
            if method == "rewrite":
                transformed = rewrite_query(text, complete)
                return _documents(index, transformed, chunk_k, depth, as_query=True)
            if method == "hyde":
                transformed = hyde_passage(text, complete)
                return _documents(index, transformed, chunk_k, depth, as_query=False)
            raise ValueError(method)

        try:
            metrics = self._score(search)
        finally:
            index.close()
        experiment_id = "bge_query_rewrite" if method == "rewrite" else "bge_hyde"
        return self._row(experiment_id, metrics, {
            "framework": "llamaindex+milvus+openai-compatible",
            "query_transform": method,
            "embedding_model": self.cfg["embedding"]["model_id"],
            "embedding_revision": self.cfg["embedding"]["revision"],
            "llm_model": settings["model"],
            "llm_base_url": settings["base_url"],
            "index_build_seconds": None,
            "index_reused_from": "bge_milvus_whole",
            "notes": "HyDE embeds the hypothetical abstract as a passage, without the query prefix." if method == "hyde"
                     else "The rewritten claim is embedded with the BGE query prefix.",
        })

    def _score(self, search):
        metrics = score_claims(
            self.claims, search, self.depth, ks=self.ks,
            mrr_k=int(self.cfg["evaluation"]["mrr_k"]), ndcg_k=int(self.cfg["evaluation"]["ndcg_k"]), warmup=True,
        )
        _, eligible, gold = _gold_counts(self.claims)
        if metrics["eligible_claims"] != eligible or metrics["gold_documents"] != gold or metrics["queries_timed"] != len(self.claims):
            raise RuntimeError("Scored claim counts do not match the split")
        log.info(
            "recall@5=%.4f mrr@10=%.4f ndcg@10=%.4f latency_mean=%.3fs",
            metrics["recall"]["5"], metrics["mrr_at_10"], metrics["ndcg_at_10"], metrics["latency_seconds"]["mean"],
        )
        return metrics

    def _embedder(self):
        state = getattr(self, "_embedder_state", None)
        if state is not None:
            return state
        spec = self.cfg["embedding"]
        log.info("downloading %s if needed", spec["model_id"])
        ensure_snapshot(spec["model_id"], spec["revision"], self.cache, BGE_IGNORE)
        started = time.perf_counter()
        embedder = load_bge(spec, self.cache, show_progress=False)
        self._embedder_state = (embedder, time.perf_counter() - started)
        return self._embedder_state

    def _drop_embedder(self):
        self._embedder_state = None
        _release()

    def _milvus(self, kind, embedder):
        from .stores import MilvusChunkIndex
        uri = self._store_uri(kind)
        collection = _collection(kind)
        log.info("building %s (%s)", collection, kind)
        started = time.perf_counter()
        nodes = build_nodes(self.documents, kind, self.cfg, embedder)
        index = MilvusChunkIndex.build(
            uri, nodes, embedder, collection, overwrite=True,
            similarity=self.cfg["milvus"]["similarity"], show_progress=True,
            token=self.cfg["milvus"].get("token") or "",
        )
        return index, time.perf_counter() - started, len(nodes)

    def _open_milvus(self, kind, embedder):
        from .stores import MilvusChunkIndex
        return MilvusChunkIndex.open(
            self._store_uri(kind), embedder, _collection(kind),
            similarity=self.cfg["milvus"]["similarity"],
            token=self.cfg["milvus"].get("token") or "",
        )

    def _store_uri(self, kind):
        configured = str(self.cfg["milvus"]["uri"])
        if milvus_mode(configured) == "server":
            return configured
        return str(self._db_path(kind))

    def _index_ready(self, kind):
        configured = str(self.cfg["milvus"]["uri"])
        if milvus_mode(configured) == "lite":
            return self._db_path(kind).exists()
        from pymilvus import MilvusClient
        client = MilvusClient(uri=configured, token=self.cfg["milvus"].get("token") or "")
        try:
            return _collection(kind) in set(client.list_collections())
        finally:
            client.close()

    def _db_path(self, kind):
        directory = Path(self.cfg["milvus"]["uri"]).parent
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{_collection(kind)}.db"

    def _chunk_k(self, kind, depth):
        if kind == "whole_abstract":
            return depth
        return int(self.cfg["chunking"]["chunk_search_k"])

    def _row(self, experiment_id, metrics, extra):
        spec = _spec(experiment_id)
        row = {
            "id": experiment_id,
            "label": spec["label"],
            "status": "measured",
            "group": spec["group"],
            "chunking": extra.get("chunking", spec["chunking"]),
            "requires_llm": spec["requires_llm"],
            "recall": metrics["recall"],
            "mrr_at_10": metrics["mrr_at_10"],
            "ndcg_at_10": metrics["ndcg_at_10"],
            "latency_seconds": metrics["latency_seconds"],
            "eligible_claims": metrics["eligible_claims"],
            "gold_documents": metrics["gold_documents"],
            "queries_timed": metrics["queries_timed"],
            "warmup_queries_excluded": 1,
        }
        row.update(extra)
        return row

    def _cached(self, experiment_id, builder):
        existing = self.done.get(experiment_id)
        if existing and existing.get("status") == "measured":
            log.info("reusing checkpoint %s", experiment_id)
            return existing
        row = builder()
        self.done[experiment_id] = row
        _write_checkpoint(self.checkpoint_path, list(self.done.values()))
        log.info("finished %s", experiment_id)
        return row

    def _expect_counts(self):
        claim_count, eligible, gold = _gold_counts(self.claims)
        if eligible == 0 or gold == 0:
            raise ValueError("Split has no gold evidence documents")
        log.info("%s claims=%s eligible=%s gold_documents=%s corpus=%s", self.split, claim_count, eligible, gold, len(self.documents))


def publish(report, output, markdown=None):
    _atomic_json(output, report)
    if markdown:
        path = Path(markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(render_markdown(report), encoding="utf-8")
        os.replace(temporary, path)


def _documents(index, text, chunk_k, depth, as_query):
    hits = index.search(text, min(chunk_k, index.count), as_query=as_query)
    return aggregate_documents(hits, depth)


def _semantic_nodes(documents, embedder, threshold, max_sentences):
    grouped = []
    cursor = 0
    sentences = []
    for document in documents:
        sentences.extend(document.abstract)
        grouped.append((document, cursor, cursor + len(document.abstract)))
        cursor += len(document.abstract)
    vectors = embedder.get_text_embedding_batch(sentences, show_progress=True)
    if len(vectors) != len(sentences):
        raise RuntimeError("Semantic embedder returned the wrong number of vectors")
    nodes = []
    for document, start, end in grouped:
        nodes.extend(semantic_chunks(document, vectors[start:end], threshold, max_sentences))
    return nodes


def _gold_counts(claims):
    eligible = [claim for claim in claims.values() if claim.evidence]
    return len(claims), len(eligible), sum(len(claim.evidence) for claim in eligible)


def _chunk_experiment_id(kind):
    return {
        "whole_abstract": "bge_milvus_whole",
        "sentence": "bge_milvus_sentence",
        "sliding": "bge_milvus_sliding",
        "semantic": "bge_milvus_semantic",
    }[kind]


def _collection(kind):
    return "scifact_" + kind


def _spec(experiment_id):
    for spec in EXPERIMENTS:
        if spec["id"] == experiment_id:
            return spec
    raise KeyError(experiment_id)


def _chunk_note(kind, cfg):
    chunking = cfg["chunking"]
    if kind == "whole_abstract":
        return "One vector per abstract: title, newline, then the abstract sentences joined by spaces."
    if kind == "sentence":
        return "One vector per sentence. Each chunk is the title, a newline, and that sentence."
    if kind == "sliding":
        return f"Window {chunking['sliding_window']}, stride {chunking['sliding_stride']}. Each chunk is the title plus that sentence span."
    return (
        f"Adjacent sentences stay together while cosine similarity is at least {chunking['semantic_threshold']} "
        f"and the chunk has at most {chunking['semantic_max_sentences']} sentences."
    )


def _read_checkpoint(path):
    if not path.exists():
        return {}
    from ..common import read_json
    payload = read_json(path)
    return {row["id"]: row for row in payload.get("experiments", []) if row.get("status") == "measured"}


def _write_checkpoint(path, rows):
    _atomic_json(path, {"experiments": rows})


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def _as_floats(scores):
    values = []
    for score in scores:
        if hasattr(score, "reshape"):
            score = score.reshape(-1)[0]
        values.append(float(score))
    return values


def _release():
    gc.collect()


def _torch_threads():
    try:
        import torch
        return torch.get_num_threads()
    except Exception:
        return None
