"""LlamaIndex index backed by Milvus Lite or a Milvus server."""
import hashlib
import math
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding

from .config import milvus_mode


class HashEmbedder(BaseEmbedding):
    """Deterministic bag-of-tokens vectors for tests. Not a retrieval model."""

    dimension: int

    def __init__(self, dimension=8, **kwargs):
        super().__init__(model_name="hash-embedder", dimension=dimension, **kwargs)

    def _vector(self, text):
        tokens = text.lower().split()
        if not tokens:
            raise ValueError("Embedder input must be nonempty")
        totals = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            for index in range(self.dimension):
                totals[index] += digest[index] / 255.0
        norm = math.sqrt(sum(value * value for value in totals)) or 1.0
        return [value / norm for value in totals]

    def _get_text_embedding(self, text):
        return self._vector(text)

    def _get_query_embedding(self, query):
        return self._vector(query)

    def _get_text_embeddings(self, texts):
        return [self._vector(text) for text in texts]

    async def _aget_text_embedding(self, text):
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)


class MilvusChunkIndex:
    def __init__(self, vector_store, embedder, collection, uri, mode):
        self.vector_store = vector_store
        self.embedder = embedder
        self.collection = collection
        self.uri = uri
        self.mode = mode

    @classmethod
    def build(cls, uri, nodes, embedder, collection, overwrite=True, similarity="COSINE", show_progress=False, token=""):
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.schema import TextNode

        if not nodes or any(not row["text"].strip() for row in nodes):
            raise ValueError("Milvus index requires nonempty nodes")
        mode = milvus_mode(str(uri))
        endpoint = str(uri)
        if mode == "lite":
            path = Path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite and path.exists():
                path.unlink()
            endpoint = str(path.resolve())
        dimension = getattr(embedder, "dimension", None)
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            dimension = len(embedder.get_text_embedding(nodes[0]["text"]))
        text_nodes = [
            TextNode(
                text=row["text"],
                id_=f"{row['doc_id']}-{row['chunk_index']}",
                metadata={"corpus_doc_id": int(row["doc_id"]), "chunk_index": int(row["chunk_index"])},
            )
            for row in nodes
        ]
        vector_store = _vector_store(endpoint, dimension, collection, overwrite, similarity, token)
        VectorStoreIndex(
            text_nodes,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embedder,
            show_progress=show_progress,
        )
        index = cls(vector_store, embedder, collection, endpoint, mode)
        index.count = len(text_nodes)
        return index

    @classmethod
    def open(cls, uri, embedder, collection, similarity="COSINE", token=""):
        mode = milvus_mode(str(uri))
        endpoint = str(Path(uri).resolve()) if mode == "lite" else str(uri)
        vector_store = _vector_store(endpoint, embedder.dimension, collection, False, similarity, token)
        index = cls(vector_store, embedder, collection, endpoint, mode)
        stats = vector_store.client.get_collection_stats(collection)
        index.count = int(stats["row_count"])
        return index

    def close(self):
        client = getattr(self.vector_store, "client", None)
        if client is not None:
            client.close()

    def search(self, query, chunk_k, as_query=True):
        from llama_index.core.vector_stores.types import VectorStoreQuery

        if not isinstance(query, str) or not query.strip() or type(chunk_k) is not int or chunk_k < 1:
            raise ValueError("Search requires nonempty text and a positive chunk_k")
        if as_query:
            vector = self.embedder.get_query_embedding(query)
        else:
            vector = self.embedder.get_text_embedding(query)
        if getattr(self, "count", None):
            chunk_k = min(chunk_k, self.count)
        result = self.vector_store.query(VectorStoreQuery(
            query_embedding=[float(value) for value in vector],
            similarity_top_k=chunk_k,
        ))
        hits = []
        for node, score in zip(result.nodes, result.similarities):
            hits.append({
                "doc_id": int(node.metadata["corpus_doc_id"]),
                "chunk_index": int(node.metadata["chunk_index"]),
                "score": float(score),
            })
        hits.sort(key=lambda hit: (-hit["score"], hit["doc_id"], hit["chunk_index"]))
        return hits


def _vector_store(endpoint, dimension, collection, overwrite, similarity, token=""):
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from pymilvus import DataType
    return MilvusVectorStore(
        uri=endpoint,
        token=token or "",
        collection_name=collection,
        dim=dimension,
        overwrite=overwrite,
        similarity_metric=similarity,
        enable_sparse=False,
        use_async_client=False,
        index_config={"index_type": "FLAT"},
        scalar_field_names=["corpus_doc_id", "chunk_index"],
        scalar_field_types=[DataType.INT64, DataType.INT64],
    )
