"""BGE bi-encoder and cross-encoder loaders. Query text gets the retrieval prefix; passages do not."""
from llama_index.core.base.embeddings.base import BaseEmbedding


class BgeEmbedder(BaseEmbedding):
    dimension: int
    query_prefix: str = ""
    show_progress: bool = False

    def __init__(self, model, query_prefix, dimension, batch_size, show_progress=False):
        super().__init__(
            model_name="bge",
            dimension=dimension,
            query_prefix=query_prefix,
            embed_batch_size=batch_size,
            show_progress=show_progress,
        )
        self._model = model

    def _embed(self, texts, prefix, progress):
        payload = [prefix + text for text in texts]
        vectors = self._model.encode(
            payload,
            batch_size=self.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress and len(payload) > 1,
        )
        return [vector.tolist() for vector in vectors]

    def _get_text_embedding(self, text):
        return self._embed([text], "", False)[0]

    def _get_query_embedding(self, query):
        return self._embed([query], self.query_prefix, False)[0]

    def _get_text_embeddings(self, texts):
        return self._embed(list(texts), "", self.show_progress)

    async def _aget_text_embedding(self, text):
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)


def ensure_snapshot(model_id, revision, cache, ignore_patterns=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=revision, cache_dir=cache, ignore_patterns=ignore_patterns)


def load_bge(spec, cache, show_progress=False):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        spec["model_id"],
        revision=spec["revision"],
        cache_folder=cache,
        device="cpu",
        local_files_only=True,
    )
    model.prompts = {}
    model.default_prompt_name = None
    model.max_seq_length = int(spec["max_length"])
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension != int(spec["dimension"]):
        raise ValueError(f"Embedding dimension {dimension} does not match config {spec['dimension']}")
    return BgeEmbedder(model, spec["query_prefix"], dimension, int(spec["batch_size"]), show_progress)


def load_reranker(spec, cache):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(
        spec["model_id"],
        revision=spec["revision"],
        max_length=int(spec["max_length"]),
        device="cpu",
        cache_dir=cache,
        local_files_only=True,
    )
