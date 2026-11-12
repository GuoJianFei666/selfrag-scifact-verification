"""Document chunkers. Every chunk keeps the source doc id and sentence span."""
import math


def whole_abstract_chunks(document):
    text = document.title + "\n" + " ".join(document.abstract)
    return [_chunk(document, 0, 0, len(document.abstract), text)]


def sentence_chunks(document):
    return [
        _chunk(document, index, index, index + 1, document.title + "\n" + sentence)
        for index, sentence in enumerate(document.abstract)
    ]


def sliding_chunks(document, window, stride):
    if type(window) is not int or type(stride) is not int or window < 1 or stride < 1:
        raise ValueError("Sliding window and stride must be positive integers")
    sentences = document.abstract
    count = len(sentences)
    if count <= window:
        spans = [(0, count)]
    else:
        spans = []
        start = 0
        while start < count:
            end = min(start + window, count)
            spans.append((start, end))
            if end == count:
                break
            start += stride
    return [
        _chunk(document, index, start, end, document.title + "\n" + " ".join(sentences[start:end]))
        for index, (start, end) in enumerate(spans)
    ]


def semantic_spans(vectors, threshold, max_sentences):
    """Greedy adjacent split. A boundary falls when cosine similarity drops or the cap is hit."""
    if not vectors or type(max_sentences) is not int or max_sentences < 1:
        raise ValueError("Semantic chunking requires vectors and a positive sentence cap")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not -1 <= threshold <= 1:
        raise ValueError("Semantic threshold must be a cosine similarity between -1 and 1")
    spans = []
    start = 0
    for index in range(1, len(vectors)):
        similarity = _cosine(vectors[index - 1], vectors[index])
        if index - start >= max_sentences or similarity < threshold:
            spans.append((start, index))
            start = index
    spans.append((start, len(vectors)))
    return spans


def semantic_chunks(document, vectors, threshold, max_sentences):
    if len(vectors) != len(document.abstract):
        raise ValueError("Semantic vectors must match the abstract sentence count")
    sentences = document.abstract
    return [
        _chunk(document, index, start, end, document.title + "\n" + " ".join(sentences[start:end]))
        for index, (start, end) in enumerate(semantic_spans(vectors, threshold, max_sentences))
    ]


def _chunk(document, index, start, end, text):
    if not text.strip():
        raise ValueError(f"Empty chunk for document {document.doc_id}")
    return {
        "doc_id": document.doc_id,
        "chunk_index": index,
        "sentence_start": start,
        "sentence_end": end,
        "text": text,
    }


def _cosine(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Semantic vectors must be nonempty and aligned")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Semantic chunking received a zero vector")
    return dot / (left_norm * right_norm)
