import json
from pathlib import Path
import pytest

from evidence_rag.common import file_hash, write_json, write_jsonl
from evidence_rag.data import Dataset
from evidence_rag.views import full_view


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data"
    write_jsonl(directory / "corpus.jsonl", [
        {"doc_id": 100, "title": "Synthetic intervention study", "abstract": [f"Synthetic sentence {i}." for i in range(12)]},
        {"doc_id": 200, "title": "Synthetic unrelated study", "abstract": [f"Control sentence {i}." for i in range(10)]}])
    write_jsonl(directory / "claims_train.jsonl", [
        {"id": 1, "claim": "Synthetic claim", "cited_doc_ids": [100, 200],
         "evidence": {"100": [{"label": "CONTRADICT", "sentences": [0, 1]}, {"label": "CONTRADICT", "sentences": [11]}]}}])
    write_jsonl(directory / "claims_dev.jsonl", [
        {"id": 2, "claim": "Other synthetic claim", "cited_doc_ids": [100],
         "evidence": {"100": [{"label": "SUPPORT", "sentences": [3]}]}}])
    write_jsonl(directory / "claims_test.jsonl", [{"id": 3, "claim": "Unlabeled synthetic claim"}])
    write_json(directory / "manifest.json", {"files": {p.name: file_hash(p) for p in directory.glob("*.jsonl")}})
    return Dataset(directory)


@pytest.fixture
def full(dataset):
    return full_view(dataset.claims["train"][1], dataset.corpus[100], "fit", "source-1", True)


@pytest.fixture
def split_manifest(dataset, tmp_path):
    path = tmp_path / "synthetic-splits.json"
    write_json(path, {"dataset_manifest_path": str(dataset.directory / "manifest.json"),
                     "dataset_manifest_sha256": file_hash(dataset.directory / "manifest.json"),
                     "assignments": {"1": "fit"}, "groups": {"1": "source-1"}})
    return path


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast
    # This byte alphabet tokenizer is a synthetic engineering fixture, not the SelfRAG tokenizer.
    inner = Tokenizer(BPE(unk_token="<unk>"))
    inner.pre_tokenizer = ByteLevel(add_prefix_space=False)
    inner.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=260, initial_alphabet=ByteLevel.alphabet(),
                         special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"])
    inner.train_from_iterator(["synthetic tokenizer engineering fixture"], trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=inner, bos_token="<bos>", eos_token="<eos>",
                                   unk_token="<unk>", pad_token="<pad>")


def prediction(view, relation=None, ids=None, status="accepted", confidence=0.8):
    relation = view.relation if relation is None else relation
    ids = ([] if relation == "NOINFO" else list(view.rationales[0])) if ids is None else ids
    return {"view_id": view.view_id, "claim_id": view.claim_id, "doc_id": view.doc_id,
            "status": status, "relation_candidate": relation, "original_sentence_ids": ids,
            "confidence": confidence}
