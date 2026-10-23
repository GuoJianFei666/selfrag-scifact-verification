"""SciFact import and explicit citation-source grouping."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
import random
import tarfile
import unicodedata
import urllib.request

from .common import digest, file_hash, integer, parse_json, read_json, read_jsonl, string, utcnow, write_json, write_jsonl

RELATIONS = ("SUPPORT", "CONTRADICT", "NOINFO")
DATA_URL = "https://scifact.s3.us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    abstract: tuple[str, ...]

    @classmethod
    def parse(cls, row):
        doc_id = integer(row["doc_id"], "doc_id")
        title = string(row["title"], "title")
        if not isinstance(row["abstract"], list) or not row["abstract"]:
            raise ValueError("abstract must be a nonempty sentence array")
        sentences = tuple(string(s, "abstract sentence") for s in row["abstract"])
        return cls(doc_id, title, sentences)

    @property
    def content_hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Evidence:
    relation: str
    rationales: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    cited_doc_ids: tuple[int, ...]
    evidence: dict[int, Evidence]
    split: str
    labeled: bool

    @classmethod
    def parse(cls, row, corpus, split):
        cid = integer(row["id"], "claim_id")
        claim_text = string(row["claim"], "claim")
        cited = row.get("cited_doc_ids", []) if split == "test" else row["cited_doc_ids"]
        if not isinstance(cited, list) or any(type(d) is not int or d not in corpus for d in cited):
            raise ValueError(f"Invalid cited_doc_ids in claim {cid}")
        raw = row.get("evidence", {}) if split == "test" else row["evidence"]
        if not isinstance(raw, dict):
            raise ValueError("evidence must be an object")
        evidence = {}
        for key, entries in raw.items():
            if not key.isdigit() or str(int(key)) != key:
                raise ValueError("Evidence document IDs must be canonical decimal strings")
            doc_id = int(key)
            if doc_id not in corpus or doc_id not in cited or not isinstance(entries, list) or not entries:
                raise ValueError(f"Unknown or unreviewed evidence document: {key}")
            labels = {e["label"] for e in entries}
            if len(labels) != 1 or not labels <= {"SUPPORT", "CONTRADICT"}:
                raise ValueError("A claim-document pair must have one relation")
            rationales = []
            for entry in entries:
                ids = entry["sentences"]
                if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < len(corpus[doc_id].abstract) for i in ids) or len(ids) != len(set(ids)):
                    raise ValueError(f"Invalid rationale in claim {cid}, document {key}")
                rationale = tuple(ids)
                if rationale in rationales:
                    raise ValueError("Duplicate rationale set")
                rationales.append(rationale)
            evidence[doc_id] = Evidence(next(iter(labels)), tuple(rationales))
        if split == "test" and evidence:
            raise ValueError("Test evidence must not be interpreted as public gold")
        return cls(cid, claim_text, tuple(dict.fromkeys(cited)), evidence, split, split != "test")


class Dataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            for filename, expected in manifest["files"].items():
                if file_hash(self.directory / filename) != expected:
                    raise ValueError(f"Dataset file changed: {filename}")
        docs = [Document.parse(r) for r in read_jsonl(self.directory / "corpus.jsonl")]
        self.corpus = {d.doc_id: d for d in docs}
        if len(self.corpus) != len(docs):
            raise ValueError("Duplicate document IDs")
        self.claims = {}
        all_ids = set()
        for split in ("train", "dev", "test"):
            values = [Claim.parse(r, self.corpus, split) for r in read_jsonl(self.directory / f"claims_{split}.jsonl")]
            ids = [c.claim_id for c in values]
            if len(ids) != len(set(ids)) or all_ids & set(ids):
                raise ValueError("Duplicate claim IDs within or across official splits")
            all_ids.update(ids)
            self.claims[split] = {c.claim_id: c for c in values}
        self.corpus_hash = digest([asdict(d) for d in sorted(docs, key=lambda d: d.doc_id)])

    def gold_relation(self, claim, doc_id):
        if not claim.labeled or doc_id not in claim.cited_doc_ids:
            raise ValueError("Relation gold is only defined for reviewed cited documents")
        item = claim.evidence.get(doc_id)
        return item.relation if item else "NOINFO"


def download_archive(path, url=DATA_URL, expected=DATA_SHA256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "evidence-rag/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read(30_000_001)
        if len(body) > 30_000_000 or hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("SciFact archive size or SHA256 mismatch")
        path.write_bytes(body)
    if file_hash(path) != expected:
        raise ValueError("SciFact archive SHA256 mismatch")
    return path


def import_archive(archive_path, output, expected_counts=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Import output must be empty")
    required = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"}
    found = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name in required | {"claims_with_citances.jsonl"} and member.isfile():
                if name in found or member.size > 30_000_000:
                    raise ValueError("Duplicate or oversized archive member")
                found[name] = archive.extractfile(member).read()
    if not required <= found.keys():
        raise ValueError("Archive is missing required JSONL files")
    output.mkdir(parents=True, exist_ok=True)
    for name, body in found.items():
        output.joinpath(name).write_bytes(body)
    data = Dataset(output)
    counts = {"corpus": len(data.corpus), **{key: len(value) for key, value in data.claims.items()}}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Unexpected SciFact counts: {counts}")
    duplicates = []
    for split in ("train", "dev", "test"):
        for row in read_jsonl(output / f"claims_{split}.jsonl"):
            ids = row.get("cited_doc_ids", [])
            if len(ids) != len(set(ids)):
                duplicates.append({"claim_id": row["id"], "split": split, "raw_cited_doc_ids": ids,
                                   "internal_unique_cited_doc_ids": list(dict.fromkeys(ids))})
    manifest = {"created_at": utcnow(), "archive_sha256": file_hash(archive_path), "counts": counts,
                "duplicate_cited_document_audit": duplicates,
                "files": {name: file_hash(output / name) for name in sorted(found)}, "corpus_hash": data.corpus_hash}
    write_json(output / "manifest.json", manifest)
    return manifest


def normalize_claim(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def audit_sources(dataset, sources_path, audit_path, split_output=None, seed=42, ratios=(0.8, 0.1, 0.1)):
    matches = defaultdict(set)
    for row in read_jsonl(sources_path):
        paper = row["s2orc_id"]
        if not isinstance(paper, (str, int)) or isinstance(paper, bool):
            raise ValueError("Missing source paper identifier")
        citance = string(row["citance"], "citance")
        group = digest({"paper": str(paper), "citance": normalize_claim(citance)})[:24]
        for item in row["claims"]:
            variants = [item[k] for k in ("claim_text", "text") if k in item]
            if not variants or any(not isinstance(v, str) or not v.strip() for v in variants) or len({normalize_claim(v) for v in variants}) != 1:
                raise ValueError("Missing or conflicting source claim text fields")
            if type(item["is_negation"]) is not bool:
                raise ValueError("is_negation must be boolean")
            matches[normalize_claim(variants[0])].add(group)
    mapping, unresolved = {}, []
    for split in ("train", "dev", "test"):
        for claim in dataset.claims[split].values():
            groups = matches.get(normalize_claim(claim.text), set())
            if len(groups) == 1:
                mapping[str(claim.claim_id)] = next(iter(groups))
            else:
                unresolved.append({"claim_id": claim.claim_id, "split": split, "candidate_groups": sorted(groups)})
    train_groups = {mapping[str(c)] for c in dataset.claims["train"] if str(c) in mapping}
    dev_groups = {mapping[str(c)] for c in dataset.claims["dev"] if str(c) in mapping}
    same_text = defaultdict(list)
    for split, claims in dataset.claims.items():
        for claim in claims.values():
            same_text[normalize_claim(claim.text)].append({"claim_id": claim.claim_id, "split": split})
    crosssplit = [{"claim_text": text, "members": members} for text, members in sorted(same_text.items())
                  if len({m["split"] for m in members}) > 1]
    coverage = {split: {"total": len(claims),
                        "matched": sum(str(cid) in mapping for cid in claims),
                        "unresolved": sum(str(cid) not in mapping for cid in claims)}
                for split, claims in dataset.claims.items()}
    audit = {"source_file_sha256": file_hash(sources_path), "mapping": mapping, "unresolved": unresolved,
             "coverage_by_split": coverage, "cross_split_exact_claim_text": crosssplit,
             "grouping_policy": "source_paper_and_citance",
             "train_dev_shared_source_groups": sorted(train_groups & dev_groups)}
    write_json(audit_path, audit)
    train_unresolved = [r for r in unresolved if r["split"] == "train"]
    if train_unresolved:
        raise ValueError(f"Source alignment unresolved for {len(train_unresolved)} training claims; inspect {audit_path}")
    if split_output:
        if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
            raise ValueError("Split ratios must be three positive numbers summing to one")
        by_group = defaultdict(list)
        for claim in dataset.claims["train"].values():
            by_group[mapping[str(claim.claim_id)]].append(claim.claim_id)
        groups = sorted(by_group)
        random.Random(seed).shuffle(groups)
        sizes = [0, 0, 0]
        total = len(dataset.claims["train"])
        assignments = {}
        for group in groups:
            index = min(range(3), key=lambda i: sizes[i] / (ratios[i] * total))
            for cid in by_group[group]:
                assignments[str(cid)] = ("fit", "model_select", "calibration")[index]
            sizes[index] += len(by_group[group])
        classes = {name: Counter() for name in ("fit", "model_select", "calibration")}
        for cid, split in assignments.items():
            claim = dataset.claims["train"][int(cid)]
            for did in claim.cited_doc_ids:
                classes[split][dataset.gold_relation(claim, did)] += 1
        result = {"seed": seed, "ratios": list(ratios), "counts": sizes, "groups": mapping,
                  "assignments": assignments, "relation_counts": {k: dict(v) for k, v in classes.items()},
                  "dataset_manifest_path": str((dataset.directory / "manifest.json").resolve()),
                  "dataset_manifest_sha256": file_hash(dataset.directory / "manifest.json"),
                  "source_file_sha256": file_hash(sources_path)}
        write_json(split_output, result)
    return audit


def validate_split(manifest):
    dataset = Dataset(Path(manifest["dataset_manifest_path"]).parent)
    if file_hash(dataset.directory / "manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("Source split dataset changed")
    if set(manifest["assignments"]) != {str(cid) for cid in dataset.claims["train"]}:
        raise ValueError("Source split must cover all official training claims")
    grouped = defaultdict(set)
    for cid, partition in manifest["assignments"].items():
        group = manifest["groups"].get(cid)
        if partition not in ("fit", "model_select", "calibration") or not isinstance(group, str) or not group or group == "UNRESOLVED_SOURCE":
            raise ValueError("Unresolved source or invalid internal partition")
        grouped[group].add(partition)
    if any(len(partitions) != 1 for partitions in grouped.values()):
        raise ValueError("Source group crosses internal partitions")
    return dataset
