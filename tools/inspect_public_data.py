"""Read-only inspection for scifact; only this project dataset is included."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile
import time
import urllib.request
import zipfile


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def obtain(spec, cache, download):
    target = cache / spec["filename"]
    if not target.exists():
        if not download:
            raise FileNotFoundError(f"Missing {target}; use --download to retrieve public data")
        print(f"Downloading {spec['id']}", flush=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "project-public-data/1.0"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=20) as response:
            chunks = []
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                if time.monotonic() - started > 180:
                    raise TimeoutError(f"Total download time exceeds 180 seconds: {spec['id']}")
            body = b"".join(chunks)
        expected = spec.get("sha256")
        actual = hashlib.sha256(body).hexdigest()
        if expected and actual != expected:
            raise ValueError(f"SHA256 mismatch for {spec['id']}: {actual}")
        target.write_bytes(body)
    body = target.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if spec.get("sha256") and spec["sha256"] != actual:
        raise ValueError(f"Cached SHA256 mismatch: {spec['id']}")
    git_sha = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
    if spec.get("github_blob_sha") and git_sha != spec["github_blob_sha"]:
        raise ValueError(f"GitHub blob SHA mismatch: {spec['id']}")
    return {**spec, "bytes": len(body), "sha256": actual,
            "github_blob_verified": bool(spec.get("github_blob_sha"))}


def inspect_scifact(cache):
    with tarfile.open(cache / "scifact-data.tar.gz", "r:gz") as archive:
        documents = {}
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith(".jsonl"):
                stream = archive.extractfile(member)
                documents[Path(member.name).name] = [json.loads(line) for line in stream if line.strip()]
    corpus = documents["corpus.jsonl"]
    by_id = {row["doc_id"]: row for row in corpus}
    result = {"corpus_documents": len(corpus), "unique_document_ids": len(by_id),
              "abstract_sentences": sum(len(r["abstract"]) for r in corpus), "splits": {}}
    ids = {}
    for split in ("train", "dev", "test"):
        rows = documents[f"claims_{split}.jsonl"]
        ids[split] = {r["id"] for r in rows}
        labels, errors = Counter(), []
        rationales = documents_with_alternatives = claims_with_evidence = 0
        available = all("evidence" in r for r in rows)
        for row in rows:
            evidence = row.get("evidence", {})
            claims_with_evidence += bool(evidence)
            for doc_id, sets in evidence.items():
                documents_with_alternatives += len(sets) > 1
                for evidence_set in sets:
                    rationales += 1
                    labels[evidence_set["label"]] += 1
                    if int(doc_id) not in by_id:
                        errors.append({"claim_id": row["id"], "reason": "document_missing"})
                    elif not all(0 <= x < len(by_id[int(doc_id)]["abstract"]) for x in evidence_set["sentences"]):
                        errors.append({"claim_id": row["id"], "reason": "sentence_index_out_of_range"})
        result["splits"][split] = {"claims": len(rows), "unique_claim_ids": len(ids[split]),
            "evidence_key_present_in_all_records": available,
            "claims_with_nonempty_evidence": claims_with_evidence if available else None,
            "rationale_sets": rationales if available else None,
            "claim_document_pairs_with_alternative_rationales": documents_with_alternatives if available else None,
            "rationale_label_counts": dict(sorted(labels.items())) if available else None,
            "invalid_evidence_references": errors}
    result["claim_id_intersections"] = {a + "_" + b: len(ids[a] & ids[b])
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    result["test_note"] = "Missing evidence means unavailable annotations; it is not a NOINFO gold label."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'data/downloads.json').read_text(encoding='utf-8'))
    args.cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda spec: obtain(spec, args.cache, args.download), manifest['files']))
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Python standard-library read-only data inspection; not model or application evaluation',
              'scifact': inspect_scifact(args.cache)}
    dump(args.output, result)
    dump(args.output.with_name('source-files.json'), {'files': records})
    print(json.dumps({'dataset': 'scifact', 'inspected_files': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
