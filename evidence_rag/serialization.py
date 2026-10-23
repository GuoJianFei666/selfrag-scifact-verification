"""One serialization contract for SFT, relation scoring and evidence generation."""
import json

from .data import RELATIONS
from .views import training_target

TEMPLATE_VERSION = "scifact-json-v1"
RELATION_PREFIX = '{"relation":"'
RELATION_END = '","sentence_ids":'
INSTRUCTION = ("Determine whether this abstract supports, contradicts, or lacks sufficient information about the claim. "
               "Return JSON with relation SUPPORT, CONTRADICT, or NOINFO and sentence_ids containing local sentence numbers. "
               "Use an empty sentence list for NOINFO. Select a complete evidence set for SUPPORT or CONTRADICT.")


class ContextLimitError(RuntimeError):
    pass


def prompt(view):
    sentences = "\n".join(f"[{i}] {text}" for i, text in enumerate(view.sentences))
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\nClaim: {view.claim}\n<paragraph>\nTitle: {view.title}\n{sentences}\n</paragraph>\n\n### Response:\n"


def target_json(view):
    return json.dumps(training_target(view), ensure_ascii=False, separators=(",", ":"))


def encode_supervised(view, tokenizer, max_length=2048):
    prefix, target = prompt(view), target_json(view)
    if not tokenizer.is_fast:
        raise ValueError("Supervision boundary checks require a fast tokenizer with offsets")
    encoded = tokenizer(prefix + target, add_special_tokens=False, return_offsets_mapping=True)
    ids, labels = [], []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        labels.append(-100)
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        if start < len(prefix) < end:
            raise ValueError("Tokenizer merged an input token across the target boundary")
        ids.append(token)
        labels.append(token if start >= len(prefix) and end > start else -100)
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS token is required")
    ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    if len(ids) > max_length:
        raise ContextLimitError(f"Overlong sample {view.view_id}: {len(ids)} > {max_length}; no truncation applied")
    if not any(value != -100 for value in labels[:-1]):
        raise ValueError("Sample has no supervised JSON tokens")
    relation_prefix = tokenizer.encode(prefix + RELATION_PREFIX + training_target(view)["relation"] + RELATION_END,
                                       add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        relation_prefix = [tokenizer.bos_token_id] + relation_prefix
    if ids[:len(relation_prefix)] != relation_prefix:
        raise ValueError("Relation candidate is not a token prefix of the SFT target")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def relation_candidates(view, tokenizer, max_length=2048):
    sequences = []
    for relation in RELATIONS:
        ids = tokenizer.encode(prompt(view) + RELATION_PREFIX + relation + RELATION_END, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        if len(ids) > max_length:
            raise ContextLimitError(f"Overlong relation candidate: {view.view_id}")
        probes = ([],) if relation == "NOINFO" else ([0], [max(0, len(view.sentences) - 1)])
        for evidence in probes:
            probe = json.dumps({"relation": relation, "sentence_ids": evidence}, separators=(",", ":"))
            complete = tokenizer.encode(prompt(view) + probe, add_special_tokens=False)
            if tokenizer.bos_token_id is not None:
                complete = [tokenizer.bos_token_id] + complete
            if complete[:len(ids)] != ids:
                raise ValueError("Relation/evidence token boundary differs from the canonical target")
        sequences.append(ids)
    shared = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        shared += 1
    if shared < 1 or any(shared == len(seq) for seq in sequences):
        raise ValueError("Relation candidate token boundaries are invalid")
    return sequences, shared


def collate(samples, pad_token_id):
    import torch
    if pad_token_id is None or not samples:
        raise ValueError("Padding token and nonempty samples are required")
    width = max(len(s["input_ids"]) for s in samples)
    return {key: torch.tensor([s[key] + [padding] * (width - len(s[key])) for s in samples], dtype=torch.long)
            for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def relation_scores(model, sequences, shared, pad_token_id):
    import torch
    width = max(map(len, sequences))
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([s + [pad_token_id] * (width - len(s)) for s in sequences], device=device)
    attention = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in sequences], device=device)
    logits = model(input_ids=ids, attention_mask=attention).logits
    scores = []
    for row, seq in enumerate(sequences):
        positions = torch.arange(shared, len(seq), device=device)
        log_probs = torch.nn.functional.log_softmax(logits[row, positions - 1].float(), dim=-1)
        scores.append(log_probs.gather(1, ids[row, positions, None]).sum())
    return torch.stack(scores)
