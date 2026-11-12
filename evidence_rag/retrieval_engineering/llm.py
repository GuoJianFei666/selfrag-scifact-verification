"""Optional OpenAI-compatible query rewrite and HyDE. Disabled without credentials."""
import json
import urllib.request


def llm_settings(spec, env):
    key = env.get(spec["api_key_env"], "").strip()
    model = env.get(spec["model_env"], "").strip()
    base = env.get(spec["base_url_env"], "").strip() or spec["default_base_url"]
    missing = [name for name, present in ((spec["api_key_env"], key), (spec["model_env"], model)) if not present]
    if missing:
        return {"enabled": False, "reason": "missing " + ", ".join(missing)}
    return {"enabled": True, "base_url": base.rstrip("/"), "api_key": key, "model": model}


def gated_transform(settings, experiment_id, label, execute):
    """Run an LLM experiment only when the endpoint is configured."""
    if not settings["enabled"]:
        return {"id": experiment_id, "label": label, "status": "skipped", "reason": settings["reason"]}
    return execute()


def rewrite_query(claim, complete):
    prompt = (
        "Rewrite this scientific claim as a short standalone search query for finding "
        "supporting or contradicting paper abstracts. Return only the query.\n\nClaim:\n" + claim
    )
    return _nonempty(complete(prompt), "Query rewrite")


def hyde_passage(claim, complete):
    prompt = (
        "Write a short scientific abstract that would contain evidence relevant to the claim. "
        "Do not say whether the claim is true. Return only the abstract.\n\nClaim:\n" + claim
    )
    return _nonempty(complete(prompt), "HyDE")


def chat_complete(base_url, api_key, model, prompt, opener=urllib.request.urlopen, timeout=60):
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
        return payload["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError("OpenAI-compatible chat completion failed") from exc


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} returned empty text")
    return value.strip()
