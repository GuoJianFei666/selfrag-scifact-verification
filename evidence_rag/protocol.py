"""Content-addressed protocols and held-out calibration."""
import math
from pathlib import Path

from .common import digest, file_hash, read_json, utcnow, write_json

REQUIRED_FILES = {
    "method": {"dataset_manifest", "split_manifest", "retrieval_config", "training_config", "selection_rule"},
    "views": {"method", "views", "quadruples", "exclusion_audit"},
    "final_run": {"method", "view_protocol", "dataset_manifest", "split_manifest", "model_config",
                  "retrieval_hits", "calibration", "checkpoint_selection"},
}


def require_bindings(path, phase, bindings):
    record = verify_frozen(path)
    if record["phase"] != phase or not REQUIRED_FILES[phase] <= set(record["files"]):
        raise ValueError("Frozen protocol lacks required named inputs for " + phase)
    for name, actual in bindings.items():
        if name not in record["files"] or file_hash(actual) != record["files"][name]["sha256"]:
            raise ValueError("Actual input differs from frozen binding: " + name)
    return record


def require_final(path, dataset, splits, config=None, hits=None, prediction=None, diagnostic_views=None):
    if not path or not splits:
        raise ValueError("Final dev run requires a frozen protocol and source split manifest")
    bindings = {"dataset_manifest": Path(dataset) / "manifest.json", "split_manifest": splits}
    if config:
        bindings["model_config"] = config
    if hits:
        bindings["retrieval_hits"] = hits
    record = require_bindings(path, "final_run", bindings)
    from .data import validate_split
    validate_split(read_json(splits))
    method = require_bindings(record["files"]["method"]["path"], "method", bindings={
        "dataset_manifest": bindings["dataset_manifest"], "split_manifest": splits})
    view_protocol = require_bindings(record["files"]["view_protocol"]["path"], "views", {
        "method": record["files"]["method"]["path"],
        **({"views": diagnostic_views} if diagnostic_views else {})})
    if view_protocol["parameters"].get("partition") != "dev":
        raise ValueError("Final view protocol must use the complete frozen dev set")
    if record["parameters"].get("baseline") not in ("B0", "B1", "B2", "B3"):
        raise ValueError("Final protocol must identify B0/B1/B2/B3")
    import yaml
    cfg = yaml.safe_load(Path(record["files"]["model_config"]["path"]).read_text(encoding="utf-8"))
    calibration = read_json(record["files"]["calibration"]["path"])
    if calibration["fit_on"] != "calibration_original_cited" or any(cfg.get(k) != calibration[k] for k in ("temperature", "threshold")):
        raise ValueError("Inference configuration does not use frozen calibration")
    if record["parameters"]["baseline"] == "B0":
        if cfg.get("adapter") is not None:
            raise ValueError("B0 must use the frozen base without an adapter")
    else:
        adapter = Path(cfg["adapter"])
        for name, filename in (("adapter_manifest", "run_manifest.json"), ("adapter_config", "adapter_config.json"),
                               ("adapter_weights", "adapter_model.safetensors")):
            require_bindings(path, "final_run", {name: adapter / filename})
        selection = read_json(record["files"]["checkpoint_selection"]["path"])
        if selection["selected_on"] != "model_select" or Path(selection["selected"]["path"]).resolve() != adapter.resolve():
            raise ValueError("Final adapter differs from model_select choice")
        adapter_manifest = read_json(adapter / "run_manifest.json")
        if adapter_manifest["run"] != record["parameters"]["baseline"] or selection["selected"]["baseline"] != adapter_manifest["run"] or selection["selected"]["checkpoint_manifest_sha256"] != file_hash(adapter / "run_manifest.json"):
            raise ValueError("Adapter baseline or selected weight record changed")
        if adapter_manifest["run"] == "B3":
            from .training import validate_same_exposure
            if "b2_comparison_manifest" not in record["files"]:
                raise ValueError("B3 final run requires a frozen B2 comparison at equal exposure")
            validate_same_exposure(read_json(record["files"]["b2_comparison_manifest"]["path"]), adapter_manifest)
    version = calibration["model_version"]
    if version["revision"] != cfg.get("revision") or any(version[k] != cfg.get(k) for k in ("precision", "quantization")):
        raise ValueError("Calibration model conditions differ from inference")
    if cfg.get("adapter"):
        if not version.get("adapter") or version["adapter"]["manifest_sha256"] != file_hash(Path(cfg["adapter"]) / "run_manifest.json"):
            raise ValueError("Calibration belongs to a different adapter")
        if version["adapter"]["weights"]["adapter_model.safetensors"] != file_hash(Path(cfg["adapter"]) / "adapter_model.safetensors"):
            raise ValueError("Calibration adapter weights changed")
    elif version.get("adapter") is not None:
        raise ValueError("Base inference cannot use adapter calibration")
    for name in ("prediction_path", "official_prediction_path", "diagnostic_prediction_path"):
        if not isinstance(record["parameters"].get(name), str):
            raise ValueError("Final protocol must bind all output paths")
    if prediction and Path(prediction).resolve() not in {Path(record["parameters"][k]).resolve() for k in
                                                        ("prediction_path", "official_prediction_path", "diagnostic_prediction_path")}:
        raise ValueError("Prediction output path differs from final protocol")
    return record


def freeze_files(output, files, parameters, phase):
    if phase not in ("method", "views", "final_run"):
        raise ValueError("Unknown protocol phase")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Frozen protocol cannot be overwritten")
    entries = {}
    for name, path in files.items():
        original = Path(path).absolute()
        if original.is_symlink() or any(p.is_symlink() or p.is_junction() for p in original.parents) or original.is_junction():
            raise ValueError("Frozen inputs cannot traverse reparse points")
        path = original.resolve()
        if not path.is_file():
            raise ValueError("Frozen entries must be ordinary files")
        entries[name] = {"path": str(path), "sha256": file_hash(path)}
    record = {"phase": phase, "created_at": utcnow(), "files": entries, "parameters": parameters}
    record["content_hash"] = digest(record)
    write_json(output, record)
    return record


def verify_frozen(path):
    record = read_json(path)
    expected = record.pop("content_hash")
    if digest(record) != expected:
        raise ValueError("Protocol contents changed")
    for name, entry in record["files"].items():
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    record["content_hash"] = expected
    return record


def select_checkpoint(records, output):
    if not records or any(r["split"] != "model_select" for r in records):
        raise ValueError("Checkpoint selection only uses internal model_select")
    selected = min(records, key=lambda r: (-r["abstract_rationalized"], -(r["joint"] if r["joint"] is not None else -1), r["forward_calls"], r["step"], r["path"]))
    write_json(output, {"selected_on": "model_select", "rule": "abstract_rationalized,joint,cost,earlier_step", "selected": selected})
    return selected


def calibrate(rows, output, threshold=0.0):
    if not rows or any(r["split"] != "calibration" or r["view_kind"] != "Full" or r["condition"] != "cited" for r in rows):
        raise ValueError("Temperature fitting requires original cited calibration pairs")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be fixed inside [0,1]")
    versions = {digest(row["model_version"]) for row in rows}
    if len(versions) != 1:
        raise ValueError("Calibration predictions mix model versions")
    def nll(temperature):
        total = 0.0
        for row in rows:
            scores = row["label_scores"]
            if len(scores) != 3 or any(not math.isfinite(s) for s in scores) or type(row["label_index"]) is not int or not 0 <= row["label_index"] < 3:
                raise ValueError("Invalid calibration logits or label")
            scaled = [s / temperature for s in scores]
            peak = max(scaled)
            total += peak + math.log(sum(math.exp(s - peak) for s in scaled)) - scaled[row["label_index"]]
        return total / len(rows)
    grid = [math.exp(-2 + i * 4 / 160) for i in range(161)]
    temperature = min(grid, key=lambda t: (nll(t), t))
    result = {"fit_on": "calibration_original_cited", "temperature": temperature, "threshold": threshold,
              "n": len(rows), "nll_before": nll(1), "nll_after": nll(temperature), "grid_log_bounds": [-2, 2], "grid_points": 161}
    result.update(model_version=rows[0]["model_version"], input_rows_sha256=digest(rows))
    write_json(output, result)
    return result
