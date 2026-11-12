"""Render a markdown table from a measured results object. No filled-in defaults."""
KS = ("1", "3", "5", "10", "20")


def render_markdown(report):
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Report requires experiments")
    lines = [
        "# SciFact dev 检索对比",
        "",
        "本表由结果 JSON 生成。Recall@k、MRR@10、nDCG@10 是合格主张上的宏平均，范围 0 到 1。没有实际跑过的实验写 `not yet run`，不补 0。",
        "",
    ]
    hardware = report.get("hardware")
    if isinstance(hardware, dict) and hardware:
        lines.append(
            "硬件：{cpu_model}，{cpu_cores} 核，内存 {ram_gib} GiB，GPU {gpu}。".format(
                cpu_model=hardware.get("cpu_model", "未记录"),
                cpu_cores=hardware.get("cpu_cores", "未记录"),
                ram_gib=hardware.get("ram_gib", "未记录"),
                gpu=hardware.get("gpu", "未记录"),
            )
        )
        lines.append("")
    if report.get("generated_at"):
        lines.append(f"生成时间：{report['generated_at']}")
        lines.append("")
    models = report.get("models")
    if isinstance(models, dict):
        for name, spec in models.items():
            if isinstance(spec, dict) and spec.get("model_id"):
                lines.append(f"- {name}: `{spec['model_id']}` @ `{spec.get('revision', '未记录')}`")
        if lines[-1] != "":
            lines.append("")
    for note in report.get("metric_notes") or []:
        lines.append(f"- {note}")
    if report.get("metric_notes"):
        lines.append("")
    lines.extend([
        "| Experiment | Status | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Mean latency (s) | Index build (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in experiments:
        lines.append("| " + " | ".join(_cells(row)) + " |")
    lines.append("")
    for row in experiments:
        detail = row.get("notes") or row.get("reason")
        if detail and row.get("status") != "measured":
            lines.append(f"- `{row.get('id')}`: {detail}")
        elif row.get("notes"):
            lines.append(f"- `{row.get('id')}`: {row['notes']}")
    if any(row.get("notes") or (row.get("reason") and row.get("status") != "measured") for row in experiments):
        lines.append("")
    return "\n".join(lines)


def _cells(row):
    label = row.get("label") or row.get("id")
    status = row.get("status")
    if status in ("not_run", "skipped"):
        return [str(label), *["not yet run"] * 10]
    if status != "measured":
        raise ValueError(f"measured experiment is missing a finished status: {row.get('id')}")
    recall = row.get("recall")
    latency = row.get("latency_seconds")
    if not isinstance(recall, dict) or any(key not in recall for key in KS):
        raise ValueError(f"measured experiment is missing recall: {row.get('id')}")
    if "mrr_at_10" not in row or "ndcg_at_10" not in row or not isinstance(latency, dict) or "mean" not in latency:
        raise ValueError(f"measured experiment is missing metrics: {row.get('id')}")
    if "index_build_seconds" not in row:
        raise ValueError(f"measured experiment is missing index build time: {row.get('id')}")
    build = row["index_build_seconds"]
    build_cell = "reused" if build is None else f"{float(build):.3f}"
    return [
        str(label),
        "measured",
        *[_rate(recall[key]) for key in KS],
        _rate(row["mrr_at_10"]),
        _rate(row["ndcg_at_10"]),
        f"{float(latency['mean']):.3f}",
        build_cell,
    ]


def _rate(value):
    return f"{float(value):.4f}"
