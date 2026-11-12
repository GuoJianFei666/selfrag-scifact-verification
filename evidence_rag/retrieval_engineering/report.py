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
    if report.get("finetune_evaluated_at"):
        lines.append(f"微调评测写入时间：{report['finetune_evaluated_at']}")
        lines.append("")
    finetune = report.get("finetune")
    if isinstance(finetune, dict) and finetune:
        lines.extend(_finetune_lines(finetune))
    return "\n".join(lines)


def merge_dev_report(existing, rows, finetune, evaluated_at):
    """Append fine-tuned rows. Previously measured rows stay byte-for-value as loaded."""
    if not isinstance(existing, dict) or not isinstance(existing.get("experiments"), list):
        raise ValueError("Existing dev report is missing experiments")
    if existing.get("split") != "dev":
        raise ValueError("Fine-tuned metrics are merged only into the dev report")
    allowed = {spec["id"] for spec in _finetune_specs()}
    experiments = [dict(row) for row in existing["experiments"]]
    index = {row.get("id"): position for position, row in enumerate(experiments)}
    for row in rows:
        if row.get("id") not in allowed or row.get("status") != "measured":
            raise ValueError(f"Refusing to merge {row.get('id')}")
        if row["id"] in index:
            experiments[index[row["id"]]] = row
        else:
            experiments.append(row)
    merged = dict(existing)
    merged["experiments"] = experiments
    merged["finetune"] = finetune
    merged["finetune_evaluated_at"] = evaluated_at
    models = dict(existing.get("models") or {})
    models["finetuned_embedding"] = {
        "model_id": finetune["base_model_id"],
        "revision": finetune["base_revision"],
        "weights": "local fine-tune, not committed",
        "pooling": "cls",
        "normalize": True,
    }
    merged["models"] = models
    return merged


def _finetune_specs():
    from .plan import FINETUNE_EXPERIMENTS
    return FINETUNE_EXPERIMENTS


def _finetune_lines(record):
    lines = ["## 训练集微调", ""]
    fields = (
        ("status", "状态"),
        ("base_model_id", "底模"),
        ("base_revision", "底模提交"),
        ("seed", "随机种子"),
        ("epochs", "epoch"),
        ("batch_size", "batch"),
        ("learning_rate", "学习率"),
        ("warmup_ratio", "warmup 比例"),
        ("weight_decay", "weight decay"),
        ("max_length", "最大长度"),
        ("loss", "损失"),
        ("loss_scale", "损失温度"),
        ("hard_negatives_per_example", "每条困难负样本"),
        ("hard_negative_source", "困难负样本"),
        ("positive", "正样本"),
        ("monitor_fraction", "监测集比例"),
        ("train_claims", "train 主张"),
        ("eligible_train_claims", "有证据的 train 主张"),
        ("fit_claims", "训练主张"),
        ("monitor_claims", "监测主张"),
        ("fit_examples", "训练样本"),
        ("monitor_examples", "监测样本"),
        ("unique_train_gold_documents", "train 金标准文档"),
        ("train_dev_gold_document_overlap", "train 与 dev 金标准文档交集"),
        ("mine_seconds", "负样本挖掘秒数"),
        ("train_seconds", "参数更新秒数"),
        ("optimizer_steps", "优化步数"),
        ("dev_used_for_training", "用 dev 训练"),
        ("dev_used_for_selection", "用 dev 选模"),
        ("weights_committed", "权重入库"),
    )
    for key, label in fields:
        if key in record:
            lines.append(f"- {label}：{record[key]}")
    for name in ("loss_before", "loss_after"):
        value = record.get(name)
        if isinstance(value, dict):
            lines.append(f"- {name}：fit {value.get('fit')}，monitor {value.get('monitor')}")
    if record.get("epoch_train_loss_mean") is not None:
        lines.append(f"- 每个 epoch 的训练损失均值：{record['epoch_train_loss_mean']}")
    reranker = record.get("reranker")
    if isinstance(reranker, dict):
        lines.append(
            f"- 重排序微调：{reranker.get('status')}，{reranker.get('reason')}"
        )
        if "probe_step_seconds" in reranker:
            lines.append(
                f"- 重排序探针：batch {reranker.get('probe_batch_size')}，一步 {reranker.get('probe_step_seconds')} 秒，损失 {reranker.get('probe_loss')}"
            )
    if record.get("loss_mode"):
        lines.append(f"- 损失口径：{record['loss_mode']}")
    lines.append("")
    return lines


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
