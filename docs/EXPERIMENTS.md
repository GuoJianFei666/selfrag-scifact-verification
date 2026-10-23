# 训练、选择与正式评测

以下命令对应已经实现的入口。当前完整来源映射和 SelfRAG 7B 权重未准备好，尚未运行正式领域训练。不要把命令示例当作实验结果。

所有相对路径均以项目根目录为工作目录。先运行 `python -m evidence_rag --help` 查看参数。本文为节省重复，将实际解释器记作 `$pythonPath`：

```powershell
$pythonPath = '.venv/Scripts/python.exe'
```

## 1. 数据划分

`audit-sources` 只有在全部 train 主张唯一对齐时才写出内部 split。fit、model_select、calibration 按来源组分配，变体继承父主张；同组不能跨内部 partition。官方 dev 和 test 均保留原成员。当前 280 条 train 未匹配已完整记录，没有正式 split。

划分清单绑定数据清单路径和哈希，含每条训练主张的来源组与 partition。`train` 必须提供该清单，并检查完整 train 覆盖、同组隔离和视图继承。不要手工用随机组填补缺项。

## 2. 生成与复核视图

```powershell
& $pythonPath -m evidence_rag build-views --data var/scifact `
  --splits var/splits.json --partition fit --output var/fit-candidates.jsonl
& $pythonPath -m evidence_rag review-queue --views var/fit-candidates.jsonl `
  --round 1 --output var/review-round1.jsonl
& $pythonPath -m evidence_rag review-queue --views var/fit-candidates.jsonl `
  --round 2 --output var/review-round2.jsonl
```

两轮队列基于同一候选快照生成，排列使用不同固定种子。队列隐藏干预类别和拟定标签；阅读主张、标题与剩余摘要后，填写 reviewer、note、relation、decision。第二轮应在间隔后独立复核。工具检查两轮内容哈希和决定是否一致，不声称自动完成语义判断。

```powershell
& $pythonPath -m evidence_rag apply-reviews --views var/fit-candidates.jsonl `
  --reviews var/review-round1.jsonl var/review-round2.jsonl `
  --output var/fit-reviewed.jsonl
& $pythonPath -m evidence_rag prepare-training --views var/fit-reviewed.jsonl `
  --run B1 --output var/fit-b1.jsonl --audit var/fit-b1-exclusions.json
& $pythonPath -m evidence_rag prepare-training --views var/fit-reviewed.jsonl `
  --run B2 --output var/fit-augmented.jsonl --audit var/fit-augmented-exclusions.json
```

B1 只取原始 Full。B2/B3 共用同一份 `fit-augmented.jsonl`，其中派生视图必须 VERIFIED，并具备同组 Full 父视图。若没有已复核干预样本，B2/B3 会拒绝训练。Break 命中全部标注证据只是结构条件，剩余摘要或标题仍可能包含证据，必须语义复核。

## 3. 模型合约与训练

```powershell
& $pythonPath -m evidence_rag fetch-model --cache var/model-cache --include-weights
& $pythonPath -m evidence_rag model-contract --config configs/model.yaml --output var/model-contract.json
& $pythonPath -m evidence_rag train --config configs/B1.yaml --views var/fit-b1.jsonl --output var/B1
```

`configs/B1.yaml` 是起始配置，学习率 0.0001 尚未通过本项目实验选优。比较多个预先列出的学习率时，每次使用独立输出目录，并记录全部尝试。模型加载检查固定 revision、32016 词表、4096 上下文、控制标记、pad 和输出头维度。SFT 不截断超长样本，超长项带 ID 报错。

每个 epoch 保存 adapter、tokenizer、固定输入重载探针和当时的累计训练成本。`training_log.json` 记录损失、监督 token 和训练前向次数；`validation_forward_calls` 单列重载探针开销。最终保存后执行同一基础模型内的 adapter 重载检查。

冷重载需在新进程中执行。将 `configs/model.yaml` 的 adapter 改为选定检查点，再运行：

```powershell
& $pythonPath -m evidence_rag cold-reload --config configs/model.yaml `
  --probe var/B1/checkpoint-epoch-1/reload_probe.json --output var/cold-reload.json
```

## 4. 用真实检索选择检查点

先为内部 model_select 缓存真实检索结果，不能用 cited 文档代替召回：

```powershell
& $pythonPath -m evidence_rag cache-hits --data var/scifact `
  --index var/contriever-index --split train --partition model_select --splits var/splits.json `
  --top-k 5 --cache var/model-cache --output var/model-select-hits.json
& $pythonPath -m evidence_rag score-checkpoint --data var/scifact `
  --splits var/splits.json --hits var/model-select-hits.json --config configs/model.yaml `
  --adapter var/B1/checkpoint-epoch-1 --output var/B1-selection-epoch1
```

可追加已复核的 `--views` 和 `--quadruples`，计算内部联合指标 J。没有完整、已复核四元组时 J 为 null，不能写成零或伪造样本。对每个预先约定的检查点执行评分，将 `selection_record.json` 汇总成 JSON 数组，再使用 `select-checkpoint --records ... --output ...`。排名顺序固定为摘要证据 F1、J、较少训练前向次数、较早 step。

对 B1 学习率候选记录执行 `select-optimizer --records ... --output var/common_optimizer.json`，该命令只接受 B1 的 model_select 记录。B2/B3 使用此公共学习率：

```powershell
& $pythonPath -m evidence_rag train --config configs/B2.yaml --views var/fit-augmented.jsonl --output var/B2
& $pythonPath -m evidence_rag train --config configs/B3.yaml --views var/fit-augmented.jsonl --output var/B3
& $pythonPath -m evidence_rag compare-exposure --b2 var/B2/run_manifest.json --b3 var/B3/run_manifest.json
```

B3 默认 margin=1、方向项及保持项权重均为 0.1，是待检验的配置。方向项用 softplus 比较 Full/Break 的可判断分数，保持项用三个关系分布的 Jensen–Shannon 散度比较 Full/Keep。B2/B3 的 SFT 数据、序列、token、优化步数及初始化配置必须相同。比较选定检查点时也要选择相同暴露量；最终 B3 协议需冻结对应 B2 清单。

## 5. 独立校准

从 calibration partition 构造原始、已标注的 cited Full 对作为校准对象，不加入派生视图。

```powershell
& $pythonPath -m evidence_rag build-views --data var/scifact `
  --splits var/splits.json --partition calibration --original-only --output var/calibration-full.jsonl
& $pythonPath -m evidence_rag predict-views --views var/calibration-full.jsonl `
  --config configs/model.yaml --output var/calibration-predictions.jsonl
& $pythonPath -m evidence_rag calibration-rows --views var/calibration-full.jsonl `
  --predictions var/calibration-predictions.jsonl --output var/calibration-rows.jsonl
& $pythonPath -m evidence_rag calibrate --rows var/calibration-rows.jsonl `
  --threshold 0 --output var/calibration.json
```

温度在固定对数网格上最小化 NLL，阈值由事先约定的规则指定，不读取 dev。校准文件绑定原预测、视图和模型版本。将输出温度及阈值写入独立的最终模型配置，再冻结该配置。

## 6. 三阶段冻结

`freeze --phase ... --spec spec.json --output ...` 读取 `{"files": {"角色": "文件路径"}, "parameters": {...}}`，保存绝对路径及哈希，禁止覆盖。运行时检查必要角色，空清单不能放行正式 dev。

|阶段|必须绑定的文件角色|
|---|---|
|method|dataset_manifest、split_manifest、retrieval_config、training_config、selection_rule|
|views|method、views、quadruples、exclusion_audit|
|final_run|method、view_protocol、dataset_manifest、split_manifest、model_config、retrieval_hits、calibration、checkpoint_selection|

`selection_rule` 文件写明学习率尝试范围、检查点排名、阈值与干预生成规则版本。method 冻结后，`build-views --partition dev --frozen-method ...` 才能构造 dev 候选。

完成 dev 双轮复核后，运行 `prepare-evaluation --views ... --output ... --frozen-method ...`，输出固定视图、完整四元组和逐项排除原因，并生成 views 协议。保留全部原始 cited Full 对；J 的分母只包含四种条件均有效的固定案例。

final_run 的参数必须包含 baseline，以及三个输出路径 `prediction_path`、`official_prediction_path`、`diagnostic_prediction_path`。官方文件路径必须等于原始预测路径追加 `.official.jsonl`。B1/B2/B3 还需绑定 adapter_manifest、adapter_config、adapter_weights；B3 额外绑定 b2_comparison_manifest。B0 使用无 adapter 配置，并将基础发布版本记录为 checkpoint_selection。

## 7. 正式 dev 与诊断

```powershell
& $pythonPath -m evidence_rag run --data var/scifact --split dev `
  --splits var/splits.json --hits var/dev-hits.json --config var/final-model.yaml `
  --frozen-manifest var/final-run.json --output var/dev-predictions.jsonl
& $pythonPath -m evidence_rag evaluate --data var/scifact --split dev `
  --splits var/splits.json --predictions var/dev-predictions.jsonl.official.jsonl `
  --frozen-manifest var/final-run.json --output var/dev-metrics.json
```

`predict-views` 与 `diagnostics` 在输入含 dev 时，同样要求 `--data`、`--splits` 和 `--frozen-manifest`。预测回执绑定冻结清单与所有关联输出；评分前重新校验。输出缺行、重复、非法索引和关联文件改动会报错。

最终分别报告官方四项指标、真实召回、有效引用、三分类诊断、联合 J、固定覆盖率下的风险、失败率及训练/推理成本。没有覆盖率时风险为 null；无法达到指定覆盖率时明确标记不可达。一次干预的多种视图不作为相互独立的新主张。
