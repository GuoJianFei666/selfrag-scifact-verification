# 8GB 上的 B0 / B1

RTX 5060 有 8GB 显存和 16GB 内存。SelfRAG-7B 的 BF16 参数约 13.5GB（权重索引里的 `total_size` 为 13477093376 字节，分成两个 `pytorch_model-*.bin`），FP16 推理放不进这张卡。因此这台机器上的对照主线改为 4-bit：

- B0：冻结的 SelfRAG-7B，NF4 加载后直接推理，不挂 adapter。
- B1：同一套 NF4 基座加 LoRA（QLoRA）。只保存 adapter，推理时再挂到 4-bit 基座上。

两边的量化设置相同：NF4、double quant、计算类型 bfloat16。不把 LoRA 合并回 FP16 模型；合并会再要大约 13GB 内存。B2/B3 仍是原来的全精度配置，这一步不跑。`configs/B1.yaml` 也放不下，不要用它做这次训练。

固定版本不变：Self-RAG 代码 `1fcdc420`，SelfRAG-7B `190261383b0779ff66d2f95a73c7ad267d94b820`，SciFact 代码 `68b98a56`。

下面的 F1、召回、风险覆盖、时延和显存都还没有跑出来。结果文件出现之前，不要写数字。

## 原生 Windows

这条通路用 bitsandbytes 的 NF4，不用 Triton，也不用 Unsloth。bitsandbytes 从 0.45.3 起为 Windows x86-64 提供 CUDA 12.8 二进制，里面包含 Blackwell 的 sm_120。RTX 5060 是计算能力 12.0。因此用原生 Windows 即可，不需要为了这套库改走 WSL2。

驱动要能支持 CUDA 12.8 或更新。`nvidia-smi` 右上角的 CUDA 版本是驱动能力，不是本机必须另装的 CUDA Toolkit。PyTorch 用 CUDA 12.8 的轮子，bitsandbytes 会去加载对应的 `libbitsandbytes_cuda128.dll`。

如果 `python -m bitsandbytes` 报没有 kernel image，或找不到 CUDA 12.8 的动态库，先升级 bitsandbytes，并确认 PyTorch 是 CUDA 12.8 或更新的构建。这是安装不匹配，不是 Windows 不能做 4-bit。

## 安装

在仓库根目录打开 PowerShell。不要先执行 README 里的 CPU 版 PyTorch；那个环境不能在这张卡上做 4-bit。

```powershell
python -m venv .venv
& '.venv/Scripts/python.exe' -m pip install --upgrade pip
& '.venv/Scripts/python.exe' -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cu128
& '.venv/Scripts/python.exe' -m pip install -e '.[test,nf4]'
& '.venv/Scripts/python.exe' -m pip check
```

`nf4` 额外依赖是 `bitsandbytes>=0.45.3`。先装 CUDA 版 PyTorch，再装项目，避免 pip 另拉一个 CPU 版 torch。

检查：

```powershell
& '.venv/Scripts/python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability()); print(torch.cuda.get_device_name())"
& '.venv/Scripts/python.exe' -m bitsandbytes
```

`get_device_capability()` 应为 `(12, 0)`。bitsandbytes 的自检应显示它选中了与当前 PyTorch 匹配的 CUDA 12.8（或更新）库。然后可以跑 CPU 测试，确认环境没有把工程测试弄坏。测试不下载 7B：

```powershell
& '.venv/Scripts/python.exe' -m pytest -q
```

后面的 GPU 命令都加上：

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:PYTHONIOENCODING = 'utf-8'
```

跑的时候关掉浏览器和其他占显存的程序。权重是两个 BF16 分片，索引里的总字节数是 13477093376。按这个数对半看，单个分片大约 6–7GB。加载时一次处理一个分片，同时积累 4-bit 权重。这是按文件大小做的估计，不是这台机器上的内存曲线。系统再占掉几 GB 的话，加载会被杀掉。

## 数据和权重

冒烟的 cited 模式只需要官方数据和权重，不需要索引，也不需要来源划分。检索对照还需要索引和命中缓存。划分仍是 B1 正式训练的前提。

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag import-data --download `
  --archive var/downloads/scifact-data.tar.gz --output var/scifact
& '.venv/Scripts/python.exe' -m evidence_rag fetch-model --cache var/model-cache --include-weights
```

检索对照再执行 README 里的 `build-index`，然后：

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag cache-hits --data var/scifact `
  --index var/contriever-index --split dev --top-k 5 --cache var/model-cache `
  --output var/dev-hits.json
```

`audit-sources` 在 280 条 train 主张对不上来源时会退出，并且不写 `var/splits.json`。这一步没有补这些缺项。没有这份划分，B1 训练会拒绝开始。冒烟推理不依赖它。

## 冒烟

先确认 4-bit 加载、一条关系打分和结果文件。`--limit` 是按 claim id 排序后的前 N 条主张，不是正式成绩。输出在 `var/results/`，该目录不入库。

B0，2 条 train 主张的原始 cited 摘要：

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag experiment --group B0 `
  --config configs/B0-4bit.yaml --data var/scifact --split train `
  --cited --limit 2 --output var/results/smoke-b0
```

B1 的训练冒烟要等 `var/splits.json` 和 `var/fit-b1.jsonl` 都有了再跑。它只更新 2 条视图，`run_manifest.json` 里会写下 `smoke_limit`。这种 adapter 不能进入正式冻结协议。

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag train --config configs/B1-4bit.yaml `
  --views var/fit-b1.jsonl --output var/B1-smoke --limit 2
& '.venv/Scripts/python.exe' -m evidence_rag experiment --group B1 `
  --config configs/B0-4bit.yaml --adapter var/B1-smoke --data var/scifact `
  --split train --cited --limit 2 --output var/results/smoke-b1
```

第二条命令故意从 B0 的配置出发，只用 `--adapter` 挂上刚保存的 adapter，避免另写一份量化参数。B0 不能加 `--adapter`。

权重已经在本地时，冒烟的时间主要花在加载模型。估计是几分钟，不是实测。

## 全量 B0

官方四项（摘要标签、摘要标签加证据、句子选择、句子选择加标签）走检索命中，不用 gold 文档冒充检索：

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag experiment --group B0 `
  --config configs/B0-4bit.yaml --data var/scifact --split dev `
  --hits var/dev-hits.json --output var/results/B0-dev
```

风险覆盖和三分类诊断走原始 cited Full。这些视图的标签来自 SciFact 标注，`verification` 为 VERIFIED。没有已复核四元组时，`joint` 写成 null，不写成 0。

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag experiment --group B0 `
  --config configs/B0-4bit.yaml --data var/scifact --split dev `
  --cited --output var/results/B0-dev-cited
```

已有复核视图和四元组时，把它们加到对应命令上：`--views ... --quadruples ...`。检索运行上的 `--views` 必须和该次预测的 view id 对齐，否则诊断会拒绝这份文件。

`experiment` 会把传入的 `--frozen-manifest` 做完整性校验，并把哈希写入结果。它不替代 `run` / `evaluate` 那套路径绑定。没传清单时，`frozen_manifest_sha256` 为 null；那份文件是一次真实运行，还不是锁定后的正式 dev。

## 全量 B1

训练前先看长度。监督编码超过 `max_length` 会带 view id 报错，不截断，也不悄悄丢掉。

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag audit-lengths --views var/fit-b1.jsonl `
  --max-length 2048 --cache var/model-cache --output var/fit-b1-lengths.json
& '.venv/Scripts/python.exe' -m evidence_rag train --config configs/B1-4bit.yaml `
  --views var/fit-b1.jsonl --output var/B1
```

`configs/B1-4bit.yaml` 的设置是：NF4、double quant、bfloat16、梯度检查点、每次前向 1 条、梯度累积 16、`paged_adamw_8bit`、最大长度 2048、3 个 epoch。学习率 0.0001 仍是原来的起始值，这次没有做选择实验。2048 沿用原来的上下文合约。此前在 CPU 上对 tokenizer 做过长度检查，最长的输入加目标是 1809，能放进 2048。那次检查没有加载 7B，也不是显存测量。

每个 epoch 会在 `var/B1/checkpoint-epoch-N` 写一份 adapter。最终目录也只有 adapter、tokenizer 和训练记录，没有合并后的全量权重。

推理挂这份 adapter，基座仍是 B0 用的 NF4：

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag experiment --group B1 `
  --config configs/B1-4bit-infer.yaml --data var/scifact --split dev `
  --hits var/dev-hits.json --output var/results/B1-dev
& '.venv/Scripts/python.exe' -m evidence_rag experiment --group B1 `
  --config configs/B1-4bit-infer.yaml --data var/scifact --split dev `
  --cited --output var/results/B1-dev-cited
```

`configs/B1-4bit-infer.yaml` 的 `adapter` 是 `var/B1`。量化、精度、revision、温度和阈值与 `configs/B0-4bit.yaml` 相同。

网页如果要加载这张卡上的 B0，不要用默认的 `configs/model.yaml`（那是 FP32）。可以：

```powershell
.\Start.ps1 -WithModel -ModelConfig configs/B0-4bit.yaml
```

## 结果文件

每次 `experiment` 在输出目录写三份文件：

| 文件 | 内容 |
| --- | --- |
| `predictions.jsonl` | 逐条预测，含 `status` 和 `elapsed_seconds` |
| `official.jsonl` | 交给现有官方指标的导出 |
| `results.json` | 配置、峰值显存、逐条时延和指标 |

`results.json` 里有这些字段：`group`、`condition`（`retrieved` 或 `cited`）、`smoke`、`config`、`peak_vram`、`per_example`、`official`、`diagnostics`、`joint`、`citation`、`frozen_manifest_sha256`。没有 CUDA 时，峰值显存的字节数是 null，不会填 0 去冒充测量。

指标函数是仓库里原有的：

- `official`：摘要级关系、摘要级关系加证据、句子选择、句子选择加标签。gold 的 relevant 计数留在分母里。模型失败或给出 NOINFO 时，不会被导出成一条证据。
- `diagnostics`：只在 cited 运行，或检索运行另外传入了对齐的 `--views` 时才有。`failure_rate` 的分母是全部视图，`invalid_output` 和 `execution_failed` 留在里面。固定覆盖率上的风险也在这里；覆盖不到时 `reachable` 为 false，风险是 null。
- `citation`：句子编号能否映射回该摘要的原句。失败行算在 `n` 里。
- `joint`：四个条件都有效才计入。没传四元组就是 null。

训练目录里的 `run_manifest.json` 另有 `peak_vram`、`gradient_checkpointing`、`smoke_limit` 和 `optimizer`。

## 显存和时间（估计，不是这张卡上的测量）

- 4-bit 常驻权重大约 4.5–5.5GB。这是按参数量和 4-bit 打包估的。
- B0 推理还要加上 KV cache。2048 上下文时，整次加载加生成的峰值估计落在 6–8GB，长摘要会贴近上限。
- B1 训练是 batch 1、梯度检查点和 8-bit 分页优化器。2048 长度的反向峰值估计也在 8GB 上下，有可能超出。真的超出时，进程会失败并留下报错；不要改成截断。
- 冒烟在权重已缓存时，估计是几分钟。dev 检索大约是 300 条主张乘以 top-5。按每条数秒粗算，全量是小时级。训练 3 个 epoch 同样是小时级。以 `elapsed_seconds` 和 `peak_vram` 里的实测为准。

这些数字都还没有在 RTX 5060 上测过。
