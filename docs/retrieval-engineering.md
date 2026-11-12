# 检索质量与评测

这条线和核验模型分开。核验应用仍然使用冻结的 Contriever 与 FAISS 精确索引。这里增加一条可对照的检索实验：用 LlamaIndex 把 SciFact 的标题和摘要写入 Milvus，再在官方 dev 的证据文档上比较嵌入、重排序、查询改写和分块。

不下载 7B 权重。Milvus 默认用 Milvus Lite，数据文件留在本机，不入库。查询改写和 HyDE 需要一个 OpenAI 兼容接口，默认关闭；没有密钥时对应命令正常退出，不发请求。

## 安装

Python 3.12。先装 CPU 版 PyTorch，再装带 `retrieval` 额外依赖的包。`retrieval` 包含 LlamaIndex、Milvus Lite 和 sentence-transformers。安装本身不下载嵌入模型。

Windows PowerShell，工作目录为仓库根目录：

```powershell
python -m venv .venv
& '.venv/Scripts/python.exe' -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
& '.venv/Scripts/python.exe' -m pip install -e '.[test,retrieval]'
& '.venv/Scripts/python.exe' -m pytest -q
```

Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test,retrieval]'
.venv/bin/python -m pytest -q
```

导入 SciFact。输出目录必须为空；已经导入过就不要重复执行。

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag import-data --download `
  --archive var/downloads/scifact-data.tar.gz --output var/scifact
```

```bash
.venv/bin/python -m evidence_rag import-data --download \
  --archive var/downloads/scifact-data.tar.gz --output var/scifact
```

## 一次跑完不含 LLM 的对照

这个命令跑 Contriever/FAISS、BGE 整篇摘要、BGE 加 bge-reranker，以及整篇、句子、滑窗和相邻语义四种分块。查询改写和 HyDE 只写入 `not yet run`，不会访问接口。

首次运行会把三个模型下载到 `var/model-cache`。Contriever、`BAAI/bge-small-en-v1.5` 和 `BAAI/bge-reranker-base` 的提交号写在 `configs/retrieval_engineering.yaml`。工作目录必须为空，重跑时换一个 `--work`，或者加上 `--resume` 继续已完成的实验。

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering compare `
  --data var/scifact --cache var/model-cache --work var/retrieval-runs `
  --output results/retrieval-dev.json --markdown results/retrieval-dev.md
```

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venv/bin/python -m evidence_rag.retrieval_engineering compare \
  --data var/scifact --cache var/model-cache --work var/retrieval-runs \
  --output results/retrieval-dev.json --markdown results/retrieval-dev.md
```

`results/retrieval-dev.json` 是原始记录。`results/retrieval-dev.md` 的表由这份 JSON 生成。表里的数字只来自当次运行；没有跑的格子是 `not yet run`。硬件、包版本和模型提交号写在 JSON 里，也抄在 Markdown 开头。

CPU 可以跑完。嵌入模型选的是英文 `bge-small-en-v1.5`（384 维，CLS 池化，向量做了归一化）。查询侧加上模型卡里的前缀 `Represent this sentence for searching relevant passages: `，文档侧不加。重排序用 `bge-reranker-base`。Milvus 索引类型是 FLAT，检索是精确的，不把近似误差算进对照。

## 每个实验单独的入口

| 命令 | 内容 |
| --- | --- |
| `baseline` | Contriever，attention-mask 平均池化，不做归一化，FAISS `IndexFlatIP` |
| `bge` | LlamaIndex + Milvus。`--chunking` 可选 `whole`、`sentence`、`sliding`、`semantic`，默认整篇摘要 |
| `rerank` | 整篇 BGE 先取前 50 篇，再用 bge-reranker-base 重排，指标看重排后的前 20 |
| `rewrite` | 查询改写和 HyDE。没有密钥时跳过 |
| `chunking` | 整篇、句子、滑窗、语义分块。`--variants` 用逗号选择 |
| `compare` | 上面除 `rewrite` 以外的全部，并把改写和 HyDE 记为尚未运行 |
| `finetune` | 只用 SciFact train 微调 bge-small。必须带 `--split train` |
| `eval-finetuned` | 用微调后的编码器在 dev 上评整篇摘要和滑窗，并写回同一张结果表 |

PowerShell 示例：

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering baseline `
  --data var/scifact --cache var/model-cache --work var/retrieval-baseline `
  --output var/retrieval-baseline/baseline.json
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering bge `
  --data var/scifact --cache var/model-cache --work var/retrieval-bge `
  --chunking whole --output var/retrieval-bge/bge.json
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering rerank `
  --data var/scifact --cache var/model-cache --work var/retrieval-rerank `
  --output var/retrieval-rerank/rerank.json
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering chunking `
  --data var/scifact --cache var/model-cache --work var/retrieval-chunking `
  --variants sentence,sliding,semantic --output var/retrieval-chunking/chunking.json
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering rewrite `
  --data var/scifact --config configs/retrieval_engineering.yaml `
  --output var/retrieval-rewrite.json
```

Linux：

```bash
.venv/bin/python -m evidence_rag.retrieval_engineering baseline \
  --data var/scifact --cache var/model-cache --work var/retrieval-baseline \
  --output var/retrieval-baseline/baseline.json
.venv/bin/python -m evidence_rag.retrieval_engineering bge \
  --data var/scifact --cache var/model-cache --work var/retrieval-bge \
  --chunking whole --output var/retrieval-bge/bge.json
.venv/bin/python -m evidence_rag.retrieval_engineering rerank \
  --data var/scifact --cache var/model-cache --work var/retrieval-rerank \
  --output var/retrieval-rerank/rerank.json
.venv/bin/python -m evidence_rag.retrieval_engineering chunking \
  --data var/scifact --cache var/model-cache --work var/retrieval-chunking \
  --variants sentence,sliding,semantic --output var/retrieval-chunking/chunking.json
.venv/bin/python -m evidence_rag.retrieval_engineering rewrite \
  --data var/scifact --config configs/retrieval_engineering.yaml \
  --output var/retrieval-rewrite.json
```

`rewrite` 在没有密钥时不读取 `--data` 里的语料，仍然要求该参数，以便和别的命令保持同一组参数。

## 查询改写和 HyDE

默认关闭。三个环境变量都有值时，`rewrite` 才会对一个 OpenAI 兼容的 `/chat/completions` 发请求：

| 变量 | 作用 |
| --- | --- |
| `RETRIEVAL_LLM_API_KEY` | 密钥。为空则跳过 |
| `RETRIEVAL_LLM_MODEL` | 模型名。为空则跳过 |
| `RETRIEVAL_LLM_BASE_URL` | 可选，默认 `https://api.openai.com/v1` |

```powershell
$env:RETRIEVAL_LLM_API_KEY = 'your-key'
$env:RETRIEVAL_LLM_MODEL = 'gpt-4.1-mini'
$env:RETRIEVAL_LLM_BASE_URL = 'https://api.openai.com/v1'
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering rewrite `
  --data var/scifact --cache var/model-cache --work var/retrieval-runs `
  --output results/retrieval-rewrite.json --markdown results/retrieval-rewrite.md
```

```bash
export RETRIEVAL_LLM_API_KEY=your-key
export RETRIEVAL_LLM_MODEL=gpt-4.1-mini
export RETRIEVAL_LLM_BASE_URL=https://api.openai.com/v1
.venv/bin/python -m evidence_rag.retrieval_engineering rewrite \
  --data var/scifact --cache var/model-cache --work var/retrieval-runs \
  --output results/retrieval-rewrite.json --markdown results/retrieval-rewrite.md
```

改写后的查询仍走 BGE 的查询前缀。HyDE 让模型写一段假设摘要，再按文档向量编码，不加查询前缀。`compare` 不读取这些变量，也不会因为本机已经设置了密钥就去调用接口。仓库里这次提交的 dev 结果没有跑这两项。

## Milvus Lite 和 Milvus 服务

`configs/retrieval_engineering.yaml` 的 `milvus.uri`：

- 以 `.db` 结尾：Milvus Lite。不需要单独起服务。程序在该文件所在目录写入每个集合一个库：`scifact_whole_abstract.db`、`scifact_sentence.db`、`scifact_sliding.db`、`scifact_semantic.db`。配置里的文件名本身不会被打开。
- 以 `http://`、`https://` 或 `tcp://` 开头：连接已有的 Milvus。`milvus.token` 只在这种情况下传给客户端，例如 `user:password`。集合名同上。

把本地实验改到服务上时，只改 `uri` 和 `token`，实验命令不用改。Lite 的库文件、FAISS 索引和模型缓存都在 `var/` 或配置指向的数据目录里，已被忽略，不要放进提交。

## 指标

评测集是 SciFact 的 dev。金标准文档是 `claim.evidence` 里的摘要，也就是标注为 SUPPORT 或 CONTRADICT 的文档。只被引用、但没有证据条目的摘要不计入金标准。没有金标准的主张仍会计入时延，不进入 Recall、MRR 和 nDCG。

- Recall@k：该主张的金标准文档有多少出现在前 k 名，再对合格主张做宏平均。k 为 1、3、5、10、20。
- MRR@10：第一篇金标准文档排名的倒数；10 名以外记 0。宏平均。
- nDCG@10：二元相关，折扣为 log2。宏平均。
- 时延：模型已经载入、并先做一次不计时的热身查询之后，每条主张一次检索的墙钟时间，包含查询编码。重排序还包含交叉编码器。报告平均值和中位数。
- 建索引时间：权重已经在本地缓存之后，编码语料并写完索引的墙钟时间。下载时间不计入。重排序若复用整篇 BGE 索引，这一格为复用，不把同一次建库再记一遍。

分块先按块检索，再按同一文档的最高分收成文档。分数相同则 doc id 较小的在前。句子块和滑窗、语义块会多取 `chunk_search_k`（默认 2000）个块，避免一篇特别长的摘要占满候选。滑窗是 2 句、步长 1。语义分块看相邻句子的余弦相似度，默认不低于 0.5，并且一块最多 3 句。这个阈值是固定预设，没有在 dev 上调过。

每块文本都带上标题。整篇摘要的文本与 Contriever 基线相同：标题、换行、再用空格接上摘要句子。

## 在训练集上微调 bge-small

只更新 `bge-small-en-v1.5` 的向量模型。`bge-reranker-base` 不微调。命令在开始前会用 8 条合成句做一次 AdamW 更新并丢掉这次权重，把这一步的耗时记进结果；评测不使用这次更新。

训练数据只来自 SciFact train 的 809 条主张。没有证据条目的主张不构造样本。dev 不能作为 `--split` 传给 `finetune`，也不会进入损失。监测集是有证据的 train 主张里、按种子打乱后取 `int(n * monitor_fraction)` 条，至少 1 条、至多 n-1 条。超参写在 `configs/retrieval_engineering.yaml` 的 `finetune` 段，在看到 dev 指标之前就固定，监测损失只用来观察拟合，不用来改轮数或学习率。

正样本：每个金标准文档（`claim.evidence` 里的 SUPPORT 或 CONTRADICT）取一个滑窗。窗长 2、步长 1，和目前 dev 上最好的分块一致。窗口必须盖住至少一句 rationale，优先盖住更多 rationale，其次更短，再次更靠前。文本格式与检索时相同：标题、换行、窗口内的句子。

困难负样本：先用微调前的 bge-small 对整篇摘要做余弦检索（查询带模型卡前缀，文档不带），去掉该主张的全部金标准文档，取得分最高的一篇。再在这篇里用主张和窗口的词面重合选一个同样的滑窗。同一 batch 里的其他正样本和其他困难负样本是 in-batch negatives。损失是 sentence-transformers 的 `MultipleNegativesRankingLoss`（scale 20）。一个 batch 里同一条主张只出现一次，避免把同一主张的另一篇证据当成负样本。查询文本在训练时已经加上前缀；保存的模型不写入 prompt，评测仍由原来的 BGE 封装在查询侧加前缀。

优化器是 AdamW。bias 和 LayerNorm 不做 weight decay。学习率线性 warmup 后再线性降到 0。梯度裁剪 1.0。随机种子、轮数、batch、学习率都在配置里。权重写到 `var/retrieval-runs/finetuned-bge/`，不入库。

PowerShell：

```powershell
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering finetune `
  --data var/scifact --cache var/model-cache --work var/retrieval-runs `
  --split train --model-dir var/retrieval-runs/finetuned-bge `
  --output var/retrieval-runs/finetune-record.json
& '.venv/Scripts/python.exe' -m evidence_rag.retrieval_engineering eval-finetuned `
  --data var/scifact --cache var/model-cache --work var/retrieval-runs `
  --split dev --model var/retrieval-runs/finetuned-bge `
  --record var/retrieval-runs/finetune-record.json `
  --results results/retrieval-dev.json --markdown results/retrieval-dev.md
```

Linux：

```bash
.venv/bin/python -m evidence_rag.retrieval_engineering finetune \
  --data var/scifact --cache var/model-cache --work var/retrieval-runs \
  --split train --model-dir var/retrieval-runs/finetuned-bge \
  --output var/retrieval-runs/finetune-record.json
.venv/bin/python -m evidence_rag.retrieval_engineering eval-finetuned \
  --data var/scifact --cache var/model-cache --work var/retrieval-runs \
  --split dev --model var/retrieval-runs/finetuned-bge \
  --record var/retrieval-runs/finetune-record.json \
  --results results/retrieval-dev.json --markdown results/retrieval-dev.md
```

`eval-finetuned` 会复用已有的 `--work`。新索引使用 `scifact_ft_` 前缀，不覆盖基线的 Milvus 文件。它只追加 `bge_finetuned_whole` 和 `bge_finetuned_sliding` 两行，不改已经测过的行。评测指标与前两步相同：Recall@1/3/5/10/20、MRR@10、nDCG@10、单条延迟、建索引时间。

语料是 train 和 dev 共用的。训练时配过对的摘要，有一部分也会出现在 dev 的金标准里。交集的篇数写在结果 JSON 的 `finetune.train_dev_gold_document_overlap`，这只是事后统计，不参与选模。被引用但标成 NOINFO 的文档可以成为困难负样本。dev 主张的文本和标签都没有进入损失。

重排序器、查询改写和 HyDE 都不在这次微调里。改写和 HyDE 在总表里仍然是 `not yet run`。

### 这次 dev 上的结果

下面的数来自 2026-11-12 这次运行，写在 `results/retrieval-dev.json`。机器是 Linux 6.12、Intel Xeon、4 核、内存 15.63 GiB、无 GPU，`OMP_NUM_THREADS=4`。底模提交 `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`。种子 42，2 个 epoch，batch 16，学习率 2e-5，68 步。挖困难负样本 195.916 秒，参数更新 129.080 秒。评测损失（dropout 关闭）训练对从 0.8836 降到 0.4519，监测对从 0.7372 降到 0.4111。

| 设置 | Recall@5 | MRR@10 | nDCG@10 | 单条延迟 (s) | 建索引 (s) |
| --- | --- | --- | --- | --- | --- |
| 微调前，整篇摘要 | 0.8997 | 0.8335 | 0.8611 | 0.234 | 262.651 |
| 微调后，整篇摘要 | 0.9359 | 0.8553 | 0.8824 | 0.231 | 278.832 |
| 微调前，滑窗 2/1 | 0.9309 | 0.8588 | 0.8854 | 1.455 | 573.232 |
| 微调后，滑窗 2/1 | 0.9495 | 0.8794 | 0.9018 | 1.493 | 600.433 |

整篇摘要的 Recall@5 提高 0.0362，MRR@10 提高 0.0218，nDCG@10 提高 0.0213。滑窗的 Recall@5 提高 0.0186，MRR@10 提高 0.0205，nDCG@10 提高 0.0163。滑窗的 Recall@10 和 Recall@20 与微调前相同；整篇的 Recall@20 也相同。完整的 R@1/3/5/10/20 在生成的总表里。

监测损失和 dev 检索一起变好，没有出现监测损失上升、dev 变差这种只记住 fit 主张的情况。182 篇 dev 金标准文档里有 110 篇也是 train 的金标准文档。模型训练时见过这些摘要，配对的是 train 主张，不是 dev 主张。因此 dev 上的涨幅不能全部看成对没见过的文献的泛化。重排序器没有微调；合成句上的一步 AdamW 用了 3.808 秒，权重已丢弃。

## 不要提交的文件

模型权重、`var/` 下的索引和 Milvus Lite 的 `*.db`、`var/retrieval-runs/finetuned-bge/`，以及完整 SciFact 数据。可以提交的是 `results/retrieval-dev.json` 和由它生成的 `results/retrieval-dev.md`。
