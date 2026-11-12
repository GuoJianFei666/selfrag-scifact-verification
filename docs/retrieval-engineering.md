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

## 不要提交的文件

模型权重、`var/` 下的索引和 Milvus Lite 的 `*.db`、以及完整 SciFact 数据。可以提交的是 `results/retrieval-dev.json` 和由它生成的 `results/retrieval-dev.md`。
