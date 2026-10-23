"use strict";
const $ = id => document.getElementById(id);
let documents = [];
let report = null;
let ready = {retrieval_ready:false,generation_ready:false};
let busy = false;
const relations = {SUPPORT:"支持",CONTRADICT:"反驳",NOINFO:"当前摘要信息不足"};
const states = {accepted:"已接受",review_required:"需要人工复核",invalid_output:"输出格式无效",execution_failed:"执行失败"};

async function api(path, value) {
  const response = await fetch(path, value === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(value)});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

function controls() {
  $("search").disabled = busy || !ready.retrieval_ready;
  $("verify").disabled = busy || !ready.retrieval_ready || !ready.generation_ready;
  $("search").textContent = busy ? "正在处理…" : "检索摘要";
}

function showDocument(doc) {
  $("empty").hidden = true; $("article").hidden = false;
  $("document-id").textContent = `SCIFACT / ${doc.doc_id}`;
  $("rank").textContent = `检索排序 ${doc.rank}`;
  $("title").textContent = doc.title;
  const result = report && report.results.find(r => r.doc_id === doc.doc_id);
  const selected = new Set(result ? result.original_sentence_ids : []);
  $("sentences").replaceChildren();
  doc.sentences.forEach((sentence,index) => {
    const row=document.createElement("li");
    const number=document.createElement("span"); number.className="sentence-number"; number.textContent=String(index);
    const text=document.createElement("span"); text.textContent=sentence;
    row.append(number,text); row.classList.toggle("evidence",selected.has(index)); $("sentences").append(row);
  });
  $("relation").hidden=!result;
  $("raw-detail").hidden=!result;
  $("review-reason").hidden=!(result && result.review_reason);
  if(result) {
    $("relation").className="relation " + (result.relation_candidate || "").toLowerCase();
    const probability=Number.isFinite(result.confidence) ? ` · 候选关系概率 ${(result.confidence*100).toFixed(1)}%` : "";
    $("relation").textContent=`${relations[result.relation_candidate] || "未产生有效关系"} · ${states[result.status] || result.status}${probability}`;
    $("review-reason").textContent=result.review_reason || "";
    $("raw").textContent=JSON.stringify(result,null,2);
  }
  document.querySelectorAll(".document-button").forEach(button=>button.classList.toggle("active",Number(button.dataset.docId)===doc.doc_id));
}

function renderDocuments() {
  $("documents").replaceChildren(); $("candidate-count").textContent=`${documents.length} 篇`;
  for(const doc of documents) {
    const button=document.createElement("button"); button.type="button"; button.className="document-button"; button.dataset.docId=doc.doc_id;
    button.textContent=doc.title;
    const meta=document.createElement("small"); meta.textContent=`${String(doc.rank).padStart(2,"0")} / ${doc.doc_id} · 检索分数 ${doc.score.toFixed(3)}`;
    button.append(meta); button.addEventListener("click",()=>showDocument(doc)); $("documents").append(button);
  }
  if(documents.length)showDocument(documents[0]);
  else {$("article").hidden=true;$("empty").hidden=false;}
}

async function run(verify) {
  if(busy)return;
  const claim=$("claim").value.trim();
  if(!claim){$("claim").focus();return;}
  busy=true;controls();$("error").hidden=true;$("export").hidden=true;report=null;
  documents=[];renderDocuments();
  try {
    const payload={claim,top_k:Number($("top-k").value)};
    const search=await api("/api/search",payload); documents=search.documents;
    if(verify)report=await api("/api/verify",payload);
    renderDocuments();$("reading-label").textContent=verify ? "逐文献核验报告" : "候选摘要与原句";
    if(report){$("export").href="/api/reports/"+report.request_id;$("export").hidden=false;}
  } catch(error){$("error").textContent=error.message;$("error").hidden=false;}
  finally{busy=false;controls();}
}

$("claim-form").addEventListener("submit",event=>{event.preventDefault();run(false);});
$("verify").addEventListener("click",()=>run(true));
async function health() {
  try {
    ready=await api("/api/health");
    $("corpus").textContent=`${ready.corpus_documents.toLocaleString()} 篇摘要`;
    $("retriever").textContent=ready.retrieval_ready ? "检索器已就绪" : "检索器未加载";
    $("model").textContent=ready.generation_ready ? "SelfRAG 已加载" : "SelfRAG 未加载";
    $("actual-scope").textContent=`检索范围：SciFact 全部 ${ready.corpus_documents.toLocaleString()} 篇英文摘要。`;
    $("versions").textContent=JSON.stringify({corpus_sha256:ready.corpus_hash,retrieval:ready.retrieval,model:ready.model},null,2);
    $("availability").textContent=ready.generation_ready ? "核验结果按文献分别展示，原句编号由系统回查。" :
      (ready.retrieval_ready ? "可以检索并阅读原文摘要。模型加载后可进行关系判断与证据核验。" : "语料已导入。配置检索索引后可检索摘要，加载模型后可核验主张。");
  } catch(error){$("availability").textContent="服务连接失败："+error.message;}
  controls();
}
health();
