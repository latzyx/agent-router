"""Self-contained operator console so the API remains easy to run locally."""

from fastapi.responses import HTMLResponse


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lazy Agent Router</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#172033;font:15px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.shell{max-width:1050px;margin:58px auto;padding:0 24px}.hero{margin-bottom:28px}.hero p{color:#61708a}.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:20px}.card{background:#fff;border:1px solid #e4eaf3;border-radius:14px;padding:22px;box-shadow:0 8px 25px #24365a0b}h1{margin:0;font-size:28px}h2{margin:0 0 14px;font-size:18px}textarea,input,select{width:100%;border:1px solid #cbd5e1;border-radius:9px;padding:10px 12px;font:inherit}textarea{min-height:108px;resize:vertical}label{display:block;margin:12px 0 6px;font-weight:600;font-size:13px}button{margin-top:12px;background:#2563eb;border:0;border-radius:8px;padding:10px 16px;color:#fff;font:inherit;font-weight:650;cursor:pointer}button:hover{background:#1d4ed8}button:disabled{background:#94a3b8;cursor:wait}pre{margin:16px 0 0;background:#0f172a;color:#dbeafe;border-radius:9px;padding:15px;min-height:202px;white-space:pre-wrap;overflow:auto}.status{border-radius:9px;background:#eff6ff;color:#1e40af;padding:12px;line-height:1.5}.hint{color:#64748b;font-size:13px;line-height:1.55}@media(max-width:720px){.shell{margin:28px auto}.grid{grid-template-columns:1fr}}
</style></head><body><main class="shell"><section class="hero"><h1>Lazy Agent Router 控制台</h1><p>查询意图路由结果，或用 Hugging Face 模型与自有数据集启动训练。</p></section><section class="grid"><article class="card"><h2>意图路由测试</h2><textarea id="query">帮我查询采购审批流程</textarea><button id="route">执行路由</button><pre id="result">等待查询…</pre></article><article class="card"><h2>模型训练</h2><div class="status" id="training">正在获取训练状态…</div><label for="model">Hugging Face 模型</label><select id="model"><option value="hfl/chinese-macbert-base">MacBERT Base</option><option value="bert-base-chinese">BERT Base Chinese</option><option value="custom">自定义模型 ID / 本地路径</option></select><input id="custom-model" placeholder="例如：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment" hidden><label for="dataset">训练数据集（可选）</label><input id="dataset" type="file" accept=".jsonl"><p class="hint">数据格式为每行一个 JSON：{&quot;text&quot;:&quot;请查询订单&quot;,&quot;intent&quot;:&quot;workflow.query&quot;}。不上传时使用内置 datasets/raw/intents.jsonl。默认自动使用 GPU；CUDA 不可用时自动降级 CPU。可通过 LAZY_AGENT_ROUTER_DEVICE=cpu 强制使用 CPU。</p><button id="start">开始训练</button></article></section></main><script>
const $=id=>document.getElementById(id);const training=$("training"),start=$("start");
async function status(){const r=await fetch("/v1/training/status");const d=await r.json();training.textContent=d.message;start.disabled=d.state==="running";start.textContent=d.state==="running"?"训练中…":"开始训练"}
$("route").onclick=async()=>{const b=$("route"),out=$("result");b.disabled=true;out.textContent="正在路由…";try{const r=await fetch("/v1/route",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({query:$("query").value})});out.textContent=JSON.stringify(await r.json(),null,2)}catch(e){out.textContent="请求失败："+e.message}finally{b.disabled=false}};
$("model").onchange=()=>$("custom-model").hidden=$("model").value!=="custom";
start.onclick=async()=>{start.disabled=true;const form=new FormData();form.append("model_source",$("model").value==="custom"?$("custom-model").value:$("model").value);if($("dataset").files[0])form.append("dataset",$("dataset").files[0]);const r=await fetch("/v1/training/start",{method:"POST",body:form});const d=await r.json();training.textContent=d.message||d.detail;status()};status();setInterval(status,3000);
</script></body></html>"""


def console_page() -> HTMLResponse:
    return HTMLResponse(PAGE)
