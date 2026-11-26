/* Clean-up lab tab: compare photo clean-up options on one scan. */
let LABBOUND=false;
async function labInit(){ try{const p=await api('/api/cleanup/packs'),sel=$('lpack'),cur=sel.value;sel.innerHTML='<option value="auto">Best match (newest on ties)</option>'+p.map(x=>`<option value="${esc(x.file)}">${esc(x.name)} · default: ${esc(x.default)}${x.has_reference?'':' · no contrast reference'}</option>`).join('');sel.value=[...sel.options].some(o=>o.value===cur)?cur:'auto'}catch(e){}
 if(!LABBOUND){LABBOUND=true;const z=$('ldrop');['dragover','dragenter'].forEach(e=>z.addEventListener(e,x=>{x.preventDefault();z.classList.add('over')}));['dragleave','drop'].forEach(e=>z.addEventListener(e,x=>{x.preventDefault();z.classList.remove('over')}));z.addEventListener('drop',e=>labRun(e.dataTransfer.files[0]));$('lfile').onchange=e=>{labRun(e.target.files[0]);e.target.value=''}}}
async function labRun(f){if(!f)return;const o=$('lres');o.innerHTML='<div class="msg">Aligning and reading with every option…</div>';
 try{const r=await api('/api/cleanup?pack='+encodeURIComponent($('lpack').value),{method:'POST',body:f});labShow(r)}catch(e){o.innerHTML=`<div class="msg bad">${esc(e.message)}</div>`}}
function labShow(r){const ok=r.methods.filter(m=>m.available),name=m=>m.label;
 const cards=r.methods.map(m=>m.available?`<div class="card" style="margin:0"><a href="${m.image}" target="_blank" rel="noopener"><img src="${m.image}" alt="${esc(m.label)}" style="width:100%;border:1px solid var(--line);border-radius:6px"></a>
   <div style="margin-top:8px"><b>${esc(m.label)}</b> ${m.is_pack_default?'<span class="badge bg-blue-lt">pack default</span>':''}</div>
   <div class="tag">${m.read} cells read of ${m.with_ink} with ink · <span class="${m.flagged?'warn':''}">${m.flagged} flagged</span> · mean confidence ${m.mean_confidence??'-'}</div>
   ${(m.checks||[]).length?`<div class="tag warn" style="margin-top:6px"><b>Card checks (${m.checks.length})</b><ul style="padding-left:16px;margin:4px 0 0">${m.checks.slice(0,8).map(c=>`<li>${esc(c.field)} visit ${c.visit}: ${esc(c.note)}${c.suggestion?` (suggested ${esc(c.suggestion)})`:''}</li>`).join('')}</ul></div>`:''}</div>`
   :`<div class="card" style="margin:0"><b>${esc(m.label)}</b><div class="tag" style="margin-top:6px">Not available: ${esc(m.note)}</div></div>`).join('');
 const rows=r.disagreements.slice(0,80).map(d=>`<tr><td>${esc(d.field)}</td><td>${d.visit}</td>`+ok.map(m=>{const x=d.reads[m.method];return `<td class="${x.flagged?'warn':''}">${x.text?esc(x.text):'<span class="tag">(blank)</span>'}<div class="tag">${x.confidence}</div></td>`}).join('')+'</tr>').join('');
 $('lres').innerHTML=`<div class="card"><h2>${esc(r.pack)} · ${r.inliers} inliers · photo covers ${r.covered_percent}% of the card</h2>
  <div class="tag">Pack default clean-up: ${esc(r.pack_default)}. Amber = flagged for review (confidence below the field's minimum).</div>
  <div class="charts" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">${cards}</div></div>
  <div class="card"><h2>Cells where the options disagree (${r.disagreements.length} of ${r.cells_total})</h2>
  ${r.disagreements.length?`<table class="table table-vcenter"><tr><th>Field</th><th>Visit</th>${ok.map(m=>`<th>${esc(m.label)}</th>`).join('')}</tr>${rows}</table>${r.disagreements.length>80?'<div class="tag">First 80 shown.</div>':''}`:'<div class="tag">Every option read every cell the same.</div>'}
  <ul class="tag" style="padding-left:18px;margin-top:10px">${r.notes.map(n=>`<li>${esc(n)}</li>`).join('')}</ul></div>`}

