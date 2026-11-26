/* Playground tab: try a card/model pair on a fresh photo. */
let PGBOUND=false,PGOPT=null;
async function pgInit(){const cs=$('pgcard'),ms=$('pgmodel');
 try{PGOPT=await api('/api/playground/options')}catch(e){$('pghelp').textContent='Could not load the card and model lists: '+e.message;return}
 const cc=cs.value,cm=ms.value;
 cs.innerHTML=PGOPT.cards.map(c=>`<option value="${esc(c.id)}">${esc(c.id)}${c.from_pack?' · from a pack':c.pack_versions.length?'':' · no model yet'}</option>`).join('');cs.value=[...cs.options].some(x=>x.value===cc)?cc:((PGOPT.cards[0]||{}).id||'');
 ms.innerHTML=PGOPT.models.map(m=>`<option value="${esc(m.id)}">${esc(m.label)}</option>`).join('');
 const pick=()=>{const c=PGOPT.cards.find(x=>x.id===cs.value)||{};const noOwn=!(c.pack_versions||[]).length;const own=[...ms.options].find(o=>o.value==='card');if(own)own.disabled=noOwn;
  $('pghelp').textContent=noOwn?`${cs.value} has no trained pack, so pick another model to read it with.`:'';if(noOwn&&ms.value==='card'){const o=[...ms.options].find(x=>x.value!=='card');if(o)ms.value=o.value}};
 cs.onchange=()=>{ms.value='card';pick()};ms.value=[...ms.options].some(x=>x.value===cm)?cm:'card';pick();
 if(!PGBOUND){PGBOUND=true;const z=$('pgdrop');['dragover','dragenter'].forEach(e=>z.addEventListener(e,x=>{x.preventDefault();z.classList.add('over')}));['dragleave','drop'].forEach(e=>z.addEventListener(e,x=>{x.preventDefault();z.classList.remove('over')}));z.addEventListener('drop',e=>pgRun(e.dataTransfer.files[0]));$('pgfile').onchange=e=>{pgRun(e.target.files[0]);e.target.value=''};
  $('pgpack').onchange=async e=>{const f=e.target.files[0];e.target.value='';if(!f)return;try{const r=await api('/api/pack/import',{method:'POST',body:f});await pgInit();$('pgcard').value=r.id;$('pgcard').onchange();$('pghelp').textContent=`Added ${r.file}: its template is in the Card list and its reader in the Model list.`}catch(x){$('pghelp').textContent='Could not add it: '+x.message}}}}
async function pgRun(f){if(!f)return;const o=$('pgres');o.innerHTML='<div class="msg">Aligning and reading…</div>';
 try{const r=await api('/api/cleanup?card='+encodeURIComponent($('pgcard').value)+'&model='+encodeURIComponent($('pgmodel').value),{method:'POST',body:f});pgShow(r)}catch(e){o.innerHTML=`<div class="msg bad">${esc(e.message)}</div>`}}
function pgShow(r){const m=r.methods.find(x=>x.available&&x.is_pack_default)||r.methods.find(x=>x.available);
 $('pgres').innerHTML=`<div class="card"><h2>${esc(r.pack)} · ${r.inliers} inliers · photo covers ${r.covered_percent}% of the card</h2>
  ${m?`<a href="${m.image}" target="_blank" rel="noopener"><img src="${m.image}" alt="aligned card" style="max-width:100%;border:1px solid var(--line);border-radius:6px"></a>
  <div class="tag">${m.read} cells read of ${m.with_ink} with ink · <span class="${m.flagged?'warn':''}">${m.flagged} flagged</span> · photo clean-up: ${esc(m.label)}. To compare the clean-up options, use the Clean-up lab.</div>
  ${(m.checks||[]).length?`<div class="tag warn" style="margin-top:6px"><b>Card checks (${m.checks.length})</b><ul style="padding-left:16px;margin:4px 0 0">${m.checks.slice(0,8).map(c=>`<li>${esc(c.field)} visit ${c.visit}: ${esc(c.note)}${c.suggestion?` (suggested ${esc(c.suggestion)})`:''}</li>`).join('')}</ul></div>`:''}`:''}</div>
  <div class="card"><h2>Every cell read (${(r.cells||[]).length} with ink)</h2>
  ${(r.cells||[]).length?`<div class="tag">Amber = below the field's minimum confidence. Confidence is the posterior, the margin or the greedy minimum, as the field says.</div><table class="table table-vcenter"><tr><th>Field</th><th>Visit</th><th>Cell</th><th>Read</th><th>Confidence</th><th>Decode</th></tr>${r.cells.map(c=>`<tr class="${c.flagged?'warn':''}"><td>${esc(c.field)}</td><td>${c.visit}</td><td>${c.image?`<img src="${c.image}" style="height:30px;border:1px solid var(--line)">`:''}</td><td><b>${c.text?esc(c.text):'(blank)'}</b></td><td>${c.confidence}</td><td class="tag">${esc(c.decode)}${c.decode==='grammar'?' · '+esc(c.confidence_kind):''}</td></tr>`).join('')}</table>`:'<div class="tag">No cell had ink.</div>'}</div>`}

