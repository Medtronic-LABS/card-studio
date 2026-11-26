/* Review tab: approve, reject or flag a labeled cell before it can train on (studio/review.py). A rejected cell never trains; a
pending or flagged one is held out until a person decides. */
let RV={card:null,show:'needs_attention',cells:[]};
async function rvInit(){try{const cs=await api('/api/cards'),sel=$('rvcard'),cur=sel.value;
 sel.innerHTML=cs.filter(c=>!c.error).map(c=>`<option value="${esc(c.id)}">${esc(c.id)}</option>`).join('');
 sel.value=[...sel.options].some(o=>o.value===cur)?cur:(RV.card&&[...sel.options].some(o=>o.value===RV.card)?RV.card:(cs[0]||{}).id||'');
 RV.card=sel.value;rvLoad()}catch(e){$('rvlist').innerHTML=`<span class="bad">${esc(e.message)}</span>`}}
function rvCard(v){RV.card=v;rvLoad()}
function rvShowSet(v){RV.show=v;rvLoad()}
async function rvLoad(){if(!RV.card)return;const el=$('rvlist');el.innerHTML='<div class="tag">Loading…</div>';
 try{const r=await api(`/api/card/${encodeURIComponent(RV.card)}/review?show=${RV.show}`);RV.cells=r.cells;rvRender()}catch(e){el.innerHTML=`<span class="bad">${esc(e.message)}</span>`}}
function rvRender(){const el=$('rvlist');$('rvcount').textContent=`${RV.cells.length} cell${RV.cells.length===1?'':'s'}`;
 if(!RV.cells.length){el.innerHTML='<div class="tag">Nothing here for this filter.</div>';return}
 el.innerHTML=RV.cells.map((c,i)=>`<div style="display:flex;gap:10px;align-items:center;padding:8px 0;border-top:1px solid var(--line)">
  <img alt="the labeled cell" style="height:56px;max-width:160px;object-fit:contain;border:1px solid var(--line);background:#fff" src="/api/card/${encodeURIComponent(RV.card)}/crop/${encodeURIComponent(c.field)}/${encodeURIComponent(c.col)}/${encodeURIComponent(c.page)}.png">
  <div style="flex:1 1 auto;min-width:0"><b>${esc(c.label)||'<i>blank</i>'}</b> <span class="tag">${esc(c.field)} &middot; page ${esc(c.page)} &middot; ${esc(c.col)}</span>${c.reason?`<div class="tag warn">${esc(c.reason)}</div>`:''}</div>
  <span class="badge ${c.status==='approved'?'bg-green-lt':c.status==='rejected'?'bg-red-lt':c.status==='flagged'?'bg-orange-lt':'bg-secondary-lt'}">${esc(c.status)}</span>
  <button class="btn" onclick="rvSet(${i},'approved')">Approve</button><button class="btn" onclick="rvSet(${i},'rejected')">Reject</button><button class="btn" onclick="rvSet(${i},'flagged')">Flag</button>
  </div>`).join('')}
async function rvSet(i,status){const c=RV.cells[i];try{await api(`/api/card/${encodeURIComponent(RV.card)}/review/set`,{method:'POST',body:JSON.stringify({page:c.page,field:c.field,col:c.col,status,reviewer:($('rvreviewer').value||'').trim()})});rvLoad()}catch(e){alert(e.message)}}
async function rvMigrate(){if(!RV.card)return;try{const r=await api(`/api/card/${encodeURIComponent(RV.card)}/review/migrate`,{method:'POST',body:'{}'});alert(r.skipped||`Seeded: ${r.approved} approved, ${r.pending} pending`);rvLoad()}catch(e){alert(e.message)}}
