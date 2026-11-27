/* Datasets tab: every card type in one table (studio/datasets.py). Read-only; a row opens that card in Train or Review. */
const DSLVL={ok:['bg-green-lt','ready'],info:['bg-secondary-lt','note'],warn:['bg-orange-lt','check'],bad:['bg-red-lt','fix']};
function dsetGo(card,t){GCARD=card;if(t==='review')RV.card=card;tab(t)}
async function dsetLoad(){const el=$('dsetbody');if(!el)return;
 try{const r=await api('/api/datasets'),n=v=>v==null?'-':Number(v).toLocaleString();
  el.className='';el.innerHTML=`<table class="table table-vcenter"><tr><th>Card</th><th>Labels</th><th>Pages</th><th>Writers</th><th>Split by</th><th>Test set</th><th>Pack</th><th>Data checks</th><th></th></tr>`+
  r.cards.map(c=>{const L=DSLVL[c.level]||DSLVL.bad,why=c.error?esc(c.error):c.failing.map(f=>esc(f.id)).join(', ');
   return `<tr><td><b>${esc(c.id)}</b></td><td>${n(c.labels)}</td><td>${n(c.pages)}</td><td>${n(c.writers)}</td><td>${esc(c.split_unit||'-')}</td><td>${c.test_pages?c.test_pages+' pages':'none'}</td><td>${c.pack_version?'v'+c.pack_version:'-'}</td>
    <td><span class="badge ${L[0]}" title="${esc(c.failing.map(f=>f.message).join('\n')||c.blocker||'')}">${L[1]}</span> <span class="tag">${why}</span></td>
    <td><button class="btn" onclick="dsetGo('${esc(c.id)}','train')">Train</button> <button class="btn" onclick="dsetGo('${esc(c.id)}','review')">Review</button></td></tr>`}).join('')+
  `<tr><td><b>${r.totals.cards} card${r.totals.cards===1?'':'s'}</b></td><td><b>${n(r.totals.labels)}</b></td><td><b>${n(r.totals.pages)}</b></td><td><b>${n(r.totals.writers)}</b></td><td></td><td></td><td></td><td class="tag">${Object.entries(r.totals.levels).filter(([k,v])=>v).map(([k,v])=>`${v} ${DSLVL[k][1]}`).join(' · ')}</td><td></td></tr></table>`}
 catch(e){el.innerHTML=`<span class="bad">${esc(e.message)}</span>`}}
