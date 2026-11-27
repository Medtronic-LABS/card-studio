/* Cards tab: dashboard, templates, scans, grid layout, the upload/orient/preview flow, the field editor, model registry. */
function home(){$('res').innerHTML='';tab((location.hash||'#cards').slice(1))}
async function loadTpls(){const t=await api('/api/templates');TPLNAMES=t.map(x=>x.name);TPLS=t;$('tpls').innerHTML=t.length?'<table class="table table-vcenter"><tr><th>Name</th><th>Version</th><th>Fields</th><th>Saved</th><th></th></tr>'+t.map(x=>`<tr><td>${x.name}</td><td>v${x.latest}</td><td>${x.fields}${x.grid?` <span class="tag">· grid ${x.grid.rows}×${x.grid.cols}</span>`:''}</td><td>${x.created}</td><td><button class="btn" onclick="openT('${x.name}',${x.latest},'view')">View</button> <button class="btn" onclick="gridOpen('${x.name}',${x.latest})">Grid</button> <a class="btn" href="/api/template/${x.name}/${x.latest}/download" download>Download</a> <button class="btn" onclick="openT('${x.name}',${x.latest},'update')">Update</button></td></tr>`).join('')+'</table>':'No templates yet. Drop a photo above to create one.'}

let SCANS=[];
async function loadScans(){try{const s=await api('/api/scans');SCANS=s;const sel=$('scard'),cur=sel.value;
 sel.innerHTML='<option value="auto">Auto-identify</option>'+s.map(x=>`<option value="${esc(x.card)}">${esc(x.card)}</option>`).join('');if([...sel.options].some(o=>o.value===cur))sel.value=cur;
 $('sbody').innerHTML=s.length?'<table class="table table-vcenter"><tr><th>Card</th><th>Scans</th><th>Cells cut</th><th>With ink</th><th>Labeled</th><th></th></tr>'+s.map(x=>x.error?`<tr><td>${esc(x.card)}</td><td colspan="5" class="bad">${esc(x.error)}</td></tr>`:
  `<tr><td>${esc(x.card)} <span class="tag">· ${x.grid.fields} fields × ${x.grid.visits} visits</span></td><td>${x.totals.ok} ok${x.totals.check?`, <b class="warn">${x.totals.check} to check</b>`:''}${x.totals.stale?`, <b class="bad">${x.totals.stale} stale</b>`:''}</td><td>${x.totals.cells}</td><td>${x.totals.with_ink}</td><td>${x.totals.labeled}</td><td><button class="btn" onclick="scanDetail('${esc(x.card)}')">Scans</button> <button class="btn" onclick="scanRecut('${esc(x.card)}')">Re-cut cells${x.needs_recut?` (${x.needs_recut} cut with an older grid)`:''}</button></td></tr>`).join('')+'</table>'
  :'No template has a grid yet. Save a template above, then open Grid and save it.'}catch(e){$('sbody').textContent=e.message}}
async function scanAdd(files){const card=$('scard').value,log=$('slog');let n=0,ok=0,chk=0,rej=[];
 for(const f of files){log.textContent=`Working on ${++n} of ${files.length}: ${f.name} …`;
  try{const r=await api(`/api/scans?card=${encodeURIComponent(card)}&name=${encodeURIComponent(f.name)}&writer=${encodeURIComponent($('swriter').value.trim())}`,{method:'POST',body:f});
   r.results.forEach(x=>{if(x.status==='ok')ok++;else if(x.status==='check')chk++;else rej.push(`${x.source}: ${x.reason}`)})}catch(e){rej.push(`${f.name}: ${e.message}`)}}
 log.innerHTML=`Added ${ok+chk} scan(s)${chk?`, <b class="warn">${chk} to check</b>`:''}${rej.length?`; <b class="bad">${rej.length} not added</b><ul>${rej.slice(0,20).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}`;await loadScans();if($('sdetail').dataset.card)scanDetail($('sdetail').dataset.card)}
function scanDetail(card){const x=SCANS.find(s=>s.card===card),d=$('sdetail');if(!x||x.error){d.innerHTML='';return}d.dataset.card=card;const shots=document.documentElement.classList.contains('shots');
 d.innerHTML=`<article style="margin-top:12px"><b>Scans of ${esc(card)}</b> <span class="tag">template v${x.template_version} · read at ${x.grid.size[0]}×${x.grid.size[1]} px · "check" = matched weakly or the photo cut part of the card off: open it and see whether the grid sits on the printed lines</span> <button class="btn" onclick="$('sdetail').innerHTML='';delete $('sdetail').dataset.card">Close</button>
 ${x.scans.length?'<table class="table table-vcenter"><tr><th>Page</th><th>File</th><th>Writer</th><th>Status</th><th>Matching features</th><th>Cells with ink</th><th>Added</th><th></th></tr>'+x.scans.map(r=>`<tr><td>${esc(r.page)}</td><td class="tag">${esc(r.source)}</td><td><input class="num" maxlength="24" aria-label="writer of ${esc(r.page)}" value="${esc(r.writer||'')}" style="width:90px" onchange="writerSet('${esc(card)}','${esc(r.page)}',this.value)"></td><td><b class="${r.status==='ok'?'ok':r.status==='check'?'warn':'bad'}">${esc(r.status)}</b></td><td>${r.inliers}</td><td>${r.with_ink} of ${r.cells}</td><td class="tag">${esc(r.added)}</td><td>${shots?'':`<button class="btn" onclick="scanView('${esc(card)}','${esc(r.page)}')">View grid</button> `}<button class="btn" onclick="scanRemove('${esc(card)}','${esc(r.page)}')">Remove</button></td></tr>`).join('')+'</table>':'<div class="tag">No scans yet.</div>'}<div id="sview"></div></article>`}
function scanView(card,page){$('sview').innerHTML=`<div class="tag" style="margin:8px 0">${esc(page)}: green = row lines, red = column lines. They should sit on the printed table.</div><img alt="aligned scan with the grid drawn on it" style="max-width:100%;border:1px solid var(--line)" src="/api/card/${encodeURIComponent(card)}/scan/${encodeURIComponent(page)}.png?t=${Date.now()}">`}
async function scanRecut(card){try{const r=await api(`/api/card/${encodeURIComponent(card)}/recut`,{method:'POST',body:'{}'});$('slog').textContent=`${card}: cut ${r.recut} scan(s) again${r.stale?`; ${r.stale} are stale (the template image changed): add them again`:''}`;await loadScans();if($('sdetail').dataset.card)scanDetail($('sdetail').dataset.card)}catch(e){$('slog').textContent=e.message}}
async function writerSet(card,page,w){try{await api(`/api/card/${encodeURIComponent(card)}/writers`,{method:'POST',body:JSON.stringify({writer:w.trim(),pages:[page]})});await loadScans()}catch(e){$('slog').textContent=e.message}}
async function scanRemove(card,page){try{await api(`/api/card/${encodeURIComponent(card)}/scan/delete`,{method:'POST',body:JSON.stringify({page})});await loadScans();scanDetail(card)}catch(e){$('slog').textContent=e.message}}

let GP=null;
function gridOpen(name,ver){GP={name,ver};$('gridpanel').innerHTML=`<article style="margin-top:12px"><b>Reader grid for ${esc(name)} v${ver}</b> <span class="tag">Turns the detected cells into the table of fields and visits a card pack needs. First say which way the card runs: one field per row with a column per visit (ANC, EPI), or one visit per row with a column per field (NCD). Then check the picture: if rows or columns are off, set how many printed label columns and heading rows to skip and how many to keep, and Preview again.</span>
 <div class="row" style="margin-top:8px"><label>Each field is a <select id="ga"><option value="rows">row (visits are columns)</option><option value="columns">column (visits are rows)</option></select></label><label>Label columns to skip <input id="gl" type="number" min="0" max="20" value="0" class="num"></label><label>Columns to keep (blank = all) <input id="gd" type="number" min="1" max="60" class="num"></label><label>Heading rows to skip <input id="glr" type="number" min="0" max="20" value="0" class="num"></label><label>Rows to keep (blank = all) <input id="gdr" type="number" min="1" max="99" class="num"></label>
 <button class="btn" onclick="gridPreview()">Preview</button><button class="btn btn-primary" id="gsv" onclick="gridSave()" disabled>Save grid</button><button class="btn" onclick="$('gridpanel').innerHTML=''">Close</button></div><div id="gout" class="tag" style="margin-top:8px"></div></article>`;
 const o=(TPLS.find(x=>x.name===name&&x.latest===ver)||{}).grid_opts;if(o){$('ga').value=o.axis||'rows';$('gl').value=o.label_cols||0;$('gd').value=o.data_cols||'';$('glr').value=o.label_rows||0;$('gdr').value=o.data_rows||''}   /* start from what was saved, so Save does not silently undo it */
 gridPreview()}
function gridOpts(){return {axis:$('ga').value,label_cols:+($('gl').value||0),data_cols:$('gd').value?+$('gd').value:null,label_rows:+($('glr').value||0),data_rows:$('gdr').value?+$('gdr').value:null}}
function gridQS(){return Object.entries(gridOpts()).filter(([k,v])=>v!==null).map(([k,v])=>`${k}=${encodeURIComponent(v)}`).join('&')}
async function gridPreview(){const o=$('gout');o.textContent='Working…';$('gsv').disabled=true;
 try{const r=await api(`/api/template/${GP.name}/${GP.ver}/grid?${gridQS()}`),g=r.grid,bad=r.errors.length;
  const byCol=g.axis==='columns',nc=g.col_lines_top.length-1;
  o.innerHTML=`<div class="${bad?'bad':'ok'}"><b>${byCol?`${nc} field columns × ${g.rows.length} visit rows`:`${g.rows.length} field rows × ${nc} visit columns`}</b>${bad?' · not valid: '+esc(r.errors.join('; ')):' · valid'}</div>${r.warnings.map(w=>`<div class="warn">${esc(w)}</div>`).join('')}
   <div class="tag">Fields: ${(byCol?g.cols:g.rows).map(esc).join(', ')}. ${byCol?'Fields named col_NN had no renamed heading cell above the visit rows':'Fields named row_NN had no renamed label cell left of the visit columns'}; rename cells in the editor to name them.</div>${drawGrid(g,r.image)}`;
  $('gsv').disabled=!!bad}catch(e){o.innerHTML=`<span class="bad">${esc(e.message)}</span>`}}
async function gridSave(){try{const r=await api('/api/template/grid',{method:'POST',body:JSON.stringify({name:GP.name,version:GP.ver,...gridOpts()})});
 $('gout').insertAdjacentHTML('afterbegin',`<div class="ok"><b>Saved grid.json</b> (${r.rows} rows × ${r.cols} columns)</div>`);loadTpls();loadScans()}catch(e){$('gout').insertAdjacentHTML('afterbegin',`<div class="bad">${esc(e.message)}</div>`)}}
function drawGrid(g,img){const [W,H]=img,xAt=(c,y)=>g.col_lines_top[c]+(g.col_lines_bot[c]-g.col_lines_top[c])*(y-g.y_ref_top)/(g.y_ref_bot-g.y_ref_top),nc=g.col_lines_top.length-1;let p='';
 for(let c=0;c<nc;c++)for(let r=0;r<g.rows.length;r++){const ya=g.row_lines_by_col[c][r],yb=g.row_lines_by_col[c][r+1];
  p+=`<polygon points="${xAt(c,ya)},${ya} ${xAt(c+1,ya)},${ya} ${xAt(c+1,yb)},${yb} ${xAt(c,yb)},${yb}" fill="var(--s1)" fill-opacity=".14" stroke="var(--s1)" stroke-width="${Math.max(1,W/900)}"><title>${g.axis==='columns'?`${esc(g.cols[c])} · visit ${r+1}`:`${esc(g.rows[r])} · visit ${c+1}`}</title></polygon>`}
 return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:900px;margin-top:8px;border:1px solid var(--line);border-radius:8px" role="img" aria-label="Template with detected grid"><image href="/api/template/${GP.name}/${GP.ver}/image" width="${W}" height="${H}"/>${p}</svg>`}
async function openT(name,ver,mode){const L=await api(`/api/template/${name}/${ver}/layout`);S.ctx={mode,name,base:{name,version:ver}};S.fields=L.fields;edit(`/api/template/${name}/${ver}/image`,`${mode==='view'?'View':'Update'} ${name} v${ver}`,mode==='view')}
[['drop','file','identify'],['drop2','file2','new']].forEach(([d,f,mode])=>{const z=$(d);['dragover','dragenter'].forEach(e=>z.addEventListener(e,x=>{x.preventDefault();z.classList.add('over')}));
 ['dragleave','drop'].forEach(e=>z.addEventListener(e,x=>{x.preventDefault();z.classList.remove('over')}));z.addEventListener('drop',e=>up(e.dataTransfer.files[0],mode));$(f).onchange=e=>{up(e.target.files[0],mode);e.target.value=''}});
let ORI={rot:0,str:true,file:null};
async function runAnalyze(){return S.up=await api(`/api/analyze?rotate=${ORI.rot}&straighten=${ORI.str?1:0}`,{method:'POST',body:ORI.file})}
async function up(f,mode){if(!f)return;$('res').innerHTML='<div class="msg">Analyzing locally…</div>';ORI={rot:0,str:true,file:f};
 try{const r=await runAnalyze();if(mode==='new')newPreview(r);else show(r)}catch(e){$('res').innerHTML=`<div class="msg bad">${e.message}</div>`}}
function oriText(o){return o.straightened?`Levelled by ${(-o.tilt_deg).toFixed(1)}° (line agreement ${(o.confidence*100).toFixed(0)}%).`:o.confidence&&o.confidence>=0.3?'Already level.':ORI.str?'No clear ruling lines found, so it was not levelled. Check the picture.':'Auto-levelling is off.'}
function newPreview(r){const o=r.orientation,st=r.layout.stats;
 $('res').innerHTML=`<div class="card"><h2>Is the card the right way up?</h2><div class="tag">${oriText(o)} Rotate by quarter turns if it is sideways or upside down, then continue. The lines alone cannot tell which way is up.</div>
  <div class="row"><button class="btn" onclick="oriTurn(270)" aria-label="rotate 90 degrees counter-clockwise">&#8634; 90&deg;</button><button class="btn" onclick="oriTurn(90)" aria-label="rotate 90 degrees clockwise">&#8635; 90&deg;</button><button class="btn" onclick="oriTurn(180)">180&deg;</button>
   <label style="display:flex;gap:6px;align-items:center"><input type="checkbox" ${ORI.str?'checked':''} onchange="ORI.str=this.checked;oriGo()"> Level automatically</label>
   <button class="btn btn-primary" onclick="createNew()">Continue: ${st.cells} cells in ${st.rows} rows &rarr;</button></div>
  <img src="/api/upload/${r.upload_id}/image" alt="the card as it will be used" style="max-width:100%;max-height:560px;border:1px solid var(--line);border-radius:8px;margin-top:6px"></div>`}
async function oriTurn(d){ORI.rot=(ORI.rot+d)%360;oriGo()}
async function oriGo(){try{newPreview(await runAnalyze())}catch(e){$('res').innerHTML=`<div class="msg bad">${e.message}</div>`}}
function show(r){const m=r.matches,b=m[0],st=r.layout.stats;let h='';
 if(r.verdict==='known')h=`<div class="msg ok">Matches <b>${b.name} v${b.version}</b> (score ${b.score}).</div><div class="row"><button class="btn" onclick="openT('${b.name}',${b.version},'update')">Update ${b.name}</button><button class="btn" onclick="createNew()">Create new instead</button></div>`;
 else if(r.verdict==='similar')h=`<div class="msg warn">Looks like <b>${b.name}</b> (score ${b.score}) but the layout differs. It will not be read with that template.</div><div class="row"><button class="btn" onclick="openT('${b.name}',${b.version},'update')">Update ${b.name}</button><button class="btn btn-primary" onclick="createNew()">Create new instead</button></div>`;
 else h=`<div class="msg warn"><b>NEW TYPE</b>${b?` (best score ${b.score} vs ${b.name})`:' (no templates yet)'}.</div><div class="row"><button class="btn btn-primary" onclick="createNew()">Create new template</button></div>`;
 h+=`<div class="tag">Detected ${st.cells} cells in ${st.rows} rows, ${st.marks} checkbox-like. Printed labels are not read (no local OCR engine installed), so fields are named r1_c1… for you to rename.</div>`;
 if(m.length)h+='<div class="tag">Scores: '+m.map(x=>`${x.name} v${x.version} ${x.score}`).join(' · ')+`  (≥${r.thresholds.high} known, ≥${r.thresholds.mid} similar)</div>`;
 h+=`<div class="card" style="margin-top:12px"><b>What it identified</b> <span class="tag">blue = cell, green = checkbox. Rows are numbered at the left.</span>
  <div class="stage" style="margin-top:8px"><canvas id="pv"></canvas></div><div id="rowsum" class="tag" style="margin-top:8px"></div></div>`;
 $('res').innerHTML=h;preview(r)}
function preview(r){const im=new Image();im.onload=()=>{const cv=$('pv'),s=Math.min(1,1100/im.width);cv.width=im.width*s;cv.height=im.height*s;const c=cv.getContext('2d');c.drawImage(im,0,0,cv.width,cv.height);
 const rows={};r.layout.fields.forEach(f=>{const m=/^r(\d+)_/.exec(f.name);const k=m?+m[1]:0;(rows[k]=rows[k]||{n:0,m:0,y:f.y});rows[k].n++;if(f.kind==='mark')rows[k].m++;rows[k].y=Math.min(rows[k].y,f.y);
  c.lineWidth=2;c.strokeStyle=f.kind==='mark'?'#157f3b':'#2457d6';c.strokeRect(f.x*s,f.y*s,f.w*s,f.h*s)});
 c.font='bold 13px system-ui';Object.entries(rows).forEach(([k,v])=>{c.fillStyle='#e11';c.fillText('r'+k,2,v.y*s+13)});
 $('rowsum').textContent='Cells per row: '+Object.entries(rows).map(([k,v])=>`r${k}: ${v.n}${v.m?` (${v.m} box)`:''}`).join(' · ')};im.src=`/api/upload/${r.upload_id}/image`}
function createNew(){const r=S.up;S.ctx={mode:'new',upload_id:r.upload_id};S.fields=r.layout.fields;edit(`/api/upload/${r.upload_id}/image`,'New template',false)}
function edit(src,title,ro){S.ro=ro;S.sel=null;S.hist=[];$('home').classList.add('hide');$('editor').classList.remove('hide');$('etitle').textContent=title;
 $('approve').style.display=ro?'none':'';$('hint').textContent=ro?'Read-only view.':'Drag to move · corner handle to resize · Delete key removes';
 const im=new Image();im.onload=()=>{S.img=im;const cv=$('cv');S.scale=Math.min(1,1100/im.width);cv.width=im.width*S.scale;cv.height=im.height*S.scale;draw();list();props()};im.src=src}
function tool(t){S.tool=t;$('t-select').classList.toggle('on',t==='select');$('t-add').classList.toggle('on',t==='add')}
function snap(){S.hist.push(JSON.stringify(S.fields));if(S.hist.length>100)S.hist.shift()}
function undo(){if(S.ro||!S.hist.length)return;S.fields=JSON.parse(S.hist.pop());S.sel=null;draw();list();props()}
function del(){if(S.ro||S.sel==null)return;snap();S.fields=S.fields.filter(f=>f.id!==S.sel);S.sel=null;draw();list();props()}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.key==='Delete'||e.key==='Backspace')del();if((e.ctrlKey||e.metaKey)&&e.key==='z')undo()});
const col=k=>k==='mark'?'#157f3b':k==='number'||k==='date'||k==='bp'?'#2457d6':'#b54708';
function draw(){const cv=$('cv'),c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);c.drawImage(S.img,0,0,cv.width,cv.height);
 S.fields.forEach(f=>{const s=S.scale,sel=f.id===S.sel;c.lineWidth=sel?3:1.5;c.strokeStyle=sel?'#e11':col(f.kind);c.fillStyle=sel?'rgba(238,17,17,.12)':'rgba(36,87,214,.05)';
  c.fillRect(f.x*s,f.y*s,f.w*s,f.h*s);c.strokeRect(f.x*s,f.y*s,f.w*s,f.h*s);if(sel){c.fillStyle='#e11';c.fillRect((f.x+f.w)*s-6,(f.y+f.h)*s-6,10,10)}})}
function pt(e){const r=$('cv').getBoundingClientRect();return{x:(e.clientX-r.left)/S.scale,y:(e.clientY-r.top)/S.scale}}
function hit(p){for(let i=S.fields.length-1;i>=0;i--){const f=S.fields[i];if(p.x>=f.x&&p.x<=f.x+f.w&&p.y>=f.y&&p.y<=f.y+f.h)return f}return null}
let drag=null;$('cv').onmousedown=e=>{if(S.ro)return;const p=pt(e);
 if(S.tool==='add'){snap();const f={id:'f'+Date.now(),name:freeFieldName(),kind:'text',section:'',x:p.x,y:p.y,w:1,h:1,range:'',charset:''};S.fields.push(f);S.sel=f.id;drag={m:'new',f,sx:p.x,sy:p.y};return}
 const cur=S.fields.find(f=>f.id===S.sel);
 if(cur&&Math.abs(p.x-(cur.x+cur.w))<10/S.scale&&Math.abs(p.y-(cur.y+cur.h))<10/S.scale){snap();drag={m:'size',f:cur};return}
 const f=hit(p);S.sel=f?f.id:null;if(f){snap();drag={m:'move',f,ox:p.x-f.x,oy:p.y-f.y}}draw();props();list()};
window.onmousemove=e=>{if(!drag)return;const p=pt(e),f=drag.f;
 if(drag.m==='move'){f.x=Math.round(p.x-drag.ox);f.y=Math.round(p.y-drag.oy)}
 else if(drag.m==='size'){f.w=Math.max(6,Math.round(p.x-f.x));f.h=Math.max(6,Math.round(p.y-f.y))}
 else{f.x=Math.round(Math.min(p.x,drag.sx));f.y=Math.round(Math.min(p.y,drag.sy));f.w=Math.round(Math.abs(p.x-drag.sx));f.h=Math.round(Math.abs(p.y-drag.sy))}draw()};
window.onmouseup=()=>{if(!drag)return;if(drag.m==='new'&&(drag.f.w<6||drag.f.h<6)){S.fields.pop();S.sel=null}drag=null;tool('select');draw();props();list()};
function props(){const f=S.fields.find(x=>x.id===S.sel);if(!f){$('props').innerHTML='Click a box.';return}
 const d=S.ro?'disabled':'';$('props').innerHTML=`<label>Name</label><input ${d} value="${f.name}" oninput="setp('name',this.value)">
 <label>Kind</label><select ${d} onchange="setp('kind',this.value)">${KINDS.map(k=>`<option ${k===f.kind?'selected':''}>${k}</option>`).join('')}</select>
 <label>Section</label><input ${d} value="${f.section}" oninput="setp('section',this.value)">
 <label>Range (e.g. 30-150)</label><input ${d} value="${f.range}" oninput="setp('range',this.value)">
 <label>Charset (e.g. 0-9.)</label><input ${d} value="${f.charset}" oninput="setp('charset',this.value)">
 <label>x, y, w, h</label><div class="tag">${[f.x,f.y,f.w,f.h].map(Math.round).join(', ')}</div>`}
function setp(k,v){const f=S.fields.find(x=>x.id===S.sel);f[k]=v;if(k==='kind')draw();list()}
function list(){$('nf').textContent=S.fields.length;$('list').innerHTML=S.fields.map(f=>`<div style="cursor:pointer;padding:2px 4px;${f.id===S.sel?'background:var(--line)':''}" onclick="S.sel='${f.id}';draw();props();list()">${f.name} <span class="tag">${f.kind}</span></div>`).join('')}
function freeFieldName(){const used=new Set(S.fields.map(f=>f.name));let k=S.fields.length+1;while(used.has('field_'+k))k++;return 'field_'+k}
function dedupeNames(fs){const seen=new Set();let n=0;fs.forEach(f=>{const base=f.name;let k=2;while(seen.has(f.name))f.name=base+'_'+k++;if(f.name!==base)n++;seen.add(f.name)});return n}
$('approve').onclick=()=>{const fixed=dedupeNames(S.fields),dup=0;list();
 $('sum').innerHTML=`${S.fields.length} fields, ${S.fields.filter(f=>f.kind==='mark').length} checkbox${fixed?`<br><span class="warn">${fixed} duplicate name${fixed>1?'s were':' was'} renamed (suffix _2, _3). Rename them in the editor if you prefer.</span>`:''}`;
 $('tname').value=S.ctx.name||suggestName();nameHint();$('tname').disabled=S.ctx.mode==='update';$('derr').textContent='';dlg.showModal()};
let TPLNAMES=[],TPLS=[];
const cleanName=v=>v.trim().replace(/\s+/g,'_').replace(/[^A-Za-z0-9_-]/g,'').slice(0,40);
function suggestName(){let n='new_card',i=2;while(TPLNAMES.includes(n))n='new_card_'+i++;return n}
function nameHint(){const raw=$('tname').value,c=cleanName(raw),el=$('nmhint');
 el.innerHTML=!c?'<span class="bad">Give it a name: letters, digits, _ or - (spaces become _).</span>':c!==raw.trim()?`Will be saved as <b>${esc(c)}</b>.`:TPLNAMES.includes(c)&&S.ctx.mode!=='update'?`<span class="warn">${esc(c)} already exists: saving makes a new version of it.</span>`:'Letters, digits, _ and - only.'}
async function save(){try{const name=cleanName($('tname').value);if(!name){nameHint();$('derr').textContent='Please enter a template name.';return}
 const body={name,fields:S.fields};if(S.ctx.mode==='update')body.base=S.ctx.base;else body.upload_id=S.ctx.upload_id;
 const r=await api('/api/save',{method:'POST',body:JSON.stringify(body)});dlg.close();home();alert(`Saved ${r.name} v${r.version}`)}catch(e){$('derr').textContent=e.message}}

let REG={sel:{}};
function when(ts){const d=Math.floor((Date.now()/1000-ts)/86400);return d<=0?'today':d===1?'yesterday':d<30?d+' days ago':new Date(ts*1000).toISOString().slice(0,10)}
async function loadPacks(){const [packs,tpls,runs,lab]=await Promise.all([api('/api/packs'),api('/api/templates'),api('/api/runs'),api('/api/labels/summary')]);
 let ml=null;try{ml=await api('/api/ml')}catch(e){}
 const ok=packs.filter(x=>!x.error),byId={};ok.forEach(x=>(byId[x.id]=byId[x.id]||[]).push(x));
 const details={};await Promise.all(ok.map(async x=>{try{details[x.file]=await api('/api/pack/'+encodeURIComponent(x.file)+'/detail')}catch(e){}}));
 const ids=Object.keys(byId);
 $('packs').innerHTML=ids.length?ids.map(id=>{const vs=byId[id].sort((a,b)=>b.version-a.version),cur=vs.find(v=>v.file===REG.sel[id])||vs[0],d=details[cur.file],m=d&&d.manifest,rd=m&&m.reader,pp=m&&m.preprocess,nz=pp&&pp.normalize,ho=cur.heldout;
  const mlFile=ml?(ok.find(x=>ml.pack===x.id+' v'+x.version)||{}).file:null,sameReader=mlFile&&details[mlFile]&&m&&details[mlFile].manifest.files['reader.onnx']===m.files['reader.onnx'],mine=ml&&(ml.pack===id+' v'+cur.version||sameReader)?ml:null;
  return `<div class="reg"><h3>${esc(m?m.name:id)} <code>${esc(id)}</code> <span class="badge ${cur.ship?'bg-green-lt':'bg-secondary-lt'}">${cur.ship} of ${cur.fields} fields ship</span></h3>
   <div class="vers">${vs.map(v=>`<button class="ver ${v.file===cur.file?'on':''}" onclick="REG.sel['${esc(id)}']='${esc(v.file)}';loadPacks()" title="${v.ship} ship, ${v.review} need review${v.heldout?'':', no held-out evidence'}">v${v.version}<span class="badge ${v.ship?'bg-green-lt':'bg-secondary-lt'}">${v.ship}</span></button>`).join('')}<span class="tag" style="align-self:center">${vs.length} version${vs.length>1?'s':''}, newest first</span></div>
   <div class="tags"><span>CRNN + CTC</span>${rd?`<span>${rd.input.height}×${rd.input.width} gray input</span><span>${esc(rd.decode)}</span>`:''}<span>${ho?'held-out evidence':'no held-out evidence'}</span><span>photo clean-up: ${esc(nz?nz.default:'flatten')}</span><span>created ${esc(cur.created)}</span></div>
   ${mine?`<div class="tag" style="margin-top:8px">Held-out metrics from run ${esc(mine.run)}${ml.pack!==id+' v'+cur.version?` (${esc(ml.pack)}: this version has the identical reader)`:''}</div><div class="mx"><div><b>${f0(mine.overall.exact)}</b><span>held-out exact</span></div><div><b>${(mine.overall.cer*100).toFixed(1)}%</b><span>char error rate</span></div><div><b>${mine.model&&mine.model.params?(mine.model.params/1000).toFixed(0)+'k':'-'}</b><span>parameters</span></div><div><b>${mine.model?(mine.model.onnx_bytes/1048576).toFixed(1)+' MB':'-'}</b><span>ONNX size</span></div><div><b>${mine.model?mine.model.latency_ms_median+' ms':'-'}</b><span>per cell, CPU</span></div></div>`:`<div class="tag" style="margin:10px 0">${ho?'Metrics for this version are under Model quality only for the latest run.':'This version was never scored on pages it did not see, so every field is needs-review.'}</div>`}
   <div class="row"><button class="btn" onclick="openPack('${esc(cur.file)}')">View</button><a class="btn" href="/api/pack/${esc(cur.file)}/download" download>Download</a><button class="btn btn-primary" onclick="tab('train')">Train &amp; ship</button><button class="btn" onclick="tab('ml')">Model quality</button></div></div>`}).join('')+(packs.filter(x=>x.error).map(x=>`<div class="msg bad">${esc(x.file)}: unreadable (${esc(x.error)})</div>`).join('')):'No packs yet. Open Train &amp; ship to build one.';
 // badges
 const last=runs.filter(r=>r.exact!=null)[0];
 $('badges').innerHTML=[`<span class="badge bg-green-lt">local only</span>`,`<span class="badge bg-secondary-lt">${ids.length} card type${ids.length===1?'':'s'}</span>`,`<span class="badge bg-secondary-lt">${ok.length} pack version${ok.length===1?'':'s'}</span>`,last?`<span class="badge bg-blue-lt">last run ${esc(last.name)} · ${f0(last.exact)} held-out</span>`:'<span class="badge bg-secondary-lt">no training run yet</span>'].join('');
 // getting started, computed from real state
 let best=null,detail=null;try{best=ok.sort((a,b)=>b.ship-a.ship||b.version-a.version)[0];detail=best&&details[best.file]}catch(e){}
 const H=(detail&&detail.heldout)||{},minN=((detail&&detail.manifest.metrics.heldout&&detail.manifest.metrics.heldout.gate)||{}).min_n||20,gaps=detail?detail.fields.filter(f=>!H[f.key]||H[f.key].n<minN).length:null;
 let cds=[],scs=[];try{[cds,scs]=await Promise.all([api('/api/cards'),api('/api/scans')])}catch(e){}
 const grids=cds.filter(c=>c.grid).length,nscan=scs.reduce((a,s)=>a+((s.totals||{}).scans||0),0),ready=cds.filter(c=>c.trainable),lp=Math.max(0,...cds.map(c=>c.labels.pages)),tfl=Object.values(details).filter(d=>d.manifest.reader.tflite).length;
 const imp=(lab.imported&&lab.imported.total)||0,steps=[
  [tpls.length>0,'Save a card template',`${tpls.length} saved`,['Add one',"goIntake('drop2')"]],
  [grids>0,'Save a grid for a template',`${grids} template${grids===1?'':'s'} with a grid`,['Layouts',"goIntake('tpls')"]],
  [nscan>0,'Add filled scans',`${nscan} scan${nscan===1?'':'s'} cut into cells`,['Filled scans',"goIntake('sbody')"]],
  [ready.length>0,'Label cells on 10 pages of a card',`most on one card: ${lp} page${lp===1?'':'s'}`,['Label',"tab('label')"]],
  [runs.some(r=>r.state==='done'),'Train a reader',`${runs.length} run${runs.length===1?'':'s'}`,['Train & ship',"tab('train')"]],
  [!!best&&best.ship>0,'Get a field to ship',best?`${best.ship} of ${best.fields} fields (${esc(best.id)} v${best.version})`:'no pack yet',['Train & ship',"tab('train')"]],
  [gaps===0,'Give every field enough held-out cells',gaps==null?'':`${gaps} field${gaps===1?'':'s'} still under ${minN} cells`,['Label',"tab('label')"]],
  [tfl>0,'Have a pack with a TensorFlow Lite reader',`${tfl} pack version${tfl===1?'':'s'}`,['Model registry',"document.getElementById('packs').scrollIntoView({behavior:'smooth'})"]],
  [runs.filter(r=>r.state==='done').length>=2,'Compare two runs',`${runs.filter(r=>r.state==='done').length} finished run(s)`,['History',"tab('history')"]],
  [imp>0,'Import phone corrections',imp?`${imp} imported`:'none yet',['Label',"tab('label')"]]];
 const done=steps.filter(x=>x[0]).length;$('chkn').textContent=`${done} of ${steps.length}`;$('chkb').style.width=(100*done/steps.length)+'%';
 $('chk').innerHTML=steps.map(([d,t,sub,a])=>`<li class="${d?'done':''}"><i>${d?'&#10003;':''}</i><div><b>${esc(t)}</b><div class="tag">${sub}${d?'':` · <a onclick="${a[1]}">${a[0]} &rarr;</a>`}</div></div></li>`).join('');
 // activity feed from real timestamps
 const ev=[];runs.forEach(r=>ev.push({ts:r.ts,kind:'run',t:`Training run <b>${esc(r.name)}</b> ${esc(r.state)}`,sub:r.exact!=null?`${f0(r.exact)} held-out exact · ${r.shipped}/${r.fields} fields ship · ${r.n_cells} cells`:esc(r.detail||'')}));
 ok.forEach(x=>{const t=Date.parse(x.created+'T00:00:00')/1000;if(t)ev.push({ts:t,kind:'pack',t:`Pack <b>${esc(x.id)} v${x.version}</b> created`,sub:`${x.ship} of ${x.fields} fields ship`})});
 tpls.forEach(x=>{const t=Date.parse((x.created||'').replace(' ','T'))/1000;if(t)ev.push({ts:t,kind:'tpl',t:`Template <b>${esc(x.name)}</b> saved (v${x.latest})`,sub:`${x.fields} fields${x.grid?`, grid ${x.grid.rows}×${x.grid.cols}`:''}`})});
 ev.sort((a,b)=>b.ts-a.ts);
 $('feed').innerHTML=ev.length?ev.slice(0,8).map(e=>`<li class="${e.kind}">${e.t}<div class="tag">${e.sub} · ${when(e.ts)}</div></li>`).join(''):'<li class="tag">Nothing yet.</li>'}
async function loadStatus(){const [t,l]=await Promise.all([api('/api/train/status'),api('/api/labels/summary')]);
 const tr=t.state==='none'?'no training run yet':t.state==='done'&&t.results?`last run ${t.results.minutes} min · pooled held-out exact ${(t.results.cv_all.exact*100).toFixed(0)}% · pack ${esc(t.results.pack.id)} v${t.results.pack.version} ${t.results.pack.validation_errors.length?'INVALID':'valid'}`:`${t.state}: ${esc(t.detail||'')}`;
 $('status').innerHTML=`<details class="st"><summary><b>Training:</b> ${tr}</summary><div class="tag" style="padding:6px 0">Details and the per-field gate table are in Train &amp; Ship. <a href="#train" onclick="tab('train')">Open</a></div></details>
 <details class="st"><summary><b>Labels:</b> ${l.cells} cells on ${l.pages} pages${l.new?` · ${l.new} new this session`:''}</summary><div class="tag" style="padding:6px 0">Merged from labels.csv and labels_new.csv (new wins). <a href="#label" onclick="tab('label')">Open Label</a></div></details>`}
let JTIMER=null;const JSTATE={running:['bg-blue-lt','running'],done:['bg-green-lt','finished'],error:['bg-red-lt','failed'],stopped:['bg-orange-lt','stopped']};
function jobTime(s){if(s==null)return'';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h} h ${m} min`:m?`${m} min`:`${s} s`}
async function loadJobs(){clearTimeout(JTIMER);try{const js=await api('/api/jobs'),busy=js.some(j=>j.state==='running');
 $('jobs').innerHTML=js.length?js.map(j=>{const pr=j.progress,pct=pr?Math.round(100*pr.done/pr.total):(j.state==='done'?100:null),st=JSTATE[j.state]||JSTATE.stopped;
  return `<details class="st" ${j.state==='running'?'open':''}><summary><b>${esc(j.title)}</b> <span class="badge ${st[0]}">${st[1]}</span> <span class="tag">${j.state==='running'?`${esc(j.stage)} · ${pr?`${pr.done} of ${pr.total} parts done · `:''}running for ${esc(j.process_time||'')}`:esc(j.detail)}</span></summary>
   <div class="tag" style="margin:6px 0">${esc(j.question)}</div>
   ${pct!==null?`<div style="height:8px;background:var(--line);border-radius:5px;overflow:hidden;max-width:520px"><div style="height:100%;width:${pct}%;background:var(--acc)"></div></div>`:j.state==='running'?'<div class="tag">no fixed number of parts: see the line above</div>':''}
   ${j.state==='running'?`<div class="tag" style="margin-top:4px">Now: ${esc(j.detail)}. This line changes when a part finishes, so it can stay the same for many minutes while a model trains. The running time counts wall-clock time, including any time the Mac slept.</div>`:j.state==='stopped'?`<div class="tag warn" style="margin-top:4px">Not running and not finished: ${esc(j.detail||'it ended early')}. Start it again with <code>${esc(j.command)}</code>; parts already finished are kept.</div>`:j.state==='error'?`<div class="tag bad" style="margin-top:4px">${esc((j.detail||'').slice(-300))}</div>`:''}
   ${j.rows.length?`<table class="table table-vcenter" style="margin-top:8px"><tr>${j.columns.map(c=>`<th>${esc(c)}</th>`).join('')}</tr>${j.rows.map(r=>`<tr>${r.map((c,i)=>`<td${i===0?'':' style="font-size:12px"'}>${esc(c)}</td>`).join('')}</tr>`).join('')}</table>`:'<div class="tag" style="margin-top:6px">No results yet.</div>'}
   ${j.notes.map(n=>`<div class="tag">${esc(n)}</div>`).join('')}
   <div class="tag">Started with <code>${esc(j.command)}</code> · progress file <code>${esc(j.status_file)}</code></div></details>`}).join(''):'No experiment has run on this machine yet. They are started from the command line, for example <code>python studio/capacity.py run</code>.';
 JTIMER=setTimeout(loadJobs,busy?4000:30000)}catch(e){$('jobs').innerHTML=`<span class="bad">${esc(e.message)}</span>`;JTIMER=setTimeout(loadJobs,30000)}}
function loadIdSum(){api('/api/idtest').then(d=>{const b=(d.headline||[]).find(x=>x.method==='baseline'),g=(d.headline||[]).find(x=>x.method.startsWith('geometry'));$('idsum').textContent=b?`Simulated phone photos, 223 images. Today's matcher: ${b.anc_ok[0]}/${b.anc_ok[1]} ANC accepted, ${b.other_as_anc[0]}/${b.other_as_anc[1]} non-ANC wrongly accepted. With outline check: ${g.other_as_anc[0]}/${g.other_as_anc[1]}.`:'Not run yet (python idtest.py build, score, report).'}).catch(()=>{$('idsum').textContent='unavailable'})}
function loadStats(){loadDash();Promise.all([api('/api/packs'),api('/api/labels/summary'),api('/api/idtest'),api('/api/train/status')]).then(([p,l,i,t])=>{
 const best=p.filter(x=>!x.error).sort((a,b)=>b.ship-a.ship||b.version-a.version)[0],hb=(i.headline||[]).find(x=>x.method==='baseline'),cv=t.state==='done'&&t.results?t.results.cv_all.exact:null;
 const c=(k,v,sub,cls,go)=>`<div class="card stat${go?' link':''}" ${go?`onclick="${go}" tabindex="0" role="link" onkeydown="if(event.key==='Enter')${go}"`:''}><div class="subheader">${k}</div><div class="h1 ${cls||''}">${v}</div><div class="tag">${sub}</div></div>`;
 $('stats').innerHTML=c('Card types',p.length,p.map(x=>esc(x.id)+' v'+x.version).join(', ')||'none yet','','document.getElementById(\'packs\').scrollIntoView({behavior:\'smooth\'})')+
  c('Fields safe to ship',best?`${best.ship}<span class="tag" style="font-size:14px"> / ${best.fields}</span>`:'-',best?`${esc(best.id)} v${best.version}, target met on held-out pages`:'build a pack first',best&&best.ship?'text-green':'',"tab('train')")+
  c('Labeled cells',l.cells+(l.imported&&l.imported.total?` <span class="tag" style="font-size:14px">+${l.imported.total}</span>`:''),`${l.pages} pages${l.imported&&l.imported.total?' · +imported':''}`,'',"tab('label')")+
  c('Held-out exact match',cv==null?'-':(cv*100).toFixed(0)+'%',cv==null?'no run yet':'pooled 5-fold CV, by page','',"tab('history')")+
  c('Template ID test',hb?`${hb.anc_ok[0]}/${hb.anc_ok[1]}`:'-',hb?`ANC accepted; ${hb.other_as_anc[0]}/${hb.other_as_anc[1]} non-ANC wrongly accepted (simulated photos)`:'not run','','window.open(\'/idtest\')')}).catch(()=>{})}
function refreshDash(){loadStats();loadPacks().catch(e=>{$('packs').innerHTML=`<div class="msg bad">${esc(e.message)}</div>`})}
function refreshIntake(){loadIdSum();loadTpls().catch(()=>{});loadScans()}
function goIntake(id){tab('intake');setTimeout(()=>{const e=document.getElementById(id);if(e)e.scrollIntoView({behavior:'smooth'})},250)}

let PV=null;
async function packTflite(file){const b=$('pvtf'),o=$('pvtfmsg');b.disabled=true;o.textContent='Converting and checking against the ONNX reader…';
 try{const r=await api(`/api/pack/${file}/tflite`,{method:'POST',body:'{}'});if(r.errors.length){o.innerHTML=`<b class="bad">New pack failed validation: ${esc(r.errors.join('; '))}</b>`;return}openPack(r.file)}catch(e){o.innerHTML=`<b class="bad">${esc(e.message)}</b>`;b.disabled=false}}
async function openPack(file){try{const d=await api(`/api/pack/${file}/detail`);PV={d,sel:null};
 $('home').classList.add('hide');$('editor').classList.add('hide');$('pview').classList.remove('hide');
 const m=d.manifest,ok=!d.errors.length,sh=d.fields.filter(f=>f.status==='ship').length;
 $('pvtitle').textContent=`${m.id} v${m.version}`;$('pvsub').textContent=`${m.name} · saved ${m.created}`;
 $('pvmsg').innerHTML=`<div class="msg ${ok?'':'bad'}">${ok?'Pack is valid (hashes, grid, fields and reader checked)':'Pack failed validation: '+d.errors.map(esc).join('; ')} · ${sh} of ${d.fields.length} fields ship</div>`;
 const held=m.metrics.heldout;
 const tf=m.reader.tflite;
 $('pvinfo').innerHTML=`<div>Reader: ${m.reader.input.height}×${m.reader.input.width} → ${esc(m.reader.decode)}</div>
  <div>TensorFlow Lite: ${tf?`reader.tflite included, same weights (largest difference from the ONNX reader ${tf.parity.max_abs_diff.toExponential(1)}) · <a href="/api/pack/${esc(d.file)}/reader.tflite" download>download reader.tflite</a>`:`not in this pack · <button class="btn" id="pvtf" onclick="packTflite('${esc(d.file)}')">Make a version with a TFLite reader</button>`} <span id="pvtfmsg" class="tag"></span></div><div>Match floor: ${m.match.min_inliers} inliers (${esc(m.match.method)})</div><div>Photo clean-up default: ${esc((m.preprocess.normalize||{default:'flatten'}).default)}${m.preprocess.normalize&&m.preprocess.normalize.reference?` (reference mean ${m.preprocess.normalize.reference.mean}, std ${m.preprocess.normalize.reference.std})`:(m.preprocess.normalize?'':' (pack does not say: flatten)')}</div><div>Template: ${m.template.width}×${m.template.height}</div>
  <div style="margin-top:6px">${held?`<b>Held-out evidence:</b> ${esc(held.method||'')}<br>${esc(held.reader_trained_on||'')}`:'<b>No held-out evidence.</b> Every field is needs-review.'}</div>
  <div style="margin-top:6px" class="tag">File hashes:<br>${Object.entries(m.files).map(([k,v])=>`${esc(k)} ${v.slice(0,10)}…`).join('<br>')}</div>`;
 const im=new Image();im.onload=()=>{PV.img=im;const cv=$('pvc');PV.scale=Math.min(1,1000/im.width);cv.width=im.width*PV.scale;cv.height=im.height*PV.scale;drawPack()};im.src=`/api/pack/${file}/template`;
 pvTable()}catch(e){alert(e.message)}}
function pvCells(){const g=PV.d.grid,gs=PV.d.manifest.template.grid_scale,nc=g.col_lines_top.length-1,out=[];
 const xAt=(t,b,y)=>t+(b-t)*(y-g.y_ref_top)/(g.y_ref_bot-g.y_ref_top);
 for(let c=0;c<nc;c++){const rl=g.row_lines_by_col[c];g.rows.forEach((key,ri)=>{const ym=(rl[ri]+rl[ri+1])/2,xa=xAt(g.col_lines_top[c],g.col_lines_bot[c],ym),xb=xAt(g.col_lines_top[c+1],g.col_lines_bot[c+1],ym);
  out.push({key,c,x:xa/gs,y:rl[ri]/gs,w:(xb-xa)/gs,h:(rl[ri+1]-rl[ri])/gs})})}return out}
function drawPack(){if(!PV||!PV.img)return;const cv=$('pvc'),c=cv.getContext('2d'),s=PV.scale;c.clearRect(0,0,cv.width,cv.height);c.drawImage(PV.img,0,0,cv.width,cv.height);
 const st={};PV.d.fields.forEach(f=>st[f.key]=f.status);
 pvCells().forEach(r=>{const ship=st[r.key]==='ship',sel=PV.sel===r.key;c.lineWidth=sel?3:1;c.strokeStyle=ship?'#157f3b':'#b54708';c.fillStyle=sel?(ship?'rgba(21,127,59,.28)':'rgba(181,71,8,.28)'):(ship?'rgba(21,127,59,.10)':'rgba(181,71,8,.07)');
  c.fillRect(r.x*s,r.y*s,r.w*s,r.h*s);c.strokeRect(r.x*s,r.y*s,r.w*s,r.h*s)})}
function pvTable(){const d=PV.d,pct=v=>(v*100).toFixed(0);
 $('pvtable').innerHTML='<table class="table table-vcenter"><tr><th>Field</th><th>Kind</th><th>Rule</th><th>Min conf</th><th>Held-out</th><th>Decision</th></tr>'+d.fields.map(f=>{const h=d.heldout[f.key],rule=f.kind==='number'&&f.range?`${f.range[0]}–${f.range[1]}`:f.kind==='mark'?esc((f.valid_values||[]).join(' ')):f.kind==='bp'&&f.bp?`sys ${f.bp.sys.join('–')}, dia ${f.bp.dia.join('–')}${f.bp.sys_above_dia?', sys>dia':''}`:f.kind==='date'?'D/M/YY, real day and month':'-';
  return `<tr style="cursor:pointer" onclick="PV.sel='${f.key}';drawPack()"><td>${esc(f.key)}</td><td>${f.kind}</td><td>${rule}</td><td>${f.min_confidence==null?'-':f.min_confidence}</td><td>${h?`${h.correct}/${h.n} (${pct(h.ci95[0])}–${pct(h.ci95[1])}%)`:'<span class="tag">none</span>'}</td><td><b class="${f.status==='ship'?'ok':'warn'}">${f.status==='ship'?'SHIP':'needs-review'}</b><div class="tag">${esc(f.status_reason)}</div></td></tr>`}).join('')+'</table>'}
async function poll(){try{const t=await api('/api/training');const box=$('train');if(t.state==='none'){box.style.display='none';return}box.style.display='';
 const pct=t.state==='done'?100:t.epoch?Math.min(100,Math.round(100*t.epoch/t.max_epochs)):0,eta=t.eta_sec_max?` · up to ${Math.ceil(t.eta_sec_max/60)} min left`:'';
 const col=t.state==='done'?'ok':t.state==='running'?'warn':'bad';
 let h=`<div class="row"><b class="${col}">${t.state.toUpperCase()}</b>${t.epoch?`<span class="tag">${t.state==='done'?`stopped at epoch ${t.epoch} of up to ${t.max_epochs} (early stop)`:`epoch ${t.epoch}/${t.max_epochs} (early stop may end sooner)`}${eta} · ${t.sec_per_epoch}s/epoch</span>`:''}</div>
 <div style="height:10px;background:var(--line);border-radius:6px;overflow:hidden"><div style="height:100%;width:${pct}%;background:var(--acc)"></div></div>`;
 if(t.epoch)h+=`<div class="tag" style="margin-top:6px">loss ${t.loss} · validation exact ${(t.val_exact*100).toFixed(1)}% · validation CER ${t.val_cer}</div>`;
 if(t.results){const r=t.results,p=x=>x==null?'-':(x*100).toFixed(0)+'%';
  h+=`<div class="tag" style="margin-top:10px">Held-out TEST: ${r.n_test} cells. Every field has n&lt;10: direction only, no per-field conclusion. CURRENT model column is training-writer data, not accuracy.</div>
  <table class="table table-vcenter"><tr><th>Field</th><th>n</th><th>NEW held-out</th><th>95% CI</th><th>CER</th><th>CV pooled (n)</th><th>CURRENT (trained on)</th></tr>`+
  r.fields.map(f=>`<tr><td>${f.field}</td><td>${f.n}</td><td>${p(f.exact)}</td><td>${p(f.lo)}–${p(f.hi)}</td><td>${f.cer.toFixed(2)}</td><td>${p(f.cv_exact)} (${f.cv_n??'-'})</td><td>${p(f.cur_exact)}</td></tr>`).join('')+'</table>'}
 h+='<div class="charts" id="charts"></div>';$('tbody').innerHTML=h;renderCharts(t);if(t.state==='running')setTimeout(poll,3000);else if(t.state==='done')setTimeout(poll,30000)}catch(e){setTimeout(poll,10000)}}
async function loadDash(){try{const [p,l,runs]=await Promise.all([api('/api/packs'),api('/api/labels/summary'),api('/api/runs')]);
 const best=p.filter(x=>!x.error).sort((a,b)=>b.ship-a.ship||b.version-a.version)[0];if(!best){$('attn').innerHTML='<li class="tag">No packs yet.</li>';$('rd').innerHTML='';return}
 const d=await api('/api/pack/'+encodeURIComponent(best.file)+'/detail');renderReadiness(d,best);renderAttention(d,best,l,runs);renderTrend(runs)}catch(e){$('attn').innerHTML=`<li class="bad">${esc(e.message)}</li>`}}
function renderReadiness(d,best){const g=d.manifest.metrics.heldout&&d.manifest.metrics.heldout.gate,target=g?g.target_lower_bound_pct/100:0.85,H=d.heldout||{};
 $('rdsub').textContent=`${best.id} v${best.version} · held-out, 95% intervals · dashed line = ship target ${(target*100).toFixed(0)}%`;
 const rows=d.fields.map(f=>({f,h:H[f.key]})).sort((a,b)=>(b.f.status==='ship')-(a.f.status==='ship')||((b.h&&b.h.n)||0)-((a.h&&a.h.n)||0));
 $('rd').innerHTML=rows.map(({f,h})=>{const badge=f.status==='ship'?'<span class="badge bg-green-lt">ship</span>':'<span class="badge bg-orange-lt">needs-review</span>';
  if(!h)return `<div class="nm" title="${esc(f.key)}">${esc(f.key)}</div><div class="tag">no labeled cells yet</div><div class="nn">-</div>${badge}`;
  const ex=h.correct/h.n,[lo,hi]=h.ci95;return `<div class="nm" title="${esc(f.status_reason||'')}">${esc(f.key)}</div><div class="bar" title="${h.correct}/${h.n} correct, 95% interval ${(lo*100).toFixed(0)}-${(hi*100).toFixed(0)}%"><i style="left:${lo*100}%;width:${(hi-lo)*100}%"></i><u style="left:${ex*100}%"></u><s style="left:${target*100}%"></s></div><div class="nn">${h.correct}/${h.n}</div>${badge}`}).join('')}
function renderAttention(d,best,l,runs){const g=d.manifest.metrics.heldout&&d.manifest.metrics.heldout.gate,minN=g?g.min_n:20,target=g?g.target_lower_bound_pct/100:0.85,H=d.heldout||{},items=[];
 const none=d.fields.filter(f=>!H[f.key]).map(f=>f.key),few=d.fields.filter(f=>H[f.key]&&H[f.key].n<minN&&H[f.key].ci95[1]<target),stuck=d.fields.filter(f=>H[f.key]&&H[f.key].n>=minN&&H[f.key].ci95[1]<target);
 if(stuck.length)items.push(['warn',`${stuck.length} field${stuck.length>1?'s are':' is'} credibly below the target`,`${stuck.map(f=>`${esc(f.key)} (${H[f.key].correct}/${H[f.key].n})`).join(', ')}: with ${minN}+ held-out cells, the whole 95% interval is under ${(target*100).toFixed(0)}%.${stuck.some(f=>H[f.key].correct/H[f.key].n<0.1)?' A near-zero score usually means the crops or labels are wrong for that field, so check those before labeling more.':' More training labels may help, but check the crops and the reader too.'}`,null]);
 const needLabels=d.fields.filter(f=>H[f.key]&&H[f.key].n<minN&&f.status!=='ship');
 if(needLabels.length)items.push(['warn',`${needLabels.length} field${needLabels.length>1?'s':''} have fewer than ${minN} held-out cells`,`${needLabels.slice(0,6).map(f=>`${esc(f.key)} (${H[f.key].n})`).join(', ')}${needLabels.length>6?', …':''}. They cannot ship until there are enough cells to judge.`,['Label cells',"tab('label')"]]);
 if(none.length)items.push(['info',`${none.length} field${none.length>1?'s have':' has'} no labeled cells`,esc(none.join(', ')),['Label cells',"tab('label')"]]);
 const last=runs[0],total=l.cells+((l.imported&&l.imported.total)||0);
 if(last&&last.n_cells!=null&&total>last.n_cells)items.push(['info',`${total-last.n_cells} labels added since the last training run`,`Run ${esc(last.name)} used ${last.n_cells} cells; there are ${total} now.`,['Train & ship',"tab('train')"]]);
 if(!last)items.push(['info','No training run yet','Build the first pack from the labeled cells.',['Train & ship',"tab('train')"]]);
 if(!(d.manifest.preprocess&&d.manifest.preprocess.normalize&&d.manifest.preprocess.normalize.reference))items.push(['info',`${esc(best.id)} v${best.version} has no photo-contrast reference`,'Photos in poor light read best with the contrast-matching clean-up, which needs a pack that carries one.',null]);
 if(!items.length)items.push(['ok','Nothing needs attention','Every field with labels meets the target.',null]);
 $('attn').innerHTML=items.map(([sev,t,b,a])=>`<li><span class="dot ${sev}"></span><div><b>${t}</b><span class="tag">${b}</span>${a?` <a onclick="${a[1]}">${a[0]} &rarr;</a>`:''}</div></li>`).join('')}
function renderTrend(runs){const done=runs.filter(r=>r.exact!=null).sort((a,b)=>a.ts-b.ts);
 if(done.length<2){$('trend').innerHTML=`<div class="tag" style="margin-top:12px">${done.length?'One training run so far':'No finished run yet'}: a trend appears once there are two. <a style="cursor:pointer;text-decoration:underline" onclick="tab('history')">History</a> compares runs.</div>`;return}
 const W=620,H=140,L=40,R=12,T=14,B=26,X=i=>L+(W-L-R)*(done.length===1?.5:i/(done.length-1)),Y=v=>T+(H-T-B)*(1-v);let g='';
 for(const v of [0,.25,.5,.75,1])g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L-5}" y="${Y(v)+4}" text-anchor="end">${v*100}%</text>`;
 g+=`<polyline fill="none" stroke="var(--s1)" stroke-width="2" points="${done.map((r,i)=>`${X(i)},${Y(r.exact)}`).join(' ')}"/>`;
 done.forEach((r,i)=>{g+=`<line x1="${X(i)}" x2="${X(i)}" y1="${Y(r.lo)}" y2="${Y(r.hi)}" stroke="var(--s1)" stroke-width="2" opacity=".5"/><circle cx="${X(i)}" cy="${Y(r.exact)}" r="4.5" fill="var(--s1)" stroke="var(--card)" stroke-width="2"><title>${esc(r.name)}: ${(r.exact*100).toFixed(0)}% [${(r.lo*100).toFixed(0)}-${(r.hi*100).toFixed(0)}], ${r.shipped}/${r.fields} shipped, ${r.n_cells} cells</title></circle>`;if(done.length<=8)g+=`<text x="${X(i)}" y="${H-8}" text-anchor="middle">${esc(r.name.replace(/^run-?/,'#'))}</text>`});
 $('trend').innerHTML=`<div class="tag" style="margin-top:14px"><b>Held-out exact match by run</b> (pooled CV, 95% interval)</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Held-out exact match by training run">${g}</svg>`}

