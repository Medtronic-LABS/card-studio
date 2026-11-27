/* Shared helpers: fetch wrapper, theme, tab navigation, generic chart/diagram primitives used by more than one tab. */
const GH0=location.hash;
const $=id=>document.getElementById(id);let S={fields:[],sel:null,img:null,scale:1,tool:'select',hist:[],ctx:null};
const KINDS=['text','number','date','bp','mark'];
async function api(p,o){o=o||{};o.headers={...(o.headers||{}),'X-Card-Studio':'1'};const r=await fetch(p,o);const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);return j}
const NS='http://www.w3.org/2000/svg',esc=x=>String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function lineChart(pts,key,title,note,{max=null,best=null,fmt=v=>v.toFixed(2),color='var(--s1)'}={}){
 const W=320,H=190,L=40,R=10,T=10,B=28,xs=pts.map(p=>p.epoch),xm=Math.max(...xs,1),vals=pts.map(p=>p[key]);let ym=max??(Math.max(...vals)*1.1||1);
 const X=e=>L+(W-L-R)*e/xm,Y=v=>T+(H-T-B)*(1-v/ym);let g='';
 for(let i=0;i<=4;i++){const v=ym*i/4;g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="1"/><text x="${L-5}" y="${Y(v)+4}" text-anchor="end">${fmt(v)}</text>`}
 for(let i=0;i<=4;i++){const e=Math.round(xm*i/4);g+=`<text x="${X(e)}" y="${H-10}" text-anchor="middle">${e}</text>`}
 g+=`<text x="${(L+W-R)/2}" y="${H-0}" text-anchor="middle">epoch</text>`;
 if(best&&best<=xm)g+=`<line x1="${X(best)}" x2="${X(best)}" y1="${T}" y2="${H-B}" stroke="var(--mut)" stroke-dasharray="4 3"/><text x="${X(best)+4}" y="${T+10}">best ep ${best}</text>`;
 g+=`<path d="${pts.map((p,i)=>(i?'L':'M')+X(p.epoch)+' '+Y(p[key])).join('')}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
 if(pts.length<=60)pts.forEach(p=>g+=`<circle cx="${X(p.epoch)}" cy="${Y(p[key])}" r="4" fill="${color}" stroke="var(--card)" stroke-width="2"><title>epoch ${p.epoch}: ${fmt(p[key])}</title></circle>`);
 return `<div class="ch"><b>${title}</b><div class="tag">${note}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${g}</svg></div>`}
function dotPlot(rows){
 const W=640,rh=34,L=170,R=20,T=24,H=T+rows.length*rh+26,X=v=>L+(W-L-R)*v;let g='';
 for(let i=0;i<=4;i++){const v=i/4;g+=`<line x1="${X(v)}" x2="${X(v)}" y1="${T-6}" y2="${H-24}" stroke="var(--line)"/><text x="${X(v)}" y="${H-8}" text-anchor="middle">${v*100}%</text>`}
 rows.forEach((r,i)=>{const y=T+i*rh+rh/2;g+=`<text x="${L-8}" y="${y+4}" text-anchor="end" style="fill:var(--ink)">${esc(r.field)}</text><text x="${L-8}" y="${y+16}" text-anchor="end">test n=${r.n}, cv n=${r.cv_n??'-'}</text>`;
  const mk=(v,lo,hi,dy,c,hollow,lab)=>{if(v==null)return'';let o=lo!=null?`<line x1="${X(lo)}" x2="${X(hi)}" y1="${y+dy}" y2="${y+dy}" stroke="${c}" stroke-width="2" opacity=".6"/>`:'';
   return o+`<circle cx="${X(v)}" cy="${y+dy}" r="4.5" fill="${hollow?'var(--card)':c}" stroke="${c}" stroke-width="2"><title>${esc(r.field)} ${lab}: ${(v*100).toFixed(0)}%${lo!=null?` [95% CI ${(lo*100).toFixed(0)}-${(hi*100).toFixed(0)}]`:''}</title></circle>`};
  g+=mk(r.exact,r.lo,r.hi,-8,'var(--s1)',false,'NEW held-out TEST')+mk(r.cv_exact,r.cv_lo,r.cv_hi,2,'var(--s2)',false,'NEW pooled CV')+mk(r.cur_exact,null,null,12,'var(--s3)',true,'CURRENT (trained on these)')});
 return `<div class="ch wide"><b>Per-field exact match, with 95% Wilson intervals</b><div class="tag">Wide intervals are the finding: most fields have n&lt;10 on test. Hollow = current model, which trained on these cells (not accuracy).</div>
 <div class="lg"><span><i style="background:var(--s1)"></i>NEW, held-out TEST</span><span><i style="background:var(--s2)"></i>NEW, pooled 5-fold CV</span><span><i style="border:2px solid var(--s3)"></i>CURRENT (trained on)</span></div>
 <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Per-field exact match with confidence intervals">${g}</svg></div>`}
function confHist(bins,cut){
 const W=640,H=210,L=44,R=10,T=14,B=34,mx=Math.max(...bins.map(b=>Math.max(b.correct,b.wrong)),1),Y=v=>T+(H-T-B)*(1-v/mx),bw=(W-L-R)/10;let g='';
 for(let i=0;i<=4;i++){const v=Math.round(mx*i/4);g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L-5}" y="${Y(v)+4}" text-anchor="end">${v}</text>`}
 bins.forEach((b,i)=>{const x=L+i*bw,w=(bw-8)/2;g+=`<text x="${x+bw/2}" y="${H-16}" text-anchor="middle">${b.lo.toFixed(1)}</text>`;
  [['correct','var(--s1)',0],['wrong','var(--s2)',1]].forEach(([k,c,j])=>{const h=(H-B)-Y(b[k]);if(b[k])g+=`<path d="M${x+4+j*(w+2)} ${H-B}v${-Math.max(h,2)+4}q0 -4 4 -4h${w-8}q4 0 4 4v${Math.max(h,2)-4}z" fill="${c}"><title>conf ${b.lo.toFixed(1)}-${(b.lo+.1).toFixed(1)}: ${b[k]} ${k}</title></path>`})});
 [[cut.marks,'marks 0.5'],[cut.numbers,'numbers 0.7']].forEach(([c,l],i)=>g+=`<line x1="${L+bw*10*c}" x2="${L+bw*10*c}" y1="${T}" y2="${H-B}" stroke="var(--mut)" stroke-dasharray="4 3"/><text x="${L+bw*10*c+4}" y="${T+10+i*12}">${l}</text>`);
 g+=`<text x="${(L+W-R)/2}" y="${H-2}" text-anchor="middle">confidence (lowest per-character probability)</text>`;
 return `<div class="ch wide"><b>Confidence of held-out reads (pooled CV): correct vs wrong</b><div class="tag">Reads below the dashed cutoff go to a person. Wrong reads sitting right of the line are the ones the flag misses.</div>
 <div class="lg"><span><i style="background:var(--s1)"></i>correct</span><span><i style="background:var(--s2)"></i>wrong</span></div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Confidence histogram, correct vs wrong">${g}</svg></div>`}
function renderCharts(t){const el=$('charts');if(!el)return;let h='';const hs=t.history||[],best=t.results?.best_epoch;
 if(hs.length>1){const tag=t.history_full?'every epoch':'sampled every 25 epochs while running';
  h+=lineChart(hs,'loss','Training loss (CTC)',tag,{best});
  h+=lineChart(hs,'val_exact','Validation exact match',tag+'. Val = 7 pages, so noisy.',{max:1,best,fmt:v=>(v*100).toFixed(0)+'%',color:'var(--s1)'});
  h+=lineChart(hs,'val_cer','Validation CER (lower is better)',tag,{max:1,best,color:'var(--s2)'})}
 if(t.results){h+=dotPlot(t.results.fields.slice().sort((a,b)=>(b.cv_n||0)-(a.cv_n||0)));if(t.results.conf_bins)h+=confHist(t.results.conf_bins,t.results.cutoffs);
  const f=t.results.flags,row=(n,s)=>`<tr><td>${n}</td><td>${s.n}</td><td>${s.flagged} (${(s.flag_rate*100).toFixed(0)}%)</td><td>${s.wrong_caught}/${s.wrong}</td><td>${s.unflagged_correct}/${s.unflagged} (${(s.unflagged_acc*100).toFixed(0)}%)</td></tr>`;
  h+=`<div class="ch wide"><b>Flag rules (plausibility + confidence)</b><table class="table table-vcenter"><tr><th>Set</th><th>Cells</th><th>Flagged</th><th>Wrong reads caught</th><th>Unflagged accuracy</th></tr>${row('NEW, TEST',f.new_test.all)}${row('NEW, pooled CV',f.cv.all)}${row('CURRENT (trained on)',f.cur_all.all)}</table></div>`}
 el.innerHTML=h}

function glyph(c,h,w){const bw=Math.max(10,w/192*84),bh=Math.max(8,h/48*40),d=Math.min(5,Math.max(1,Math.round(c/32))),W=100,H=56;let g='';
 for(let i=d-1;i>=0;i--){g+=`<rect x="${4+i*4}" y="${H-bh-4-i*3}" width="${bw}" height="${bh}" rx="2" fill="var(--s1)" opacity="${.25+.15*(d-i)}" stroke="var(--s1)"/>`}
 return `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">${g}</svg>`}
function renderArch(){
 const N=[ // title, [c,h,w] or null, shape text, params, note, cls
 ['Cell crop',[1,34,150],'any size, gray',null,'from aligned card; app also removes ruled lines + flattens light',''],
 ['Resize + pad',[1,48,192],'1 × 48 × 192',0,'height to 48, keep aspect, pad white, scale to [-1, 1]',''],
 ['Conv block 1',[32,24,96],'32 × 24 × 96',384,'3×3 conv, BN, ReLU, pool 2×2',''],
 ['Conv block 2',[64,12,48],'64 × 12 × 48',18624,'same, pool 2×2',''],
 ['Conv block 3',[128,6,48],'128 × 6 × 48',74112,'same, pool 2×1 (keeps width)',''],
 ['Conv block 4',[128,1,48],'128 × 1 × 48',147840,'pool 6×1 collapses height; width stays 48',''],
 ['Frames',[1,8,48],'48 steps × 128',0,'each step sees about 4 px of the 192 px width',''],
 ['BiLSTM 128×2',[1,8,60],'48 × 256',264192,'left-to-right and right-to-left context','hl'],
 ['Linear',[1,8,70],'48 × 129',33153,'one score per class per step (incl. blank)','hl'],
 ['Field mask',[1,8,70],'allowed classes only',0,'e.g. weight_kg keeps 0-9 . - and blank; the rest set to -inf','mk'],
 ['Softmax + CTC greedy',null,'"58"',0,'argmax per step, merge repeats, drop blanks','mk'],
 ['Confidence',null,'min char prob 0.84',0,'lowest per-character probability at its emitting step','mk'],
 ['Plausibility',null,'OK / review',0,'conf ≥ 0.7 (marks 0.5), clinical range, BP order, date validity','ok']];
 const box=(n)=>`<div class="nd ${n[5]}"><b>${n[0]}</b>${n[1]?glyph(...n[1]):''}<div class="sh">${n[2]}</div>${n[3]!=null?`<div class="pm">${n[3]?n[3].toLocaleString()+' params':'no params'}</div>`:''}<div class="nt">${n[4]}</div></div>`;
 const ar='<div class="ar"><i></i></div>';
 const T=[['NumtaDB digits','55k real Bangla digit images, stitched into 4,000 fake numbers, dates and BP strings'],
  ['Pretrain','15 epochs, lr 1e-3, CTC loss, field-masked'],['Labeled cells','split by PAGE: train / val / test, never mixed'],
  ['Fine-tune','Adam, batch 64, field-balanced sampler, light augmentation (rotate, shift, brightness, blur)'],
  ['Early stop','pick epoch by validation exact match, then CER (patience 80, max 300)'],['Test once','held-out pages scored a single time, with 95% Wilson intervals'],['Export','ONNX, and the same weights as TensorFlow Lite, checked against each other; run on the phone']];
 $('abody').innerHTML=`<div class="fl"><b>Inference: one cell, left to right</b> (dots show data moving)</div><div class="flow">${N.map(box).join(ar)}</div>
 <div class="tag" style="margin-top:8px">Total 538,305 parameters (about 2.2 MB fp32). CNN turns the image into 48 vertical slices; the BiLSTM reads the slices as a sequence; CTC lets the model emit a character without knowing where it sits.</div>
 <div class="fl"><b>Training: how the data is used</b></div><div class="flow">${T.map(t=>`<div class="nd"><b>${t[0]}</b><div class="nt">${t[1]}</div></div>`).join(ar)}</div>`}

const TABS=[['cards','Cards'],['label','Label'],['review','Review'],['datasets','Datasets'],['train','Train & Ship'],['arch','Architecture'],['play','Playground'],['history','History'],['ml','Model quality'],['lab','Clean-up lab'],['how','How to use']];
function tab(t){clearTimeout(JTIMER);if(!TABS.some(x=>x[0]===t))t='cards';$('editor').classList.add('hide');$('pview').classList.add('hide');$('home').classList.remove('hide');
 TABS.forEach(([k])=>$('v-'+k).classList.toggle('hide',k!==t));
 const g=NAVGROUPS.find(g=>g.tabs.includes(t));if(g&&document.getElementById('navgrp-'+g.label)&&document.getElementById('navgrp-'+g.label).classList.contains('closed'))navToggle(g.label);
 [...$('tabs').querySelectorAll('.nav-link')].forEach(b=>b.classList.toggle('on',b.dataset.tab===t));$('pg-sub').textContent=TITLES[t][0];$('pg-title').textContent=TITLES[t][1];
 try{history.replaceState(null,'','#'+t)}catch(e){}if(t==='cards')refreshCards();if(t==='label'){loadLabel();loadLabeling()}if(t==='review')rvInit();if(t==='datasets')dsetLoad();if(t==='train'){extLoad();tsLoad();evLoad()}if(t==='arch')tsLoad();if(t==='history'){loadHistory();loadExperiments();loadJobs()};if(t==='lab')labInit();if(t==='play')pgInit();if(t==='ml'){loadMLHome();loadRunsTable()}}
const ICON={cards: `<path d="M3 5h18v14H3z"/><path d="M3 10h18M8 5v14"/>`, label: `<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>`, review: `<path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/>`, train: `<path d="M4 19V9m6 10V5m6 14v-7m6 7H2"/>`, arch: `<rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="9" y="14" width="6" height="6" rx="1"/><path d="M7 10v2a2 2 0 002 2h1m4-4h1a2 2 0 012 2v0"/>`, play: `<path d="M7 4l13 8-13 8z"/>`, datasets: `<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>`, how: `<circle cx="12" cy="12" r="9"/><path d="M12 8v.01M11 12h1v4h1"/>`};const TITLES={cards:['Home','Dashboard'],label:['Data','Label cells'],review:['Data','Review labels'],datasets:['Data','Datasets'],train:['Reader','Train & ship'],arch:['Reader','Architecture'],history:['Runs','History & compare'],ml:['Evaluation','Model quality'],play:['Try it','Playground'],lab:['Try it','Photo clean-up lab'],how:['Guide','How to use']};
const NAVGROUPS=[{label:'Data',tabs:['cards','label','review','datasets']},{label:'Model',tabs:['train','arch','history','ml']},{label:'Tools',tabs:['play','lab']}];
const GROUPED_TABS=new Set(NAVGROUPS.flatMap(g=>g.tabs));
function navLink(k,n){const b=document.createElement('a');b.className='nav-link';b.href='#'+k;b.dataset.tab=k;b.innerHTML=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICON[k]}</svg><span>${n}</span>`;b.onclick=e=>{e.preventDefault();tab(k)};return b}
function navToggle(label){const h=document.querySelector(`.navgrp[data-g="${label}"]`),items=document.getElementById('navgrp-'+label);h.classList.toggle('open');items.classList.toggle('closed')}
function navBuild(active){$('tabs').innerHTML='';const tabsByKey=Object.fromEntries(TABS);
 NAVGROUPS.forEach(g=>{const open=g.tabs.includes(active);
  const h=document.createElement('div');h.className='navgrp'+(open?' open':'');h.dataset.g=g.label;h.onclick=()=>navToggle(g.label);
  h.innerHTML=`<svg class="chev" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M9 6l6 6-6 6"/></svg>${g.label}`;$('tabs').appendChild(h);
  const items=document.createElement('div');items.className='navgrp-items'+(open?'':' closed');items.id='navgrp-'+g.label;
  g.tabs.forEach(k=>items.appendChild(navLink(k,tabsByKey[k])));$('tabs').appendChild(items)});
 TABS.filter(([k])=>!GROUPED_TABS.has(k)).forEach(([k,n])=>$('tabs').appendChild(navLink(k,n)))}
navBuild((location.hash||'#cards').slice(1));
function themeUI(){const m=themeGet('cs-mode','system'),a=themeGet('cs-accent','blue');
 $('seg').innerHTML=[['system','System'],['light','Light'],['dark','Dark']].map(([k,n])=>`<button type="button" aria-pressed="${m===k}" onclick="themeSet('cs-mode','${k}');applyTheme();themeUI()">${n}</button>`).join('');
 $('swatches').innerHTML=Object.entries(ACCENTS).map(([k,c])=>`<button type="button" title="${k}" aria-label="${k}" aria-pressed="${a===k}" style="background:${c}" onclick="themeSet('cs-accent','${k}');applyTheme();themeUI()"></button>`).join('')}

const pct1=v=>v==null?'-':(v*100).toFixed(0)+'%',pp=v=>v==null?'-':(v>=0?'+':'')+(v*100).toFixed(0)+' pp';
