const $ = (id) => document.getElementById(id);
let payload = null;

function money(value) { return Number(value).toLocaleString(undefined, {maximumFractionDigits: 0}); }
function run() {
  $('loading').textContent = 'TRAINING...'; $('run').disabled = true;
  fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
    seed:Number($('seed').value), tables:Number($('tables').value), episodes:Number($('episodes').value)
  })}).then(r => r.json()).then(data => { payload=data; render(data); $('loading').textContent='READY'; $('run').disabled=false; })
    .catch(error => { $('loading').textContent='ERROR'; $('run').disabled=false; alert(error); });
}
function render(data) {
  $('schema-count').textContent = `${data.schema.length} tables`;
  $('schema').innerHTML = data.schema.map(t => `<div class="schema-row"><strong>${t.name}</strong><span>${money(t.rows)} rows <span class="index">${t.indexes.length ? '● indexed' : ''}</span></span></div>`).join('');
  $('joins').innerHTML = data.joins.map(j => `<div class="join-row"><div><code>${j.text}</code><small>${j.selectivity.toPrecision(3)}</small></div></div>`).join('');
  $('sql').textContent = data.sql;
  renderTrace(data.trace, 0); renderResults(data.results); drawCurve(data.history); $('curve-meta').textContent=`${data.history.length} episodes`;
}
function renderTrace(trace, active) { $('trace').innerHTML = trace.map((step,i) => `<div class="trace-step ${i===active?'active':''}"><span class="step-num">0${i+1}</span><div><strong>${step.table}</strong><small>state {${step.state.join(', ') || 'empty'}} · valid [${step.valid.join(', ')}]</small></div><span class="reward">${step.reward > -0.01 ? 'START' : money(step.reward)}</span></div>`).join(''); }
function renderResults(results) { const max=Math.max(...results.map(r=>r.cost)); $('results').innerHTML=results.map((r,i)=>`<div class="result ${i===results.length-1?'best':''}"><span class="rank">0${i+1}</span><h3>${r.name}</h3><div class="cost">${money(r.cost)}</div><div class="ratio">${r.ratio.toFixed(3)}× optimal</div><div class="bar" style="width:${Math.max(8,100*r.cost/max)}%"></div></div>`).join(''); }
function drawCurve(history) { const canvas=$('curve'), ctx=canvas.getContext('2d'), width=canvas.clientWidth||500, height=220, ratio=window.devicePixelRatio||1; canvas.width=width*ratio; canvas.height=height*ratio; ctx.scale(ratio,ratio); ctx.clearRect(0,0,width,height); const values=history.map(h=>h.plan_cost), max=Math.max(...values), min=Math.min(...values), x=i=>i*(width-20)/Math.max(1,values.length-1)+10, y=v=>height-20-(v-min)/Math.max(1,max-min)*(height-40); ctx.strokeStyle='#dbe3dc'; ctx.lineWidth=1; for(let i=0;i<4;i++){let yy=20+i*60;ctx.beginPath();ctx.moveTo(10,yy);ctx.lineTo(width-10,yy);ctx.stroke()} ctx.strokeStyle='#155d56';ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.stroke(); }
$('run').addEventListener('click', run); $('replay').addEventListener('click', () => { if(!payload)return; let i=0; renderTrace(payload.trace,i); const timer=setInterval(()=>{i++; if(i>=payload.trace.length){clearInterval(timer);return} renderTrace(payload.trace,i)},650); }); window.addEventListener('resize',()=>payload&&drawCurve(payload.history)); run();