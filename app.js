(function () {
/* Matched Deep-fUS explorer. No telemetry, external runtime or browser inference. */
'use strict';
const D = window.DIIANNI_DATA;
const $ = id => document.getElementById(id);
const state = {id:'fr11', k:50, band:2, tolerance:20, mask:false, row:47, col:47};
const cache = new Map();
let currentImage = null;
let requestSequence = 0;
const pct = (x, digits=1) => x == null || !Number.isFinite(x) ? 'not defined' : `${x > 0 ? '+' : ''}${x.toFixed(digits)}%`;
const unsigned = (x, digits=2) => x == null || !Number.isFinite(x) ? 'not defined' : `${x!==0 && Math.abs(x)<10**(-digits) ? x.toPrecision(2) : x.toFixed(digits)}%`;
const getRow = (id=state.id,k=state.k) => D.rows.find(r=>r.id===id && r.k===Number(k));
const band = r => r.bands[state.band];
const eligible = r => Number.isFinite(band(r).signed_bias_percent) && Math.abs(band(r).signed_bias_percent)>state.tolerance;
function mean(xs) { return xs.reduce((a,b)=>a+b,0)/xs.length; }
function median(xs) { const v=xs.filter(Number.isFinite).sort((a,b)=>a-b); return (v[(v.length-1)>>1]+v[v.length>>1])/2; }
function initialize() {
  if (!D || D.rows.length!==200) throw new Error('Complete analysis data were not loaded.');
  D.sample_ids.forEach(id=>{ const o=document.createElement('option');o.value=id;o.textContent=id;$('sample-select').append(o); });
  $('sample-select').value=state.id; $('frames-select').value=String(state.k);
  [['fr11',50,'Small total bias, local disagreement'],['fr36',50,'Larger absolute discrepancy'],['fr1',125,'Closer agreement'],['fr36',25,'Check non-monotonic bias']].forEach(([id,k,label])=>{
    const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',()=>select(id,k));$('suggestions').append(b);
  });
  $('sample-select').addEventListener('change',e=>select(e.target.value,state.k));
  $('frames-select').addEventListener('change',e=>select(state.id,Number(e.target.value)));
  $('band-select').addEventListener('change',e=>{state.band=Number(e.target.value);render();});
  $('mask-toggle').addEventListener('change',e=>{state.mask=e.target.checked;drawImages();});
  $('tolerance').addEventListener('input',e=>{state.tolerance=Number(e.target.value);renderCohort();});
  for(const [input,key] of [['pixel-row','row'],['pixel-col','col']]) $(input).addEventListener('change',e=>{
    state[key]=Math.max(0,Math.min(95,Math.trunc(Number(e.target.value)||1)-1));e.target.value=state[key]+1;pixelReadout();
  });
  for(const id of ['reference-image','prediction-image','difference-image']) $(id).addEventListener('click',e=>{
    const box=e.currentTarget.getBoundingClientRect();state.col=Math.min(95,Math.floor((e.clientX-box.left)/box.width*96));state.row=Math.min(95,Math.floor((e.clientY-box.top)/box.height*96));
    $('pixel-row').value=state.row+1;$('pixel-col').value=state.col+1;$('pixel-row').closest('details').open=true;pixelReadout();
  });
  $('export-review').addEventListener('click',downloadReview);
  renderReproduction();select(state.id,state.k);
}
async function select(id,k) {
  state.id=id;state.k=Number(k);$('sample-select').value=id;$('frames-select').value=String(k);
  render(); const sequence=++requestSequence;
  currentImage=null;for(const name of ['reference-image','prediction-image','difference-image']) $(name).getContext('2d').clearRect(0,0,96,96);
  $('pixel-readout').textContent='Loading image data…';
  try {
    if(!cache.has(id)) { const response=await fetch(`samples/${id}.json`);if(!response.ok) throw new Error(`HTTP ${response.status}`);cache.set(id,await response.json()); }
    if(sequence!==requestSequence)return;
    currentImage=cache.get(id);
    if(currentImage.reference.length!==9216 || currentImage.predictions[String(k)].length!==9216)throw new Error('Unexpected image dimensions');
    drawImages();pixelReadout();
  } catch(error) { if(sequence!==requestSequence)return;$('pixel-readout').textContent='Image could not load. Try reloading, or use the source files in the repository.';console.error(error); }
}
function render() {
  const r=getRow(),b=band(r);
  $('metric-one').textContent=r.repository_ssim.toFixed(3);
  $('metric-two').textContent=pct(b.signed_bias_percent);
  $('metric-three').textContent=unsigned(b.power_share_percent,b.power_share_percent<.1?3:2);
  const faint=b.power_share_percent<1 ? 'This band holds less than 1% of reference power: its relative bias is especially denominator-sensitive. ' : '';
  $('readout').textContent=`${r.id}, ${r.k} real frames: whole-image total-power bias is ${pct(r.whole_bias,2)}, but the selected band’s signed bias is ${pct(b.signed_bias_percent)}. Its absolute error contributes ${unsigned(b.absolute_error_percent_total_reference,2)} of total reference power. ${faint}These are intensity-defined pixels, not validated vessels or neural states.`;
  const same=D.rows.filter(x=>x.id===state.id);
  $('selected-budget-note').textContent=`Same band, same pair, all budgets: ${same.sort((a,b)=>b.k-a.k).map(x=>`${x.k} frames ${pct(band(x).signed_bias_percent)}`).join(' · ')}. ${r.negative_count} negative predicted pixels (of 9,216) are retained in these metrics.`;
  renderCohort();if(currentImage && currentImage.id===state.id) { drawImages();pixelReadout(); }
}
function isInBand(value,peak) {
  const db=value>0?10*Math.log10(value/peak):-Infinity;
  const lo=D.bands[state.band].lo,hi=D.bands[state.band].hi;
  return (lo===null||db>=lo)&&(state.band===4?db<=hi:db<hi);
}
function drawImages() {
  if(!currentImage)return;
  const ref=currentImage.reference,pred=currentImage.predictions[String(state.k)],peak=Math.max(...ref);
  const names=['reference-image','prediction-image','difference-image'];
  names.forEach((name,kind)=>{
    const ctx=$(name).getContext('2d'),im=ctx.createImageData(96,96);
    for(let i=0;i<ref.length;i++) {
      let color;
      if(kind<2) { const v=kind===0?ref[i]:pred[i];const db=10*Math.log10(Math.max(v/peak,1e-4));const g=Math.round(Math.max(0,Math.min(1,(db+40)/40))*255);color=[g,g,g]; }
      else { const difference=(pred[i]-ref[i])/peak;const t=Math.min(1,Math.abs(difference)/.05),end=difference<0?[31,122,140]:[192,122,42];color=end.map(c=>Math.round(248+(c-248)*t)); }
      if(state.mask&&!isInBand(ref[i],peak))color=[228,228,228];
      im.data.set([...color,255],i*4);
    }
    ctx.putImageData(im,0,0);
  });
}
function pixelReadout() {
  if(!currentImage)return;
  const i=state.row*96+state.col,ref=currentImage.reference[i],pred=currentImage.predictions[String(state.k)][i];
  $('pixel-readout').textContent=`Row ${state.row+1}, column ${state.col+1}: reference ${ref.toPrecision(5)}, reconstruction ${pred.toPrecision(5)}, difference ${(pred-ref).toPrecision(5)} (native arbitrary units). ${isInBand(ref,Math.max(...currentImage.reference))?'Inside':'Outside'} the selected reference band. Display arrays are rounded to seven significant digits.`;
}
function renderCohort() {
  $('tolerance-value').textContent=`${state.tolerance}%`;
  const counts=D.budgets.map(k=>({k,n:D.rows.filter(r=>r.k===k&&eligible(r)).length}));
  const selected=counts.find(x=>x.k===state.k);
  $('cohort-summary').textContent=`${selected.n} / 40 images exceed ±${state.tolerance}% signed band bias at ${state.k} real frames. Change the band or tolerance to test how stable the shortlist is.`;
  const svg=$('cohort-chart'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
  function node(tag,attrs,text) { const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!==undefined)e.textContent=text;svg.append(e);return e; }
  node('title',{},'Images exceeding the descriptive band-bias tolerance');
  [0,10,20,30,40].forEach(n=>{const x=100+n*10;node('line',{x1:x,x2:x,y1:20,y2:205,stroke:'#ddd'});node('text',{x,y:225,'text-anchor':'middle','font-size':11,fill:'#666','font-family':'system-ui'},String(n));});
  counts.forEach((v,i)=>{const y=30+i*36;node('text',{x:88,y:y+13,'text-anchor':'end','font-size':12,'font-family':'system-ui',fill:'#444'},`${v.k} frames`);node('rect',{x:100,y,width:v.n*10,height:19,fill:v.k===state.k?'#1f7a8c':'#a8b5b7'});node('text',{x:108+v.n*10,y:y+13,'font-size':11,'font-family':'system-ui',fill:'#444'},`${v.n}/40`);});
  svg.setAttribute('aria-label',counts.map(v=>`${v.k} frames: ${v.n} of 40 images exceed ${state.tolerance}% band bias`).join('; '));
  const rows=D.rows.filter(r=>r.k===state.k&&eligible(r)).sort((a,b)=>band(b).absolute_error_percent_total_reference-band(a).absolute_error_percent_total_reference);
  $('review-rows').replaceChildren();
  rows.slice(0,5).forEach(r=>{const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');button.type='button';button.textContent=r.id;button.addEventListener('click',()=>{select(r.id,r.k);$('a1').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});});td.append(button);tr.append(td);[pct(band(r).signed_bias_percent),unsigned(band(r).absolute_error_percent_total_reference),r.repository_ssim.toFixed(3)].forEach(text=>{const c=document.createElement('td');c.textContent=text;tr.append(c);});$('review-rows').append(tr);});
  if(rows.length===0) { const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='No images exceed this chosen tolerance. This is not evidence of physiological validity.';tr.append(td);$('review-rows').append(tr); }
  const chosen=D.rows.filter(r=>r.k===state.k),med=median(chosen.map(r=>band(r).signed_bias_percent));
  $('main-finding').textContent=`At this setting, the median signed bias in the selected band is ${pct(med)}. The table prioritizes absolute error contribution, so a faint band with a dramatic relative percentage does not automatically dominate. Signed bias need not change monotonically with frame count; inspect all five settings before deciding what to repeat.`;
}
function renderReproduction() {
  D.budgets.forEach(k=>{const rows=D.rows.filter(r=>r.k===k),tr=document.createElement('tr');[k,`${rows[0].compression}%`,mean(rows.map(r=>r.repository_ssim)).toFixed(4),mean(rows.map(r=>r.nmse)).toFixed(4)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});$('reproduction-rows').append(tr);});
}
function reviewCSV() {
  const header=['sample_id','real_frames','data_reduction_percent','band_low_db','band_high_db','band_pixels','reference_power_share_percent','signed_band_bias_percent','band_absolute_error_percent_total_reference','repository_ssim','whole_image_bias_percent','negative_predictions','tolerance_percent','flagged_for_review'];
  const rows=D.rows.map(r=>{const b=band(r);return [r.id,r.k,r.compression,b.lo===null?'-inf':b.lo,b.hi,b.pixels,b.power_share_percent,b.signed_bias_percent,b.absolute_error_percent_total_reference,r.repository_ssim,r.whole_bias,r.negative_count,state.tolerance,eligible(r)];});
  return [header,...rows].map(row=>row.map(v=>v==null?'':String(v)).join(',')).join('\n')+'\n';
}
function downloadReview() {
  const blob=new Blob([reviewCSV()],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`deep-fus-review-band-${state.band}-tolerance-${state.tolerance}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
try { initialize(); } catch(error) { $('readout').textContent='The analysis could not be loaded. Reload the page or use the repository’s downloadable metrics.';console.error(error); }
})();
