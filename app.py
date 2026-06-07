"""
app.py – Offside AI | Football Goal Predictor (Streamlit)
=========================================================
Inference uses catboost_model.cbm which expects exactly 103 features
built by the same pipeline as improve_and_submit.py.
"""

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Offside AI | Football Goal Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ─── FONTS ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ─── DESIGN TOKENS ──────────────────────────────── */
:root {
  --green:        #00ff88;
  --green-dim:    #00cc6a;
  --green-glow:   rgba(0,255,136,0.18);
  --blue:         #0ea5e9;
  --purple:       #a855f7;
  --bg-dark:      #060a0f;
  --bg-card:      rgba(255,255,255,0.03);
  --bg-card-h:    rgba(255,255,255,0.06);
  --border:       rgba(255,255,255,0.08);
  --border-green: rgba(0,255,136,0.22);
  --text-1:       #f0f6fc;
  --text-2:       #8b949e;
  --text-3:       #484f58;
}

/* ─── RESET STREAMLIT CHROME ─────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: var(--bg-dark) !important;
  color: var(--text-1) !important;
  cursor: none !important;
}
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; }

/* Streamlit root wrappers */
.stApp, .main, .block-container {
  background: transparent !important;
}
.block-container {
  padding-top: 1rem !important;
  max-width: 1200px !important;
}
/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(6,10,15,0.92) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * { color: var(--text-1) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03) !important;
  border-radius: 12px; padding: 4px; gap: 4px;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color: var(--text-2) !important;
  font-size: 14px; font-weight: 500;
  padding: 8px 18px !important;
  background: transparent !important;
  transition: all .2s;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,255,136,0.12) !important;
  color: var(--green) !important;
  border-bottom: none !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select,
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--border-green) !important;
  box-shadow: 0 0 0 2px rgba(0,255,136,0.1) !important;
}
.stSlider [data-testid="stThumbValue"] { color: var(--green) !important; }
.stSlider div[role="slider"] { background: var(--green) !important; }
[data-testid="stSlider"] > div > div > div { background: var(--green) !important; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--green), #00d4aa) !important;
  color: #060a0f !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 28px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 15px !important;
  transition: all .3s !important;
  box-shadow: 0 0 30px rgba(0,255,136,0.25) !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 50px rgba(0,255,136,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Metrics */
[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 20px;
}
[data-testid="stMetricValue"] {
  color: var(--green) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { color: var(--blue) !important; }
[data-testid="stMetricLabel"] { color: var(--text-2) !important; font-size: 12px !important; }

/* Dataframes */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 12px; overflow: hidden; }
.stDataFrame th { background: rgba(0,255,136,0.06) !important; color: var(--green) !important; }
.stDataFrame td { color: var(--text-1) !important; }

/* Alerts */
.stSuccess { background: rgba(0,255,136,0.07) !important; border-color: var(--border-green) !important; color: var(--green) !important; border-radius: 12px !important; }
.stError   { background: rgba(239,68,68,0.07) !important; border-color: rgba(239,68,68,0.25) !important; border-radius: 12px !important; }
.stWarning { background: rgba(251,146,60,0.07) !important; border-color: rgba(251,146,60,0.25) !important; border-radius: 12px !important; }
.stInfo    { background: rgba(14,165,233,0.07) !important; border-color: rgba(14,165,233,0.25) !important; color: var(--blue) !important; border-radius: 12px !important; }

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 1.5px dashed var(--border-green) !important;
  border-radius: 16px !important;
  padding: 20px !important;
}

/* Checkbox */
[data-testid="stCheckbox"] span { color: var(--text-1) !important; }
[data-testid="stCheckbox"] input:checked + div { background: var(--green) !important; border-color: var(--green) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* ─── SCROLLBAR ──────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: rgba(0,255,136,0.25); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,255,136,0.5); }

/* ─── BACKGROUND GRID ────────────────────────────── */
.bg-grid {
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(0,255,136,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,136,0.02) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
}

/* ─── FLOATING ORBS ──────────────────────────────── */
.orb { position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; filter: blur(90px); }
.orb-1 { width:500px;height:500px; background:radial-gradient(circle,rgba(0,255,136,0.07),transparent 70%); top:-150px;right:-100px; animation:orbFloat 12s ease-in-out infinite; }
.orb-2 { width:400px;height:400px; background:radial-gradient(circle,rgba(14,165,233,0.06),transparent 70%); bottom:20%;left:-100px; animation:orbFloat 15s ease-in-out infinite reverse; }
.orb-3 { width:320px;height:320px; background:radial-gradient(circle,rgba(168,85,247,0.05),transparent 70%); top:55%;right:10%; animation:orbFloat 10s ease-in-out infinite 3s; }
@keyframes orbFloat {
  0%,100% { transform:translate(0,0) scale(1); }
  33% { transform:translate(30px,-20px) scale(1.05); }
  66% { transform:translate(-20px,15px) scale(.95); }
}

/* ─── CURSOR ─────────────────────────────────────── */
.cursor-dot {
  position:fixed; top:0; left:0; pointer-events:none; z-index:99999;
  width:10px; height:10px; border-radius:50%; background:var(--green);
  transform:translate(-50%,-50%);
  box-shadow: 0 0 8px 2px var(--green), 0 0 22px 5px rgba(0,255,136,0.5), 0 0 45px 10px rgba(0,255,136,0.18);
  transition: width .15s, height .15s, box-shadow .2s;
}
.cursor-dot.hovered { width:16px;height:16px; box-shadow:0 0 14px 4px var(--green),0 0 36px 9px rgba(0,255,136,0.6),0 0 65px 18px rgba(0,255,136,0.22); }
.cursor-ring {
  position:fixed; top:0; left:0; pointer-events:none; z-index:99998;
  width:40px; height:40px; border-radius:50%;
  border:1.5px solid rgba(0,255,136,0.5);
  transform:translate(-50%,-50%);
  box-shadow: 0 0 12px rgba(0,255,136,0.15), inset 0 0 8px rgba(0,255,136,0.05);
  transition: width .25s, height .25s, border-color .25s;
}
.cursor-ring.hovered { width:58px;height:58px; border-color:rgba(0,255,136,0.8); }
#trailCanvas { position:fixed; inset:0; pointer-events:none; z-index:99997; }

/* ─── SPLASH SCREEN ──────────────────────────────── */
#splash {
  position:fixed; inset:0; z-index:999999;
  background:var(--bg-dark);
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:22px;
  transition: opacity .7s ease, visibility .7s ease;
}
#splash.hidden { opacity:0; visibility:hidden; pointer-events:none; }
.spl-circle {
  width:90px;height:90px; border-radius:50%;
  border:2px solid var(--border-green);
  display:flex; align-items:center; justify-content:center;
  background:radial-gradient(circle, rgba(0,255,136,0.07), transparent 70%);
  position:relative;
  animation: splLogo .8s cubic-bezier(.16,1,.3,1) both;
}
.spl-circle::before {
  content:''; position:absolute; inset:-7px; border-radius:50%;
  border:1px solid rgba(0,255,136,0.15); animation:spinR 3s linear infinite;
}
.spl-circle::after {
  content:''; position:absolute; inset:-15px; border-radius:50%;
  border:1px dashed rgba(0,255,136,0.07); animation:spinR 8s linear infinite reverse;
}
@keyframes spinR { to { transform:rotate(360deg); } }
.spl-icon { font-size:36px; filter:drop-shadow(0 0 12px rgba(0,255,136,0.7)); }
.spl-word {
  font-family:'Space Grotesk',sans-serif;
  font-size:32px; font-weight:700; letter-spacing:-.04em;
  background:linear-gradient(135deg,#fff 30%,var(--green));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation: splWord .8s .15s cubic-bezier(.16,1,.3,1) both;
}
.spl-tag {
  font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--text-2);
  animation: splFade .5s .35s ease both;
}
.spl-bar { width:180px; height:2px; background:var(--border); border-radius:99px; overflow:hidden; animation: splFade .4s .45s ease both; }
.spl-fill { height:100%; background:linear-gradient(90deg,var(--green),var(--blue)); animation:splProg 1.7s .45s cubic-bezier(.4,0,.2,1) forwards; }
@keyframes splProg { to { width:100%; } }
@keyframes splFade  { from{opacity:0} to{opacity:1} }
@keyframes splLogo  { from{opacity:0;transform:scale(.7) translateY(20px)} to{opacity:1;transform:scale(1) translateY(0)} }
@keyframes splWord  { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

/* ─── HERO BANNER ────────────────────────────────── */
.hero-wrap {
  position:relative; z-index:1; text-align:center;
  padding: 48px 20px 32px;
  animation: heroBanner .8s 2.3s cubic-bezier(.16,1,.3,1) both;
}
@keyframes heroBanner { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
.hero-badge {
  display:inline-flex; align-items:center; gap:8px;
  padding:5px 14px; border-radius:99px;
  border:1px solid var(--border-green); background:rgba(0,255,136,0.05);
  font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:var(--green); margin-bottom:20px;
}
.badge-dot { width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:bdpulse 2s infinite; }
@keyframes bdpulse { 0%,100%{opacity:1}50%{opacity:.3} }
.hero-title {
  font-family:'Space Grotesk',sans-serif;
  font-size:clamp(44px,7vw,92px); font-weight:800; letter-spacing:-.04em; line-height:.95;
  margin-bottom:18px; color:var(--text-1);
}
.hero-title .grad {
  background:linear-gradient(135deg,var(--green) 0%,#00d4ff 50%,var(--purple) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  background-size:200%; animation:gradShift 4s ease infinite 3.5s;
}
@keyframes gradShift { 0%,100%{background-position:0% 50%}50%{background-position:100% 50%} }
.hero-sub {
  font-size:17px; color:var(--text-2); line-height:1.7;
  max-width:560px; margin:0 auto 32px;
}
.hero-stats-row {
  display:flex; gap:0; border:1px solid var(--border); border-radius:16px; overflow:hidden;
  background:var(--bg-card); backdrop-filter:blur(20px);
  max-width:680px; margin:0 auto;
}
.hstat { flex:1; padding:20px 28px; border-right:1px solid var(--border); text-align:left; }
.hstat:last-child { border-right:none; }
.hstat-val { font-family:'Space Grotesk',sans-serif; font-size:30px;font-weight:700;letter-spacing:-.03em;color:var(--green); }
.hstat-lbl { font-size:12px;color:var(--text-2);margin-top:2px; }

/* ─── SECTION LABEL ──────────────────────────────── */
.sec-label { font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--green);margin-bottom:8px; }
.sec-title { font-family:'Space Grotesk',sans-serif;font-size:clamp(26px,4vw,46px);font-weight:700;letter-spacing:-.03em;color:var(--text-1);margin-bottom:12px; }

/* ─── GLASS CARDS ────────────────────────────────── */
.glass-card {
  background:var(--bg-card); border:1px solid var(--border); border-radius:20px;
  padding:28px; backdrop-filter:blur(16px);
  transition: border-color .3s, box-shadow .3s, transform .3s;
}
.glass-card:hover { border-color:var(--border-green); transform:translateY(-3px); box-shadow:0 16px 48px rgba(0,0,0,.35),0 0 40px rgba(0,255,136,0.05); }
.glass-card-green {
  background:rgba(0,255,136,0.03); border:1px solid var(--border-green); border-radius:20px; padding:28px;
}

/* Legacy compat names */
.metric-card  { background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:1.4rem;margin-bottom:1rem; }
.highlight-card { background:rgba(0,255,136,0.04);border:1px solid var(--border-green);border-radius:16px;padding:1.4rem;margin-bottom:1rem; }

/* ─── PROBABILITY DISPLAY ────────────────────────── */
.prob-ring-wrap { display:flex;flex-direction:column;align-items:center;gap:14px;padding:28px 0; }
.prob-big { font-family:'Space Grotesk',sans-serif;font-size:clamp(56px,9vw,96px);font-weight:800;letter-spacing:-.05em;line-height:1; }
.prob-green  { color:var(--green); text-shadow:0 0 40px rgba(0,255,136,0.4); }
.prob-yellow { color:#f59e0b;       text-shadow:0 0 40px rgba(245,158,11,0.35); }
.prob-red    { color:#f87171;       text-shadow:0 0 40px rgba(248,113,113,0.3); }
.prob-label  { font-size:14px;font-weight:600;letter-spacing:.04em; }
.prob-sub    { font-size:13px;color:var(--text-2); }

/* ─── FEATURE PILL TAGS ──────────────────────────── */
.tag { display:inline-block;padding:4px 12px;border-radius:8px;font-size:12px;font-weight:500;margin:3px; }
.tag-green { background:rgba(0,255,136,0.1);color:var(--green);border:1px solid var(--border-green); }
.tag-blue  { background:rgba(14,165,233,0.1);color:var(--blue);border:1px solid rgba(14,165,233,0.25); }
.tag-purple{ background:rgba(168,85,247,0.1);color:var(--purple);border:1px solid rgba(168,85,247,0.25); }

/* ─── STEPS / PIPELINE ───────────────────────────── */
.step-row { display:flex;gap:14px;align-items:flex-start;margin-bottom:18px; }
.step-num { width:34px;height:34px;border-radius:10px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;border:1px solid var(--border-green);background:rgba(0,255,136,0.08);color:var(--green); }
.step-content h4 { font-size:15px;font-weight:600;color:var(--text-1);margin:0 0 4px; }
.step-content p  { font-size:13px;color:var(--text-2);margin:0;line-height:1.6; }

/* ─── MODEL BADGE ────────────────────────────────── */
.model-badge {
  display:inline-flex;align-items:center;gap:8px;
  padding:6px 16px;border-radius:10px;font-size:12px;font-weight:600;
  background:rgba(0,255,136,0.08);border:1px solid var(--border-green);color:var(--green);
}

/* ─── SIDEBAR CUSTOM ─────────────────────────────── */
.sidebar-logo {
  font-family:'Space Grotesk',sans-serif;
  font-size:22px;font-weight:800;letter-spacing:-.03em;
  background:linear-gradient(135deg,var(--green),#00d4ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.sidebar-stat { padding:10px 0;border-bottom:1px solid var(--border); }
.sidebar-stat-val { font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;color:var(--green); }
.sidebar-stat-lbl { font-size:12px;color:var(--text-2); }

/* ─── FOOTER ─────────────────────────────────────── */
.footer-band {
  margin-top:40px;padding:24px 0;border-top:1px solid var(--border);
  text-align:center;color:var(--text-3);font-size:12px;
}
.footer-band span { color:var(--green); }
</style>
""", unsafe_allow_html=True)

# ─── INJECT BACKGROUND LAYERS + CURSOR + SPLASH ──────────────────────────────
st.markdown("""
<!-- Background grid -->
<div class="bg-grid"></div>
<!-- Ambient orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>

<!-- Custom cursor -->
<div class="cursor-dot" id="cursorDot"></div>
<div class="cursor-ring" id="cursorRing"></div>
<canvas id="trailCanvas"></canvas>

<!-- Splash entry screen -->
<div id="splash">
  <div class="spl-circle"><span class="spl-icon">⚽</span></div>
  <div class="spl-word">OFF SIDE</div>
  <div class="spl-tag">Football · AI · Analytics</div>
  <div class="spl-bar"><div class="spl-fill" style="width:0"></div></div>
</div>

<script>
/* ═══ SPLASH ════════════════════════════════════════ */
window.addEventListener('load', () => {
  setTimeout(() => {
    const s = document.getElementById('splash');
    if(s) s.classList.add('hidden');
  }, 2100);
});

/* ═══ CURSOR + COMET TRAIL ══════════════════════════ */
const cDot   = document.getElementById('cursorDot');
const cRing  = document.getElementById('cursorRing');
const tCvs   = document.getElementById('trailCanvas');
if(cDot && cRing && tCvs) {
  const tctx = tCvs.getContext('2d');
  function resizeT(){ tCvs.width=window.innerWidth; tCvs.height=window.innerHeight; }
  resizeT(); window.addEventListener('resize', resizeT);

  let mx=0, my=0, rx=0, ry=0, pmx=0, pmy=0, vis=false;
  const particles = [];

  document.addEventListener('mousemove', e => {
    pmx=mx; pmy=my; mx=e.clientX; my=e.clientY;
    if(!vis){ cDot.style.opacity='1'; cRing.style.opacity='1'; vis=true; }
  });
  document.addEventListener('mouseleave', () => {
    cDot.style.opacity='0'; cRing.style.opacity='0'; vis=false;
  });

  class TDot {
    constructor(x,y){
      this.x=x+(Math.random()-.5)*4; this.y=y+(Math.random()-.5)*4;
      this.r=Math.random()*2.6+.8; this.life=1.0;
      const a=Math.random()*Math.PI*2;
      this.vx=Math.cos(a)*(Math.random()*.8);
      this.vy=Math.sin(a)*(Math.random()*.8)-.3;
      this.decay=Math.random()*.032+.022;
      this.hue=Math.random()>.25?150:185;
    }
    update(){ this.x+=this.vx;this.y+=this.vy;this.r*=.97;this.life-=this.decay; }
    draw(c){
      if(this.life<=0) return;
      const a=Math.max(0,this.life);
      const g=c.createRadialGradient(this.x,this.y,0,this.x,this.y,this.r*3.5);
      g.addColorStop(0,`hsla(${this.hue},100%,65%,${a*.9})`);
      g.addColorStop(.4,`hsla(${this.hue},100%,60%,${a*.35})`);
      g.addColorStop(1,`hsla(${this.hue},100%,55%,0)`);
      c.beginPath(); c.arc(this.x,this.y,this.r*3.5,0,Math.PI*2);
      c.fillStyle=g; c.fill();
      c.beginPath(); c.arc(this.x,this.y,this.r,0,Math.PI*2);
      c.fillStyle=`hsla(${this.hue},100%,92%,${a})`; c.fill();
    }
  }

  function spawn(x,y){
    const dx=x-pmx,dy=y-pmy,sp=Math.sqrt(dx*dx+dy*dy);
    const n=Math.min(Math.floor(sp*.4+1),5);
    for(let i=0;i<n;i++) particles.push(new TDot(x,y));
  }

  function loop(){
    cDot.style.left=mx+'px'; cDot.style.top=my+'px';
    rx+=(mx-rx)*.1; ry+=(my-ry)*.1;
    cRing.style.left=rx+'px'; cRing.style.top=ry+'px';
    if(vis) spawn(mx,my);
    tctx.clearRect(0,0,tCvs.width,tCvs.height);
    for(let i=particles.length-1;i>=0;i--){
      particles[i].update(); particles[i].draw(tctx);
      if(particles[i].life<=0) particles.splice(i,1);
    }
    if(particles.length>220) particles.splice(0,particles.length-220);
    requestAnimationFrame(loop);
  }
  loop();

  document.querySelectorAll('a,button,input,select,[role="option"],[data-testid]').forEach(el=>{
    el.addEventListener('mouseenter',()=>{
      cDot.classList.add('hovered'); cRing.classList.add('hovered');
      for(let i=0;i<10;i++) particles.push(new TDot(mx,my));
    });
    el.addEventListener('mouseleave',()=>{ cDot.classList.remove('hovered'); cRing.classList.remove('hovered'); });
  });
}

/* ═══ ANIMATED COUNTER ══════════════════════════════ */
function animCount(id, target, isFloat, dec, dur, delay){
  const el = document.getElementById(id);
  if(!el) return;
  setTimeout(()=>{
    const t0=performance.now();
    const step=ts=>{
      const p=Math.min((ts-t0)/dur,1);
      const e=1-Math.pow(1-p,3);
      const v=target*e;
      el.textContent=isFloat ? v.toFixed(dec) : Math.round(v).toLocaleString();
      if(p<1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, delay);
}
animCount('cnt-rows',  1184660, false, 0, 2000, 2300);
animCount('cnt-feats', 103,     false, 0, 1600, 2400);
animCount('cnt-ap',    0.4619,  true,  4, 1800, 2500);
</script>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_artefacts():
    """Download missing files then load everything. Cached after first run."""
    import shutil
    HF_REPO      = "RishabhRana37/offside-artefacts"
    HF_REPO_TYPE = "dataset"
    REQUIRED     = ["catboost_model.cbm", "fitted_pipeline.pkl", "te_smooth_maps.pkl"]

    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
            for fname in missing:
                path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=fname,
                    repo_type=HF_REPO_TYPE,
                    local_dir=".",
                )
                if os.path.abspath(path) != os.path.abspath(fname):
                    shutil.move(path, fname)
        except Exception as e:
            return None, None, {}, {"global_mean": 0.086}, str(e)

    m = CatBoostClassifier()
    m.load_model("catboost_model.cbm")

    with open("fitted_pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)

    profiles = {}
    if os.path.exists("player_profiles.pkl"):
        with open("player_profiles.pkl", "rb") as f:
            profiles = pickle.load(f)

    te = {"global_mean": 0.086}
    if os.path.exists("te_smooth_maps.pkl"):
        with open("te_smooth_maps.pkl", "rb") as f:
            te = pickle.load(f)

    return m, pipe, profiles, te, None


with st.spinner("Loading model artefacts... (first boot may take ~60s)"):
    model, pipeline, player_profiles, te_maps, _load_err = load_all_artefacts()

if _load_err or model is None:
    st.error(f"Could not load model artefacts: {_load_err}")
    st.stop()

GLOBAL_MEAN    = te_maps.get("global_mean", 0.086)
MODEL_FEATURES = model.feature_names_   # exact 103-feature list

# ─────────────────────────────────────────────────────────────────────────────
# Feature builder — mirrors improve_and_submit.py exactly
# ─────────────────────────────────────────────────────────────────────────────
def build_features_for_inference(row_dict: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with exactly the 103 features the model expects.
    row_dict must contain all raw input fields.
    """
    df = pd.DataFrame([row_dict])

    # ── Base pipeline transform ──────────────────────────────────────────────
    feat = pipeline.transform(df)

    # ── Extra vectorised features (same as improve_and_submit.py) ────────────
    mr  = float(row_dict.get("minutes_ratio", 0.0))
    xg  = float(row_dict.get("avg_xG",        0.0))
    xa  = float(row_dict.get("avg_xA",        0.0))
    sh  = float(row_dict.get("avg_shots",     0.0))
    npx = float(row_dict.get("avg_npxG",      0.0))
    xgc = float(row_dict.get("avg_xGChain",   0.0))
    xgb = float(row_dict.get("avg_xGBuildup", 0.0))
    kp  = float(row_dict.get("avg_key_passes",0.0))
    mv  = float(row_dict.get("market_value_before_match", 0.0))
    pmv = float(row_dict.get("highest_market_value_in_eur", 0.0))
    caps  = float(row_dict.get("international_caps",  0.0))
    goals = float(row_dict.get("international_goals", 0.0))

    feat["expected_npxG_in_match"]       = npx * mr
    feat["expected_xGChain_in_match"]    = xgc * mr
    feat["expected_xGBuildup_in_match"]  = xgb * mr
    feat["expected_key_passes_in_match"] = kp  * mr
    feat["xG_per_shot_ext"]              = xg  / (sh + 1e-5)
    feat["market_value_ratio_peak_ext"]  = mv  / (pmv + 1e-5)
    feat["attacker_mv"]                  = feat.get("is_attacker", pd.Series([0])).values[0] * mv
    feat["intl_efficiency_ext"]          = goals / (caps + 1e-5)

    # team/opponent/total goals: unknown at inference → 0
    feat["team_goals"]        = 0
    feat["opponent_goals"]    = 0
    feat["match_total_goals"] = 0

    # ── Smoothed TE lookups ──────────────────────────────────────────────────
    te_cols = ["name_y", "home_club_name", "away_club_name", "competition_type", "referee"]
    for col in te_cols:
        enc = f"{col}_te_smooth"
        mapping = te_maps.get(col, {})
        val = row_dict.get(col, "")
        feat[enc] = float(mapping.get(val, GLOBAL_MEAN))

    # ── Align to exact model feature list ───────────────────────────────────
    for col in MODEL_FEATURES:
        if col not in feat.columns:
            feat[col] = 0
    X = feat[list(MODEL_FEATURES)].copy()

    # CatBoost needs cat cols as strings
    cat_cols_in_X = [c for c in pipeline.cat_cols if c in X.columns]
    for c in cat_cols_in_X:
        X[c] = X[c].astype(str)

    return X, cat_cols_in_X

def predict_proba(row_dict: dict) -> float:
    X, cat_cols = build_features_for_inference(row_dict)
    cat_idx = [list(MODEL_FEATURES).index(c) for c in cat_cols]
    pool = Pool(X, cat_features=cat_idx)
    prob = float(model.predict_proba(pool)[0, 1])
    # post-processing: force 0 if player didn't play
    if row_dict.get("minutes_played", 90) == 0:
        prob = 0.0
    return prob

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — premium design
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">⚽ OFF SIDE</div>
    <div style="font-size:13px;color:var(--text-2);margin:8px 0 16px;line-height:1.6">
      AI-powered goal-scoring prediction engine.<br/>
      Built for <span style="color:var(--green);font-weight:600">IEEE CS MUJ Datathon 2026</span>.
    </div>
    <hr style="border-color:var(--border);margin:0 0 16px">
    <div class="sidebar-stat">
      <div class="sidebar-stat-val">1.18M</div>
      <div class="sidebar-stat-lbl">Training Appearances</div>
    </div>
    <div class="sidebar-stat">
      <div class="sidebar-stat-val">103</div>
      <div class="sidebar-stat-lbl">Engineered Features</div>
    </div>
    <div class="sidebar-stat">
      <div class="sidebar-stat-val" style="color:var(--blue)">0.4619</div>
      <div class="sidebar-stat-lbl">OOF Average Precision</div>
    </div>
    <div style="margin-top:16px">
    <div class="model-badge">🤖 CatBoost · depth=6</div>
    </div>
    <hr style="border-color:var(--border);margin:16px 0">
    <div style="font-size:12px;color:var(--text-3);line-height:1.8">
      <div>✅ 5-Fold Stratified CV</div>
      <div>✅ Smoothed OOF Target Encoding</div>
      <div>✅ Post-processing rules</div>
      <div>✅ 24,428 Player Profiles</div>
    </div>
    <hr style="border-color:var(--border);margin:16px 0">
    <div style="font-size:11px;color:var(--text-3);">Built by <span style="color:var(--green)">RishabhRana37</span></div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header — full frontend design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">
    <div class="badge-dot"></div>
    Live · IEEE Datathon 2026
  </div>
  <div class="hero-title">
    Predict Who<br>
    <span class="grad">Scores Next</span>
  </div>
  <div class="hero-sub">
    Advanced CatBoost models trained on 1.18 million match appearances.
    Predict goal-scoring probability using xG, market value, playing time, and ensemble target encoding.
  </div>
  <div class="hero-stats-row">
    <div class="hstat">
      <div class="hstat-val" id="cnt-rows">1,184,660</div>
      <div class="hstat-lbl">Training Appearances</div>
    </div>
    <div class="hstat">
      <div class="hstat-val" id="cnt-feats">103</div>
      <div class="hstat-lbl">Engineered Features</div>
    </div>
    <div class="hstat">
      <div class="hstat-val" id="cnt-ap">0.4619</div>
      <div class="hstat-lbl">OOF Average Precision</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_eda, tab_about = st.tabs([
    "🎯 Player Goal Predictor",
    "📁 Batch Predictor",
    "📈 Model Insights",
    "ℹ️ Methodology",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PLAYER PREDICTOR
# ═══════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Select a Player & Match Context")

    player_names = sorted(player_profiles.keys())
    col_p, _ = st.columns([1, 1])
    with col_p:
        selected_player = st.selectbox(
            "👤 Player Profile (auto-fills stats)",
            ["— Custom Profile —"] + player_names
        )

    profile = {}
    if selected_player != "— Custom Profile —" and selected_player in player_profiles:
        profile = player_profiles[selected_player]

    # ── Defaults from profile ──────────────────────────────────────────
    def pf(key, default):
        v = profile.get(key, default)
        return v if v is not None else default

    d_age    = float(pf("age", 26.0))
    d_foot   = str(pf("foot", "right"))
    d_pos    = str(pf("position", "Attack"))
    d_subpos = str(pf("sub_position", "Center-Forward"))
    d_ctry   = str(pf("country_of_citizenship", "England"))
    d_club   = str(pf("home_club_name", "Arsenal FC"))
    d_mv     = float(pf("market_value_before_match", 10_000_000.0))
    d_pmv    = float(pf("highest_market_value_in_eur", 20_000_000.0))
    d_caps   = int(pf("international_caps", 0))
    d_goals  = int(pf("international_goals", 0))
    d_xg     = float(pf("avg_xG", 0.15))
    d_xa     = float(pf("avg_xA", 0.08))
    d_shots  = float(pf("avg_shots", 1.5))
    d_mr     = float(pf("minutes_ratio", 0.7))

    # clamp
    d_age  = min(max(d_age,  15.0), 60.0)
    d_xg   = min(max(d_xg,   0.0),  30.0)
    d_xa   = min(max(d_xa,   0.0),  20.0)
    d_shots= min(max(d_shots, 0.0), 200.0)
    d_mr   = min(max(d_mr,   0.0),  1.0)

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Player Bio**")
        age      = st.slider("Age", 15, 60, int(d_age))
        foot     = st.selectbox("Preferred Foot", ["right","left","both"],
                                index=["right","left","both"].index(d_foot)
                                if d_foot in ["right","left","both"] else 0)
        position = st.selectbox("Position Group",
                                ["Attack","Midfield","Defender","Goalkeeper"],
                                index=["Attack","Midfield","Defender","Goalkeeper"].index(d_pos)
                                if d_pos in ["Attack","Midfield","Defender","Goalkeeper"] else 0)
        sub_position  = st.text_input("Sub-position",   d_subpos)
        citizenship   = st.text_input("Country",         d_ctry)
        height_in_cm  = st.number_input("Height (cm)", 140, 220, 180)

    with col2:
        st.markdown("**📊 Performance Stats**")
        avg_xg    = st.number_input("Avg xG",    0.0, 30.0,  d_xg,   step=0.01)
        avg_xa    = st.number_input("Avg xA",    0.0, 20.0,  d_xa,   step=0.01)
        avg_shots = st.number_input("Avg Shots", 0.0, 200.0, d_shots, step=0.1)
        avg_npxg  = st.number_input("Avg npxG",  0.0, 30.0,  max(d_xg*0.9, 0.0), step=0.01)
        avg_xgchain  = st.number_input("Avg xGChain",   0.0, 20.0, 0.1, step=0.01)
        avg_xgbuildup= st.number_input("Avg xGBuildup", 0.0, 20.0, 0.05, step=0.01)
        avg_kp    = st.number_input("Avg Key Passes", 0.0, 20.0, 0.3, step=0.01)
        min_ratio = st.slider("Season Minutes Ratio", 0.0, 1.0, d_mr)

    with col3:
        st.markdown("**🏟️ Match Context**")
        home_club    = st.text_input("Player's Club",   d_club)
        opponent     = st.text_input("Opponent Club",   "Manchester City FC")
        home_away    = st.selectbox("Venue", ["Home","Away"])
        stadium      = st.text_input("Stadium",  "Emirates Stadium")
        referee      = st.text_input("Referee",  "Michael Oliver")
        comp_type    = st.selectbox("Competition", ["domestic_league","domestic_cup","international_cup","other"])
        confederation= st.selectbox("Confederation", ["uefa","conmebol","concacaf","afc","caf","ofc"])
        minutes_played = st.slider("Expected Minutes", 0, 90, 90)
        starter_flag   = st.checkbox("Starting XI?", value=True)
        caps    = st.number_input("International Caps",  0, 300, d_caps)
        int_goals= st.number_input("International Goals", 0, 200, d_goals)
        market_val = st.number_input("Market Value (€)", 0.0, 500_000_000.0, d_mv, step=500_000.0)
        highest_mv = st.number_input("Peak Market Value (€)", 0.0, 500_000_000.0, d_pmv, step=500_000.0)

    # Derived flags
    is_attacker = 1 if position == "Attack" else 0
    full_match  = 1 if minutes_played == 90 else 0
    substitute  = 1 if not starter_flag else 0
    prime_age   = 1 if 24 <= age <= 29 else 0
    veteran     = 1 if age >= 32 else 0
    has_nt      = 1 if caps > 0 else 0
    finisher    = 1 if (avg_xg > 0.35 and is_attacker) else 0
    creative    = 1 if avg_xa > 0.20 else 0

    st.divider()
    predict_clicked = st.button("🔮 Predict Goal Probability", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Running inference…"):
            row = {
                "date":              "2026-01-15",
                "foot":              foot,
                "position":          position,
                "sub_position":      sub_position,
                "country_of_citizenship": citizenship,
                "home_club_name":    home_club if home_away == "Home" else opponent,
                "away_club_name":    opponent  if home_away == "Home" else home_club,
                "stadium":           stadium,
                "referee":           referee,
                "competition_type":  comp_type,
                "confederation":     confederation,
                "market_value_tier": "High" if market_val > 20_000_000 else ("Medium" if market_val > 5_000_000 else "Low"),
                "age_bucket":        "24-29" if prime_age else ("30+" if age >= 30 else "U23"),
                "name_x":            "Premier League",
                "home_away":         home_away,
                "country_name":      citizenship,
                "name_y":            selected_player if selected_player != "— Custom Profile —" else "Custom Player",
                "avg_xG":            avg_xg,
                "avg_xA":            avg_xa,
                "avg_shots":         avg_shots,
                "avg_npxG":          avg_npxg,
                "avg_xGChain":       avg_xgchain,
                "avg_xGBuildup":     avg_xgbuildup,
                "avg_key_passes":    avg_kp,
                "minutes_ratio":     min_ratio,
                "market_value_before_match":   market_val,
                "highest_market_value_in_eur": highest_mv,
                "age":               float(age),
                "height_in_cm":      float(height_in_cm),
                "international_caps":   float(caps),
                "international_goals":  float(int_goals),
                "starter_flag":      int(starter_flag),
                "minutes_played":    minutes_played,
                "is_attacker":       is_attacker,
                "full_match_flag":   full_match,
                "substitute_flag":   substitute,
                "card_flag":         0,
                "prime_age_flag":    prime_age,
                "veteran_flag":      veteran,
                "has_national_team_experience": has_nt,
                "finisher_flag":     finisher,
                "creative_player_flag": creative,
                "has_understat":     1,
                "analytics_coverage_flag": 1,
                "home_club_id":      0,
                "away_club_id":      0,
                "home_club_goals":   0,
                "away_club_goals":   0,
                "home_club_position":0,
                "away_club_position":0,
                "home_club_manager_name": "",
                "away_club_manager_name": "",
                "home_club_formation": "",
                "away_club_formation": "",
                "yellow_cards": 0,
                "red_cards":    0,
                "goals":        0,
                "assists":      0,
                "attendance":   0,
                "season":       2025,
            }

            try:
                prob = predict_proba(row)
                pct = prob * 100

                # Colour coding
                if pct >= 30:
                    col_cls = "prob-green"; label = "🔥 Extreme Goal Threat"
                elif pct >= 15:
                    col_cls = "prob-yellow"; label = "🟡 High Goal Threat"
                elif pct >= 5:
                    col_cls = "prob-yellow"; label = "⚠️ Moderate Goal Threat"
                else:
                    col_cls = "prob-red"; label = "🛑 Low Goal Threat"

                st.divider()
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.markdown("<div class='highlight-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='prob-big {col_cls}'>{pct:.1f}%</div>"
                        f"<div style='font-size:1.1rem;font-weight:600;margin-top:4px'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                    player_label = selected_player if selected_player != "— Custom Profile —" else "Custom Player"
                    st.markdown(f"**Player:** {player_label} · **{position}**")
                    st.markdown(f"**Minutes:** {minutes_played} · **Starter:** {'Yes' if starter_flag else 'No'}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with c2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("#### Key Prediction Drivers")

                    # TE lookup for the selected player
                    player_te = te_maps.get("name_y", {}).get(
                        row["name_y"], GLOBAL_MEAN
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Historical Goal Rate (TE)", f"{player_te:.2%}")
                        st.metric("Expected xG in Match", f"{avg_xg * min_ratio:.3f}")
                        st.metric("Avg xG", f"{avg_xg:.3f}")
                    with col_b:
                        st.metric("Intl. Goal Efficiency", f"{int_goals/(caps+1e-5):.2%}")
                        st.metric("Market Value Ratio", f"{market_val/(highest_mv+1e-5):.1%}")
                        st.metric("Minutes Ratio", f"{min_ratio:.0%}")

                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback; st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: BATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Predictions from CSV")
    st.markdown(
        "Upload a CSV file (same schema as `test.csv`) to generate "
        "goal-scoring probabilities for every appearance row."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"Loaded **{len(batch_df):,}** rows · {batch_df.shape[1]} columns")
            st.dataframe(batch_df.head(5))

            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner("Processing…"):
                    # Run pipeline transform on the whole batch
                    feat = pipeline.transform(batch_df)

                    # Extra features
                    mr  = batch_df.get("minutes_ratio",  pd.Series(0.0, index=batch_df.index))
                    xg  = batch_df.get("avg_xG",         pd.Series(0.0, index=batch_df.index))
                    xa  = batch_df.get("avg_xA",         pd.Series(0.0, index=batch_df.index))
                    sh  = batch_df.get("avg_shots",      pd.Series(0.0, index=batch_df.index))
                    npx = batch_df.get("avg_npxG",       pd.Series(0.0, index=batch_df.index))
                    xgc = batch_df.get("avg_xGChain",    pd.Series(0.0, index=batch_df.index))
                    xgb = batch_df.get("avg_xGBuildup",  pd.Series(0.0, index=batch_df.index))
                    kp  = batch_df.get("avg_key_passes", pd.Series(0.0, index=batch_df.index))
                    mv  = batch_df.get("market_value_before_match",  pd.Series(0.0, index=batch_df.index))
                    pmv = batch_df.get("highest_market_value_in_eur",pd.Series(0.0, index=batch_df.index))
                    caps_b  = batch_df.get("international_caps",  pd.Series(0.0, index=batch_df.index))
                    goals_b = batch_df.get("international_goals", pd.Series(0.0, index=batch_df.index))

                    feat["expected_npxG_in_match"]       = (npx * mr).values
                    feat["expected_xGChain_in_match"]    = (xgc * mr).values
                    feat["expected_xGBuildup_in_match"]  = (xgb * mr).values
                    feat["expected_key_passes_in_match"] = (kp  * mr).values
                    feat["xG_per_shot_ext"]              = (xg  / (sh.fillna(0) + 1e-5)).values
                    feat["market_value_ratio_peak_ext"]  = (mv  / (pmv + 1e-5)).values
                    feat["attacker_mv"]                  = (feat.get("is_attacker", 0).values * mv.values)
                    feat["intl_efficiency_ext"]          = (goals_b / (caps_b + 1e-5)).values
                    feat["team_goals"] = feat["opponent_goals"] = feat["match_total_goals"] = 0

                    # TE smooth
                    for col in ["name_y","home_club_name","away_club_name","competition_type","referee"]:
                        enc     = f"{col}_te_smooth"
                        mapping = te_maps.get(col, {})
                        if col in batch_df.columns:
                            feat[enc] = batch_df[col].map(mapping).fillna(GLOBAL_MEAN).values
                        else:
                            feat[enc] = GLOBAL_MEAN

                    # Align
                    for col in MODEL_FEATURES:
                        if col not in feat.columns:
                            feat[col] = 0
                    X = feat[list(MODEL_FEATURES)].copy()
                    cat_cols_in_X = [c for c in pipeline.cat_cols if c in X.columns]
                    for c in cat_cols_in_X:
                        X[c] = X[c].astype(str)
                    cat_idx = [list(MODEL_FEATURES).index(c) for c in cat_cols_in_X]

                    pool  = Pool(X, cat_features=cat_idx)
                    preds = model.predict_proba(pool)[:, 1]

                    # Post-process
                    if "minutes_played" in batch_df.columns:
                        preds[batch_df["minutes_played"].values == 0] = 0.0

                    result = batch_df.copy()
                    result["scored_probability"] = preds

                    st.success(f"✅ Done — {len(result):,} predictions generated")
                    st.dataframe(result[["appearance_id","scored_probability"] if "appearance_id" in result.columns else ["scored_probability"]].head(20))

                    csv_full = result.to_csv(index=False).encode()
                    sol_cols = (["appearance_id","scored_flag"]
                                if "appearance_id" in result.columns
                                else ["scored_probability"])
                    result2  = result.copy()
                    if "appearance_id" in result2.columns:
                        result2 = result2.rename(columns={"scored_probability": "scored_flag"})[["appearance_id","scored_flag"]]
                    kaggle_csv = result2.to_csv(index=False).encode()

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button("📥 Full Predictions CSV", csv_full,
                                           "batch_predictions.csv", "text/csv",
                                           use_container_width=True)
                    with dl2:
                        st.download_button("🎯 Kaggle solution.csv", kaggle_csv,
                                           "solution.csv", "text/csv",
                                           use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Model Validation & Insights")
    st.markdown("Validation metrics, feature importances, and SHAP explainability from training.")

    # Top features from the model
    importances  = model.get_feature_importance()
    feat_imp_df  = pd.DataFrame({"Feature": list(MODEL_FEATURES), "Importance": importances})
    feat_imp_df  = feat_imp_df.sort_values("Importance", ascending=False).head(20)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### 🌳 Top 20 Feature Importances")
        st.dataframe(feat_imp_df.reset_index(drop=True), use_container_width=True)

    with col_e2:
        st.markdown("#### 📊 Model Performance Summary")
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        mc1.metric("OOF Average Precision", "0.4619", delta="+12.9% vs baseline")
        mc2.metric("Training Rows", "1,184,660")
        mc1.metric("Features", "103")
        mc2.metric("CV Folds", "5 (Stratified)")
        mc1.metric("Positive Rate", "8.60%")
        mc2.metric("Post-Processing", "0-min rule")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 🎯 Target Distribution")
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(
            "Only **8.60%** of player appearances result in a goal — "
            "a heavily imbalanced dataset requiring **Average Precision** as the evaluation metric "
            "(not accuracy or AUC-ROC)."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # SHAP plots if available
    if os.path.exists("plots/shap_summary_plot.png"):
        st.markdown("#### 🧬 SHAP Feature Attribution")
        st.image("plots/shap_summary_plot.png", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### 📖 Methodology

This application deploys a CatBoost ML model trained to predict player-level goal-scoring.

#### Cross-Validation Framework
**Stratified 5-Fold CV** prevents leakage and maintains the 8.6% positive-class ratio across splits.
Validated using **Average Precision (AP)** — the area under the Precision-Recall curve.

#### Feature Engineering (103 Features)
| Category | Features |
|---|---|
| **Target Encoding (Smoothed)** | Player TE, home/away club TE, competition TE, referee TE (m=20) |
| **Expected Goal Ratios** | `expected_xG_in_match`, `expected_npxG_in_match`, `xG_per_shot_ext` |
| **Market Intelligence** | `market_value_ratio_peak`, `attacker_mv`, `log_market_value` |
| **International Profile** | `goal_per_cap`, `intl_efficiency_ext`, `has_national_team_experience` |
| **Temporal** | `year`, `month`, `day_of_week`, `is_weekend` |
| **Position Flags** | `is_attacker`, `is_midfielder`, `is_defender`, `is_goalkeeper` |
| **Match Context** | `attendance`, `team_goals`, `match_total_goals` |

#### Model Performance
| Model | OOF AP |
|---|---|
| LightGBM Baseline | 0.4090 |
| CatBoost (depth=6, lr=0.05, 1000 iter) | **0.4619** |
| Ensemble (nuclear_boost.py) | ~0.50 |

#### Post-Processing Rules
1. **Zero-minutes rule**: Any player with `minutes_played == 0` is forced to probability `0.0`
   → Measurable AP improvement since zero-minute players cannot score.

#### Dataset
- **Training:** 1,184,660 appearance records · Transfermarkt + Understat
- **Test:** ~200K appearance rows, no labels
- **Competition:** [IEEE CS MUJ Offside Football Analytics Datathon 2026](https://www.kaggle.com)
""")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-band">
  <span>OFF SIDE</span> &nbsp;·&nbsp; Football Analytics AI &nbsp;·&nbsp;
  Built for <span>IEEE CS MUJ Datathon 2026</span> &nbsp;·&nbsp;
  by <span>RishabhRana37</span>
</div>
""", unsafe_allow_html=True)

