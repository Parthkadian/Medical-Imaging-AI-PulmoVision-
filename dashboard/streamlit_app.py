import io, os, sys, time, base64, zipfile, json, csv
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
import plotly.graph_objects as go

# ── Fix project root on sys.path (needed when Streamlit launches from dashboard/) ─
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(page_title="PulmoVision AI", page_icon="🫁", layout="wide",
                   initial_sidebar_state="expanded")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# ── Session state ────────────────────────────────────────────────────────────
DEFAULTS = {
    "prediction_result": None, "last_uploaded_signature": None,
    "last_latency_ms": None, "timeline_steps": [], "stream_done": False,
    "uploaded_bytes": None, "uploaded_type": None, "uploaded_name": None,
    "original_image": None, "last_error": None, "history": [],
    # New sidebar state
    "dark_mode": True,
    "selected_model": "Attention U-Net (Current)",
    "upload_type": "Image Mode (PNG/JPG)",
    "error_logs": [],
    "reload_result_idx": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

/* ── Base ─────────────────────────────────────────── */
html,body,[data-testid="stApp"]{font-family:'Inter',sans-serif;background:#020906;color:#ecfeff}
.stApp{background:
  radial-gradient(ellipse at 8% 8%,rgba(45,212,191,.10),transparent 25%),
  radial-gradient(ellipse at 92% 6%,rgba(56,189,248,.09),transparent 25%),
  radial-gradient(ellipse at 50% 95%,rgba(167,243,208,.06),transparent 28%),
  linear-gradient(180deg,#020906 0%,#04130f 50%,#020906 100%) !important}
header{background:transparent!important}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#040f0b 0%,#061410 60%,#040f0b 100%) !important;
  border-right:1px solid rgba(45,212,191,.12) !important}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div{color:#ecfeff !important}

/* Force dark bg on every input / control inside sidebar */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb],
section[data-testid="stSidebar"] [data-baseweb] > div,
section[data-testid="stSidebar"] [data-baseweb] input,
section[data-testid="stSidebar"] [class*="control"],
section[data-testid="stSidebar"] [class*="Control"],
section[data-testid="stSidebar"] [class*="container"],
section[data-testid="stSidebar"] [class*="Container"]
  {background:rgba(8,20,15,.95) !important;color:#ecfeff !important}

/* Fix selectbox/dropdown — kill ALL white backgrounds */
div[data-testid="stSelectbox"],
div[data-testid="stSelectbox"] *{
  background:transparent !important;
  color:#ecfeff !important}

/* The actual visible control box */
div[data-baseweb="select"]{
  background:rgba(8,22,16,.95) !important;
  border:1px solid rgba(45,212,191,.28) !important;
  border-radius:12px !important;
  color:#ecfeff !important}
div[data-baseweb="select"]:hover{
  border-color:rgba(45,212,191,.55) !important}

/* Inner control container (the white box) */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div,
div[data-baseweb="control"],
div[data-baseweb="control"] > div,
div[data-baseweb="value-container"],
div[data-baseweb="single-value"],
div[data-baseweb="input-container"],
div[data-baseweb="base-input"],
div[data-baseweb="select"] input,
div[data-baseweb="select"] span{
  background:rgba(8,22,16,.95) !important;
  color:#ecfeff !important;
  caret-color:#2dd4bf !important}

/* Dropdown popup — full dark theme */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
div[data-baseweb="menu"] > div,
div[data-baseweb="menu"] ul,
div[data-baseweb="list"],
div[data-baseweb="popover"] ul{
  background:#07130f !important;
  border:1px solid rgba(45,212,191,.22) !important;
  border-radius:12px !important}
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li,
div[data-baseweb="option"]{
  background:#07130f !important;
  color:#ecfeff !important}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover,
div[data-baseweb="option"]:hover{
  background:rgba(45,212,191,.14) !important;
  color:#2dd4bf !important}
div[data-baseweb="option"][aria-selected="true"]{
  background:rgba(45,212,191,.18) !important;
  color:#5eead4 !important}

/* Fix radio buttons + container */
div[data-testid="stRadio"],
div[data-testid="stRadio"] > div{
  background:transparent !important}
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] p{color:#ecfeff !important}
div[data-testid="stRadio"] div[role="radiogroup"]{gap:.3rem}

/* Fix toggle */
div[data-testid="stToggle"] label,
div[data-testid="stToggle"] p{color:#ecfeff !important}
div[data-testid="stToggle"] div[data-checked="true"] span{background:#2dd4bf !important}

/* Fix expander */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] p{color:#ecfeff !important}
div[data-testid="stExpander"]{
  border:1px solid rgba(255,255,255,.07) !important;
  border-radius:12px !important;
  background:rgba(9,20,17,.6) !important}

/* Fix caption / small text */
.stCaption, div[data-testid="stCaptionContainer"] p{color:rgba(236,254,255,.6) !important}

/* Fix metric labels */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div{color:#ecfeff !important}

/* ── Main buttons ─────────────────────────────────────────── */
div[data-testid="stButton"]>button{
  width:100%;min-height:3.1rem;border-radius:14px;
  background:linear-gradient(135deg,#2dd4bf,#0e9488) !important;
  color:#ffffff !important;
  font-weight:800;font-size:.92rem;
  border:1px solid rgba(45,212,191,.3) !important;
  letter-spacing:.01em;
  transition:all .22s cubic-bezier(.4,0,.2,1);
  box-shadow:0 4px 14px rgba(45,212,191,.22)}
div[data-testid="stButton"]>button:hover{
  transform:translateY(-2px);
  box-shadow:0 10px 28px rgba(45,212,191,.35) !important;
  background:linear-gradient(135deg,#5eead4,#14b8a6) !important}
div[data-testid="stButton"]>button:disabled{
  opacity:.45 !important;transform:none !important;box-shadow:none !important}

/* Small icon-only buttons */
div[data-testid="stButton"]>button[kind="secondary"],
section[data-testid="stSidebar"] div[data-testid="stButton"]>button{
  min-height:2.4rem !important;
  background:rgba(45,212,191,.14) !important;
  color:#2dd4bf !important;
  border:1px solid rgba(45,212,191,.25) !important;
  font-size:.88rem !important;font-weight:700 !important;
  box-shadow:none !important}
section[data-testid="stSidebar"] div[data-testid="stButton"]>button:hover{
  background:rgba(45,212,191,.25) !important;transform:none !important}

/* ── Download buttons ─────────────────────────────────────── */
div[data-testid="stDownloadButton"]>button{
  width:100%;min-height:2.9rem;border-radius:14px;
  background:linear-gradient(135deg,rgba(6,95,70,.95),rgba(4,120,87,.95)) !important;
  color:#6ee7b7 !important;font-weight:800;
  border:1px solid rgba(52,211,153,.28) !important;
  transition:all .22s;box-shadow:0 4px 14px rgba(0,0,0,.22)}
div[data-testid="stDownloadButton"]>button:hover{
  background:linear-gradient(135deg,rgba(4,120,87,.95),rgba(5,150,105,.95)) !important;
  color:#d1fae5 !important;transform:translateY(-1px);
  box-shadow:0 8px 22px rgba(52,211,153,.25) !important}

/* ── File uploader ────────────────────────────────────────────────── */
div[data-testid="stFileUploader"]{
  background:rgba(10,22,18,.88) !important;
  border:1px solid rgba(45,212,191,.14) !important;
  border-radius:20px !important}
div[data-testid="stFileUploaderDropzone"]{
  border:2px dashed rgba(45,212,191,.4) !important;
  border-radius:16px !important;
  background:rgba(7,16,12,.85) !important;
  transition:all .25s !important}
div[data-testid="stFileUploaderDropzone"]:hover{
  border-color:rgba(45,212,191,.7) !important;
  background:rgba(45,212,191,.05) !important}
/* Dark all child text / icons inside dropzone */
div[data-testid="stFileUploaderDropzone"] *{
  color:#a7f3d0 !important}
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] span[class*="caption"],
div[data-testid="stFileUploaderDropzone"] p{
  color:rgba(167,243,208,.6) !important}
/* Browse files button inside uploader */
div[data-testid="stFileUploader"] button{
  background:rgba(45,212,191,.12) !important;
  color:#2dd4bf !important;
  border:1px solid rgba(45,212,191,.3) !important;
  border-radius:10px !important;
  font-weight:700 !important;
  transition:all .2s !important}
div[data-testid="stFileUploader"] button:hover{
  background:rgba(45,212,191,.22) !important;
  color:#5eead4 !important;
  border-color:rgba(45,212,191,.55) !important}

/* Refresh button */
div[data-testid="stButton"]>button[data-testid="refresh_api"]{
  background:rgba(56,189,248,.1) !important;
  color:#38bdf8 !important;
  border:1px solid rgba(56,189,248,.3) !important;
  font-size:.82rem !important}



/* ── Layout ──────────────────────────────────────────────── */
.block-container{max-width:1500px!important;padding:1rem 1.8rem 2rem!important}
label,.stMarkdown p{color:#ecfeff}

/* ══════════════════════════════════════════════════════════
   PREMIUM VISUAL UPGRADES
   ══════════════════════════════════════════════════════════ */

/* ── Keyframe Animations ─────────────────────────────────── */
@keyframes shimmer{
  0%{background-position:-200% center}
  100%{background-position:200% center}}
@keyframes fadeInUp{
  from{opacity:0;transform:translateY(14px)}
  to{opacity:1;transform:translateY(0)}}
@keyframes pulse{
  0%,100%{transform:scale(1);opacity:.9}
  50%{transform:scale(1.9);opacity:.25}}
@keyframes borderPulse{
  0%,100%{opacity:.45}
  50%{opacity:1}}

/* ── Custom Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:rgba(2,9,6,.4)}
::-webkit-scrollbar-thumb{
  background:linear-gradient(180deg,#2dd4bf,#0891b2);border-radius:999px}
::-webkit-scrollbar-thumb:hover{background:#2dd4bf}

/* ── Premium Tabs ────────────────────────────────────────── */
div[data-testid="stTabs"] [role="tablist"]{
  gap:.25rem !important;
  border-bottom:1px solid rgba(255,255,255,.08) !important;
  padding-bottom:0 !important}
div[data-testid="stTabs"] [role="tab"]{
  background:rgba(10,22,18,.72) !important;
  border:1px solid rgba(255,255,255,.07) !important;
  border-bottom:2px solid transparent !important;
  border-radius:12px 12px 0 0 !important;
  color:rgba(236,254,255,.52) !important;
  font-weight:700 !important;font-size:.86rem !important;
  padding:.55rem 1rem !important;transition:all .2s ease !important}
div[data-testid="stTabs"] [role="tab"]:hover{
  background:rgba(45,212,191,.09) !important;
  color:rgba(236,254,255,.85) !important}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  background:linear-gradient(180deg,rgba(45,212,191,.2) 0%,rgba(45,212,191,.05) 100%) !important;
  border-color:rgba(45,212,191,.3) !important;
  border-bottom:2px solid #2dd4bf !important;
  color:#5eead4 !important;
  box-shadow:0 -4px 18px rgba(45,212,191,.14) !important}

/* ── Hero ─────────────────────────────────────────────────── */
.hero{
  background:linear-gradient(135deg,#061f18 0%,#0b3b2e 45%,#0c2840 100%);
  border:1px solid rgba(45,212,191,.18);border-radius:28px;
  padding:2.6rem 3rem;position:relative;overflow:hidden;
  box-shadow:0 28px 72px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.06);
  animation:fadeInUp .55s ease-out}
.hero::before{
  content:'';position:absolute;top:0;left:0;right:0;bottom:0;
  background:
    radial-gradient(ellipse at 15% 55%,rgba(45,212,191,.09) 0%,transparent 55%),
    radial-gradient(ellipse at 85% 20%,rgba(56,189,248,.07) 0%,transparent 50%);
  pointer-events:none}
.hero::after{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent 0%,#2dd4bf 30%,#38bdf8 60%,transparent 100%);
  border-radius:28px 28px 0 0;animation:borderPulse 3s ease-in-out infinite}
.hero-title{
  font-size:3rem;font-weight:900;letter-spacing:-.05em;margin:0 0 .6rem;
  background:linear-gradient(135deg,#ffffff 0%,#a7f3d0 40%,#38bdf8 80%,#ffffff 100%);
  background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:shimmer 5s linear infinite;line-height:1.1}
.hero p{font-size:1.02rem;line-height:1.75;color:rgba(255,255,255,.78);margin-bottom:1.2rem}

/* ── Pills ───────────────────────────────────────────────── */
.pills{display:flex;flex-wrap:wrap;gap:.6rem}
.pill{
  padding:.5rem 1rem;border-radius:999px;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.13);
  color:#ecfeff;font-weight:700;font-size:.83rem;
  backdrop-filter:blur(10px);transition:all .2s ease;cursor:default}
.pill:hover{
  background:rgba(45,212,191,.14);border-color:rgba(45,212,191,.38);
  transform:translateY(-2px);box-shadow:0 4px 16px rgba(45,212,191,.18)}

/* ── KPI Cards ───────────────────────────────────────────── */
.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem}
.kpi{
  background:rgba(10,22,18,.9);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,.07);border-radius:18px;
  padding:1.1rem 1rem 1rem;
  box-shadow:0 8px 28px rgba(0,0,0,.28);
  position:relative;overflow:hidden;
  transition:all .26s cubic-bezier(.4,0,.2,1);
  animation:fadeInUp .5s ease-out both}
.kpi::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--ka,#2dd4bf),transparent);
  border-radius:18px 18px 0 0}
.kpi:hover{
  border-color:rgba(45,212,191,.28);transform:translateY(-4px);
  box-shadow:0 18px 44px rgba(0,0,0,.36),0 0 32px rgba(45,212,191,.1)}
.kpi-label{
  color:rgba(236,254,255,.58);font-size:.74rem;font-weight:700;
  margin-bottom:.38rem;letter-spacing:.04em;text-transform:uppercase}
.kpi-value{color:#fff;font-size:1.52rem;font-weight:900;letter-spacing:-.03em;line-height:1.1}
.kpi-sub{color:rgba(236,254,255,.4);font-size:.7rem;margin-top:.22rem}

/* ── General Cards ───────────────────────────────────────── */
.card{
  background:rgba(10,22,18,.88);backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,.07);border-radius:20px;
  padding:1.2rem;box-shadow:0 8px 28px rgba(0,0,0,.22);
  transition:border-color .2s,box-shadow .22s}
.card:hover{border-color:rgba(45,212,191,.16);box-shadow:0 14px 38px rgba(0,0,0,.28)}
.sec-head{font-size:1.1rem;font-weight:900;color:#fff;margin-bottom:.5rem}

/* ── Gradient Section Headings ───────────────────────────── */
.shead{
  font-size:1.12rem;font-weight:900;margin-bottom:.65rem;
  background:linear-gradient(135deg,#a7f3d0,#67e8f9);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  display:inline-block;letter-spacing:-.01em}

/* ── Live Status Pills ───────────────────────────────────── */
.lpill{
  display:inline-flex;align-items:center;gap:8px;
  padding:.6rem 1.15rem;border-radius:999px;
  font-weight:800;font-size:.82rem;
  border:1px solid rgba(255,255,255,.1);
  white-space:nowrap;letter-spacing:.04em;
  transition:all .2s ease;cursor:default}
.lpill:hover{transform:translateY(-1px);filter:brightness(1.15)}
.lp-g{background:rgba(34,197,94,.14);border-color:rgba(34,197,94,.28);box-shadow:0 0 20px rgba(34,197,94,.18)}
.lp-r{background:rgba(239,68,68,.14);border-color:rgba(239,68,68,.28);box-shadow:0 0 20px rgba(239,68,68,.18)}
.lp-y{background:rgba(245,158,11,.14);border-color:rgba(245,158,11,.28);box-shadow:0 0 20px rgba(245,158,11,.18)}
.lp-b{background:rgba(56,189,248,.14);border-color:rgba(56,189,248,.28);box-shadow:0 0 20px rgba(56,189,248,.18)}
.lp-p{background:rgba(168,85,247,.14);border-color:rgba(168,85,247,.28);box-shadow:0 0 20px rgba(168,85,247,.18)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;animation:pulse 1.5s ease-in-out infinite}
.dot-g{background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,.7)}
.dot-r{background:#ef4444;box-shadow:0 0 8px rgba(239,68,68,.7)}
.dot-y{background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,.7)}
.dot-b{background:#38bdf8;box-shadow:0 0 8px rgba(56,189,248,.7)}
.dot-p{background:#a855f7;box-shadow:0 0 8px rgba(168,85,247,.7)}

/* ── Success / Error ─────────────────────────────────────── */
.success-box{
  background:linear-gradient(135deg,rgba(20,184,166,.18),rgba(34,197,94,.12));
  border:1px solid rgba(45,212,191,.3);border-radius:16px;
  padding:1rem 1.2rem;color:#d1fae5;animation:fadeInUp .4s ease-out}
.err-box{
  background:rgba(127,29,29,.3);border:1px solid rgba(248,113,113,.3);
  border-radius:14px;padding:.9rem 1.1rem;color:#fee2e2;margin-top:.7rem}

/* ── Sidebar history cards ───────────────────────────────── */
.hcard{
  background:rgba(5,14,11,.9);
  border:1px solid rgba(45,212,191,.1);
  border-left:3px solid rgba(45,212,191,.35);
  border-radius:12px;padding:.72rem .88rem;margin-bottom:.45rem;
  transition:all .22s ease}
.hcard:hover{
  border-color:rgba(45,212,191,.25);border-left-color:#2dd4bf;
  background:rgba(9,24,18,.95);transform:translateX(2px)}
.hcard-id{font-size:.7rem;color:rgba(236,254,255,.38);font-family:'JetBrains Mono',monospace}
.hcard-cov{font-size:1rem;font-weight:900;color:#2dd4bf;margin:.12rem 0}
.hcard-conf{font-size:.74rem;font-weight:700}

/* ── Sidebar labels ──────────────────────────────────────── */
.sb-section{
  font-size:.72rem;font-weight:800;color:#5eead4;
  letter-spacing:.1em;text-transform:uppercase;
  margin:.9rem 0 .4rem;padding-left:.5rem;border-left:3px solid #2dd4bf}
.sb-box{
  background:rgba(5,14,11,.72);border:1px solid rgba(45,212,191,.09);
  border-radius:14px;padding:.85rem .9rem;margin-bottom:.65rem}

/* ── Workflow Steps ──────────────────────────────────────── */
.wf-step{
  background:rgba(10,22,18,.9);border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:.9rem .65rem;text-align:center;
  transition:all .24s ease;position:relative;overflow:hidden}
.wf-step::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#2dd4bf,transparent);
  opacity:0;transition:opacity .24s}
.wf-step:hover{
  border-color:rgba(45,212,191,.35);transform:translateY(-3px);
  box-shadow:0 10px 28px rgba(45,212,191,.14)}
.wf-step:hover::before{opacity:1}
.wf-num{
  width:32px;height:32px;border-radius:50%;
  background:rgba(45,212,191,.12);border:1.5px solid rgba(45,212,191,.3);
  display:inline-flex;align-items:center;justify-content:center;
  font-weight:900;font-size:.84rem;margin-bottom:.42rem;color:#2dd4bf;
  transition:all .24s}
.wf-step:hover .wf-num{
  background:rgba(45,212,191,.22);border-color:#2dd4bf;
  box-shadow:0 0 18px rgba(45,212,191,.4)}
.wf-label{font-size:.8rem;font-weight:700;color:#f0fdf4}

/* ── Footer ──────────────────────────────────────────────── */
.footer{
  background:rgba(10,22,18,.72);border:1px solid rgba(255,255,255,.06);
  border-radius:22px;padding:2rem 2.5rem;text-align:center;
  position:relative;overflow:hidden}
.footer::before{
  content:'';position:absolute;top:0;left:15%;right:15%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(45,212,191,.5),transparent)}

.gap{height:.9rem}

/* ── st.status() / running pipeline widget ────────────────────── */
div[data-testid="stStatusWidget"],
div[data-testid="stStatusContainer"],
div[data-testid="stStatus"]{
  background:rgba(8,20,16,.96) !important;
  border:1px solid rgba(45,212,191,.22) !important;
  border-radius:16px !important;
  color:#ecfeff !important}
div[data-testid="stStatusWidget"] > div,
div[data-testid="stStatusContainer"] > div{
  background:transparent !important;color:#ecfeff !important}
div[data-testid="stStatusWidget"] summary,
div[data-testid="stStatusContainer"] summary{
  background:rgba(8,20,16,.96) !important;
  color:#ecfeff !important;border-radius:12px !important}
div[data-testid="stStatusWidget"] p,
div[data-testid="stStatusContainer"] p{
  color:rgba(236,254,255,.82) !important;font-size:.9rem !important}
div[data-testid="stStatusWidget"] svg,
div[data-testid="stStatusContainer"] svg{color:#2dd4bf !important}

/* ── Alert / notification boxes — st.info, st.success, st.warning, st.error ─ */
div[data-baseweb="notification"]{
  background:rgba(8,20,16,.92) !important;
  border-radius:14px !important;
  border:1px solid rgba(45,212,191,.2) !important;
  backdrop-filter:blur(10px) !important}
div[data-testid="stNotification"],
div[data-testid="stAlert"]{
  background:rgba(8,20,16,.92) !important;
  border-radius:14px !important;
  border:1px solid rgba(45,212,191,.18) !important;
  backdrop-filter:blur(10px) !important}
div[data-baseweb="notification"] *,
div[data-testid="stNotification"] *,
div[data-testid="stAlert"] *{color:rgba(236,254,255,.88) !important}
/* Preserve icon tint per alert type */
div[data-testid="stAlert"][class*="info"] svg   {color:#38bdf8 !important}
div[data-testid="stAlert"][class*="success"] svg{color:#22c55e !important}
div[data-testid="stAlert"][class*="warning"] svg{color:#f59e0b !important}
div[data-testid="stAlert"][class*="error"] svg  {color:#ef4444 !important}

/* ── st.json() code viewer ───────────────────────────────────── */
div[data-testid="stJson"],
div[data-testid="stJson"] > div,
div[data-testid="stJson"] pre{
  background:rgba(5,12,10,.95) !important;
  border:1px solid rgba(45,212,191,.14) !important;
  border-radius:14px !important;color:#a7f3d0 !important}

/* ── Code / pre blocks ───────────────────────────────────────── */
code,pre{
  background:rgba(5,12,10,.92) !important;
  color:#a7f3d0 !important;border-radius:8px !important}
div[data-testid="stCode"]{
  background:rgba(5,12,10,.92) !important;
  border:1px solid rgba(45,212,191,.12) !important;
  border-radius:14px !important}

/* ── Plotly chart wrapper ─────────────────────────────────────── */
div[data-testid="stPlotlyChart"],
div[data-testid="stPlotlyChart"] > div{
  background:transparent !important;border-radius:16px !important}

/* ── Tab content panel ───────────────────────────────────────── */
div[data-testid="stTabsTabPanel"]{
  background:transparent !important;padding-top:.8rem !important}

/* ── Spinner ─────────────────────────────────────────────────── */
div[data-testid="stSpinner"] > div,
div[data-testid="stSpinner"] p{color:#2dd4bf !important}
div[data-testid="stSpinner"] svg{color:#2dd4bf !important;fill:#2dd4bf !important}

/* ── Progress bar ─────────────────────────────────────────────── */
div[data-testid="stProgress"] > div{border-radius:999px !important}
div[data-testid="stProgress"] > div > div{
  background:rgba(255,255,255,.07) !important;border-radius:999px !important}
div[data-testid="stProgress"] > div > div > div{
  background:linear-gradient(90deg,#2dd4bf,#38bdf8) !important;
  border-radius:999px !important;
  box-shadow:0 0 10px rgba(45,212,191,.4) !important}

/* ── Image containers ─────────────────────────────────────────── */
div[data-testid="stImage"] img{border-radius:12px !important}
div[data-testid="stImage"] figcaption{
  color:rgba(236,254,255,.48) !important;font-size:.78rem !important}

/* ── Column containers ────────────────────────────────────────── */
div[data-testid="column"]{background:transparent !important}

/* ── Expander with dark content area ────────────────────────── */
div[data-testid="stExpander"] details{
  background:rgba(8,18,14,.9) !important;
  border:1px solid rgba(255,255,255,.07) !important;
  border-radius:14px !important}
div[data-testid="stExpander"] details[open]{
  border-color:rgba(45,212,191,.18) !important}
div[data-testid="stExpander"] details > div{
  background:transparent !important}

/* ── Sidebar download buttons — force dark ───────────────────── */
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button{
  background:linear-gradient(135deg,rgba(6,95,70,.95),rgba(4,120,87,.95)) !important;
  color:#6ee7b7 !important;
  border:1px solid rgba(52,211,153,.28) !important}

/* ── Toast / snackbar ─────────────────────────────────────────── */
div[data-testid="stToast"]{
  background:rgba(8,22,16,.97) !important;
  border:1px solid rgba(45,212,191,.28) !important;
  border-radius:14px !important;color:#ecfeff !important;
  backdrop-filter:blur(12px) !important}

@media(max-width:1200px){.kpi-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){.kpi-grid{grid-template-columns:1fr 1fr}}
@media(max-width:550px){.kpi-grid{grid-template-columns:1fr}}
</style>""", unsafe_allow_html=True)

# ── API helpers ──────────────────────────────────────────────────────────────
def check_api_health():
    try:
        t = time.perf_counter()
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        ms = int((time.perf_counter() - t) * 1000)
        if r.status_code == 200:
            d = r.json()
            ok = str(d.get("status","")).lower() in ["healthy","online","ok"] or d.get("api","")=="online"
            return ok, d, ms
        return False, None, ms
    except Exception:
        return False, None, None

def get_model_info():
    try:
        r = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def get_api_metrics():
    try:
        r = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def call_predict_full(file_name, file_bytes, file_type):
    t = time.perf_counter()
    r = requests.post(f"{API_BASE_URL}/predict-full",
                      files={"file": (file_name, file_bytes, file_type)}, timeout=90)
    ms = int((time.perf_counter() - t) * 1000)
    if r.status_code == 200:
        return r.json(), ms
    raise Exception(f"API error {r.status_code}: {r.text[:300]}")

# ── Image helpers ─────────────────────────────────────────────────────────────
def b64_to_pil(b64str: str, mode="RGB") -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64str))).convert(mode)

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def normalise_mask(mask_img: Image.Image):
    gray = mask_img.convert("L")
    arr = np.array(gray).astype(np.uint8)
    lo, hi = int(arr.min()), int(arr.max())
    if hi > lo:
        stretched = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        stretched = arr.copy()
    thresh = 127
    bin_arr = np.where(stretched >= thresh, 255, 0).astype(np.uint8)
    if (bin_arr > 0).mean() < 0.01:
        t2 = max(25, int(np.percentile(stretched, 75)))
        bin_arr = np.where(stretched >= t2, 255, 0).astype(np.uint8)
    cov = float((bin_arr > 0).mean() * 100)
    return Image.fromarray(stretched, "L"), Image.fromarray(bin_arr, "L"), cov

def create_teal_overlay(orig: Image.Image, mask: Image.Image) -> Image.Image:
    base = orig.convert("RGB")
    try:
        m = mask.convert("L").resize(base.size, Image.Resampling.NEAREST)
    except AttributeError:
        m = mask.convert("L").resize(base.size, Image.NEAREST)
    ba = np.array(base, np.uint8); ma = np.array(m, np.uint8)
    ea = np.array(m.filter(ImageFilter.FIND_EDGES).convert("L"), np.uint8)
    ea = np.where(ea > 20, 255, 0).astype(np.uint8)
    ov = ba.copy()
    ov[ma > 0] = (0.72 * ov[ma > 0] + 0.28 * np.array([45,212,191])).astype(np.uint8)
    ov[ea > 0] = np.array([56,189,248], np.uint8)
    return Image.fromarray(ov, "RGB")

def load_uploaded_image(file_bytes, file_name) -> Image.Image:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext == "dcm":
        from src.dicom_loader import load_dicom_from_bytes
        return load_dicom_from_bytes(file_bytes)
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def make_zip(result: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if result.get("binary_mask"):
            zf.writestr("binary_mask.png", pil_to_png_bytes(result["binary_mask"]))
        if result.get("clinical_overlay"):
            zf.writestr("clinical_overlay.png", pil_to_png_bytes(result["clinical_overlay"]))
        if result.get("original_image"):
            zf.writestr("original.png", pil_to_png_bytes(result["original_image"]))
        if result.get("report"):
            zf.writestr("diagnostic_report.json",
                        json.dumps(result["report"], indent=2).encode())
    buf.seek(0); return buf.getvalue()

def load_test_metrics() -> dict:
    """Load real evaluated test metrics from results/test_metrics.json."""
    for p in [
        os.path.join(_PROJECT_ROOT, "results", "test_metrics.json"),
        os.path.join(os.path.dirname(__file__), "..", "results", "test_metrics.json"),
    ]:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f).get("metrics", {})
            except Exception:
                pass
    return {}


def load_training_data() -> list:
    """Load epoch-wise training metrics from training_log.csv."""
    for p in [
        os.path.join(_PROJECT_ROOT, "training_log.csv"),
        os.path.join(_PROJECT_ROOT, "models", "training_log.csv"),
    ]:
        if os.path.exists(p):
            try:
                rows = []
                with open(p, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            rows.append({
                                "epoch":    int(float(row.get("epoch", 0))),
                                "dice":     float(row.get("dice_coefficient", 0)),
                                "val_dice": float(row.get("val_dice_coefficient", 0)),
                                "iou":      float(row.get("iou_score", 0)),
                                "val_iou":  float(row.get("val_iou_score", 0)),
                                "lr":       float(row.get("learning_rate", 1e-4)),
                            })
                        except Exception:
                            continue
                return rows
            except Exception:
                pass
    return []


def conf_color(label):
    return {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}.get(label, "#94a3b8")

# ── Plotly chart helpers ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ecfeff", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def gauge_chart(value, title, max_val=100, color="#2dd4bf"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 14, "color": "#ecfeff"}},
        number={"suffix": "%", "font": {"size": 28, "color": "#fff"}},
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#ecfeff"),
            bar=dict(color=color),
            bgcolor="rgba(255,255,255,.05)",
            bordercolor="rgba(255,255,255,.1)",
            steps=[dict(range=[0, max_val], color="rgba(255,255,255,.03)")]
        )
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig

def bar_chart(left, right):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Left Lung", "Right Lung"], y=[left, right],
                         marker_color=["#2dd4bf", "#38bdf8"],
                         text=[f"{left}%", f"{right}%"], textposition="outside",
                         textfont=dict(color="#fff", size=14)))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      yaxis=dict(range=[0, 70], gridcolor="rgba(255,255,255,.06)"),
                      bargap=0.35)
    return fig

def radar_chart(dice, iou, coverage, conf_score):
    cats = ["Dice Score", "IoU Score", "Coverage Norm", "Confidence"]
    dice_v = float(str(dice).replace("~","").replace(">","")) if isinstance(dice, str) else float(dice)
    iou_v  = float(str(iou).replace("~","").replace(">","")) if isinstance(iou, str) else float(iou)
    vals = [dice_v * 100, iou_v * 100, min(coverage * 2, 100), conf_score * 100]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                    fill="toself",
                                    fillcolor="rgba(45,212,191,.15)",
                                    line=dict(color="#2dd4bf", width=2)))
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True, range=[0,100],
                                                 gridcolor="rgba(255,255,255,.1)",
                                                 tickcolor="#ecfeff"),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,.1)")))
    return fig

def training_curve_chart(training_data: list):
    """Dual-line Plotly chart: train Dice vs val Dice over epochs."""
    if not training_data or len(training_data) < 2:
        return None
    epochs   = [r["epoch"] + 1 for r in training_data]
    dice     = [r["dice"]     for r in training_data]
    val_dice = [r["val_dice"] for r in training_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=dice, name="Train Dice",
        mode="lines+markers",
        line=dict(color="#2dd4bf", width=2.5),
        marker=dict(size=7, color="#2dd4bf"),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_dice, name="Val Dice",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5, dash="dot"),
        marker=dict(size=7, color="#f59e0b"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,.06)", dtick=1),
        yaxis=dict(title="Dice Score", gridcolor="rgba(255,255,255,.06)", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#ecfeff", size=12)),
    )
    return fig


def history_line_chart(history: list):
    if len(history) < 2:
        return None
    x = [f"#{i+1}" for i in range(len(history))]
    y = [h.get("mask_coverage", 0) for h in history]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers",
                               line=dict(color="#2dd4bf", width=2.5),
                               marker=dict(color="#38bdf8", size=7),
                               fill="tozeroy",
                               fillcolor="rgba(45,212,191,.08)"))
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
                      xaxis=dict(gridcolor="rgba(255,255,255,.06)"),
                      yaxis=dict(gridcolor="rgba(255,255,255,.06)", title="Coverage %"))
    return fig

# ── API state (evaluated once per run) ───────────────────────────────────────
api_ok, health_data, health_latency_ms = check_api_health()
model_info   = get_model_info() if api_ok else None
api_metrics  = get_api_metrics() if api_ok else None

model_name    = (model_info or {}).get("model_name", "Attention U-Net (Lung)")
framework     = (model_info or {}).get("framework",   "TensorFlow / Keras")
model_version = (model_info or {}).get("model_version", "v3.0.0")
model_ready   = bool(model_info and model_info.get("status",""))

# Load real evaluated metrics from evaluate.py output
_test_metrics = load_test_metrics()
if _test_metrics:
    dice_score    = f"{_test_metrics.get('dice_score', 0):.4f}"
    iou_score     = f"{_test_metrics.get('iou_score', 0):.4f}"
    _f1_score     = f"{_test_metrics.get('f1_score', 0):.4f}"
    _precision    = f"{_test_metrics.get('precision', 0):.4f}"
    _recall       = f"{_test_metrics.get('recall', 0):.4f}"
    _metrics_src  = "📊 Test Set Evaluated"
else:
    _api_m        = (model_info or {}).get("metrics", {})
    dice_score    = _api_m.get("dice_score", "N/A")
    iou_score     = _api_m.get("iou_score",  "N/A")
    _f1_score     = "—"
    _precision    = "—"
    _recall       = "—"
    _metrics_src  = "API model info"

# ── Premium Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Brand header ─────────────────────────────────────────────────────────
    st.markdown("""<div style='text-align:center;padding:1rem 0 .6rem'>
      <div style='font-size:2.4rem;margin-bottom:.2rem;line-height:1'>🫁</div>
      <div style='font-size:1.55rem;font-weight:900;letter-spacing:-.03em;
        background:linear-gradient(135deg,#ffffff 0%,#a7f3d0 50%,#67e8f9 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text'>
        PulmoVision
      </div>
      <div style='font-size:.68rem;color:rgba(45,212,191,.72);font-weight:700;
        letter-spacing:.12em;text-transform:uppercase;margin-top:.18rem'>AI Segmentation Suite</div>
      <div style='margin-top:.55rem;display:inline-flex;align-items:center;gap:.4rem;
        background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.22);
        border-radius:999px;padding:.26rem .7rem;font-size:.67rem;font-weight:800;color:#86efac'>
        <span style='width:6px;height:6px;border-radius:50%;background:#22c55e;
          display:inline-block;box-shadow:0 0 8px rgba(34,197,94,.8)'></span>SYSTEM ACTIVE
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(45,212,191,.12);margin:.3rem 0 .8rem'>", unsafe_allow_html=True)

    # ── 5. Dark / Light Mode Toggle ───────────────────────────────────────────
    dm_col, dm_lbl = st.columns([1, 3])
    with dm_col:
        dark = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="dm_toggle",
                         label_visibility="collapsed")
        st.session_state.dark_mode = dark
    with dm_lbl:
        mode_txt = "🌙 Dark Mode" if dark else "☀️ Light Mode"
        st.markdown(f"<div style='font-size:.88rem;font-weight:700;color:#ecfeff;padding-top:.3rem'>{mode_txt}</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,.06);margin:.5rem 0'>", unsafe_allow_html=True)

    # ── 2. Model Switcher ─────────────────────────────────────────────────────
    st.markdown("<div class='sb-section'>🧠 Model Switcher</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-box'>", unsafe_allow_html=True)
    MODEL_OPTIONS = [
        "Attention U-Net (Current)",
        "Basic U-Net",
        "ResU-Net (Coming Soon)",
        "Swin-UNet (Coming Soon)",
    ]
    sel_model = st.selectbox("Select Model", MODEL_OPTIONS,
                             index=MODEL_OPTIONS.index(st.session_state.selected_model)
                             if st.session_state.selected_model in MODEL_OPTIONS else 0,
                             key="model_sel", label_visibility="collapsed")
    st.session_state.selected_model = sel_model
    if "Coming Soon" in sel_model:
        st.caption("⚠️ Not yet deployed — Attention U-Net is active")
    else:
        st.caption(f"✅ Active: {sel_model}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Upload Mode ────────────────────────────────────────────────────────
    st.markdown("<div class='sb-section'>📁 Upload Mode</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-box'>", unsafe_allow_html=True)
    UPLOAD_MODES = ["Image Mode (PNG/JPG)", "DICOM Mode (.dcm)"]
    up_mode = st.radio("Upload Mode", UPLOAD_MODES,
                       index=UPLOAD_MODES.index(st.session_state.upload_type)
                       if st.session_state.upload_type in UPLOAD_MODES else 0,
                       key="upload_mode_radio", label_visibility="collapsed")
    st.session_state.upload_type = up_mode
    st.caption("🏥 DICOM — upload .dcm files" if "DICOM" in up_mode else "🖼️ Standard PNG / JPG images")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 1. Patient / Case History ─────────────────────────────────────────────
    st.markdown("<div class='sb-section'>📋 Patient / Case History</div>", unsafe_allow_html=True)
    history = st.session_state.history
    if not history:
        st.markdown("<div class='sb-box'><div style='color:rgba(236,254,255,.4);font-size:.82rem;text-align:center;padding:.5rem 0'>" 
                    "No scans yet.<br>Upload a radiograph to begin.</div></div>", unsafe_allow_html=True)
    else:
        fig_hist = history_line_chart(history)
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        for i, h in enumerate(reversed(history[-8:])):
            conf_c = conf_color(h.get("confidence", "Low"))
            ridx = len(history) - 1 - i
            ts = h.get("timestamp", "")
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "—"
            col_h, col_btn = st.columns([3, 1])
            with col_h:
                st.markdown(f"""<div class='hcard'>
                  <div class='hcard-id'>#{ridx+1} · {h.get('filename','—')[:20]}</div>
                  <div style='font-size:.7rem;color:rgba(236,254,255,.38);margin:.05rem 0'>{ts_str}</div>
                  <div class='hcard-cov'>{h.get('mask_coverage',0):.1f}% · Dice {h.get('dice','~0.95')}</div>
                  <div class='hcard-conf' style='color:{conf_c}'>● {h.get('confidence','—')} · {h.get('elapsed_ms',0)} ms</div>
                </div>""", unsafe_allow_html=True)
            with col_btn:
                if st.button("↺", key=f"reload_{ridx}", help="Reload this result"):
                    st.session_state.reload_result_idx = ridx

    # ── 4. Analytics Summary ──────────────────────────────────────────────────
    st.markdown("<div class='sb-section'>📈 Analytics</div>", unsafe_allow_html=True)
    total_scans = len(history)
    avg_lat = round(sum(h.get("elapsed_ms",0) for h in history) / max(total_scans,1))
    avg_cov = round(sum(h.get("mask_coverage",0) for h in history) / max(total_scans,1), 1)
    srv_sr  = api_metrics.get("success_rate_percent", 100) if api_metrics else 100
    srv_up  = api_metrics.get("uptime_seconds", 0) // 60 if api_metrics else 0
    st.markdown(f"""<div class='sb-box'>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:.4rem'>
        <div style='text-align:center;padding:.5rem .3rem;background:rgba(45,212,191,.07);border-radius:10px;border:1px solid rgba(45,212,191,.12)'>
          <div style='font-size:1.25rem;font-weight:900;color:#2dd4bf'>{total_scans}</div>
          <div style='font-size:.68rem;color:rgba(236,254,255,.5)'>Total Scans</div>
        </div>
        <div style='text-align:center;padding:.5rem .3rem;background:rgba(56,189,248,.07);border-radius:10px;border:1px solid rgba(56,189,248,.12)'>
          <div style='font-size:1.25rem;font-weight:900;color:#38bdf8'>{avg_cov}%</div>
          <div style='font-size:.68rem;color:rgba(236,254,255,.5)'>Avg Coverage</div>
        </div>
        <div style='text-align:center;padding:.5rem .3rem;background:rgba(168,85,247,.07);border-radius:10px;border:1px solid rgba(168,85,247,.12)'>
          <div style='font-size:1.25rem;font-weight:900;color:#a78bfa'>{avg_lat} ms</div>
          <div style='font-size:.68rem;color:rgba(236,254,255,.5)'>Avg Latency</div>
        </div>
        <div style='text-align:center;padding:.5rem .3rem;background:rgba(34,197,94,.07);border-radius:10px;border:1px solid rgba(34,197,94,.12)'>
          <div style='font-size:1.25rem;font-weight:900;color:#22c55e'>{srv_sr:.0f}%</div>
          <div style='font-size:.68rem;color:rgba(236,254,255,.5)'>API Success</div>
        </div>
      </div>
      <div style='margin-top:.5rem;font-size:.75rem;color:rgba(236,254,255,.45);text-align:center'>
        API uptime {srv_up} min · endpoint {API_BASE_URL.replace("http://","")}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 6. Quick Export ───────────────────────────────────────────────────────
    st.markdown("<div class='sb-section'>📥 Quick Export</div>", unsafe_allow_html=True)
    res = st.session_state.prediction_result
    if res:
        from src.report_generator import report_to_json_bytes, report_to_csv_bytes
        pid_s = str(res.get("prediction_id","out"))[:8]
        st.download_button("⬇️ Binary Mask PNG", data=pil_to_png_bytes(res["binary_mask"]),
                           file_name=f"mask_{pid_s}.png", mime="image/png",
                           use_container_width=True, key="sb_mask")
        st.download_button("⬇️ Clinical Overlay PNG", data=pil_to_png_bytes(res["clinical_overlay"]),
                           file_name=f"overlay_{pid_s}.png", mime="image/png",
                           use_container_width=True, key="sb_overlay")
        st.download_button("📄 JSON Diagnostic Report", data=report_to_json_bytes(res["report"]),
                           file_name=f"report_{pid_s}.json", mime="application/json",
                           use_container_width=True, key="sb_json")
    else:
        st.markdown("<div class='sb-box'><div style='color:rgba(236,254,255,.35);font-size:.8rem;text-align:center;padding:.4rem 0'>"
                    "Run a segmentation to<br>enable exports.</div></div>", unsafe_allow_html=True)

    # ── 7. Error Logs ─────────────────────────────────────────────────────────
    with st.expander("🪲 Error Logs", expanded=False):
        err_logs = st.session_state.error_logs
        if not err_logs:
            st.markdown("<div style='color:rgba(236,254,255,.38);font-size:.8rem;padding:.3rem 0'>"
                        "No errors this session. ✅</div>", unsafe_allow_html=True)
        else:
            for entry in reversed(err_logs[-10:]):
                ts_s = time.strftime("%H:%M:%S", time.localtime(entry.get("ts", 0)))
                st.markdown(f"<div style='font-size:.75rem;color:#fca5a5;padding:.22rem 0;"
                            f"border-bottom:1px solid rgba(255,255,255,.05)'>"
                            f"[{ts_s}] {entry.get('msg','—')}</div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Logs", use_container_width=True, key="clear_logs"):
            st.session_state.error_logs = []

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(f"""<div class='hero'>
  <div style='position:relative;z-index:1'>
    <div class='hero-title'>🫁 PulmoVision AI</div>
    <p>Enterprise-grade chest radiograph lung segmentation · Attention U-Net architecture ·
    Real-time clinical diagnostics · DICOM support · Plotly analytics dashboard</p>
    <div class='pills'>
      <div class='pill'>🧠 {model_name}</div>
      <div class='pill'>🛰️ Live API Monitor</div>
      <div class='pill'>⚡ Single-Call Inference</div>
      <div class='pill'>🏥 DICOM Ready</div>
      <div class='pill'>📊 Plotly Analytics</div>
      <div class='pill'>🏷️ {model_version}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='gap'></div>", unsafe_allow_html=True)

# ── Live status pills ─────────────────────────────────────────────────────────
a_cls  = "lp-g" if api_ok    else "lp-r"
a_dot  = "dot-g" if api_ok   else "dot-r"
a_txt  = "API LIVE"          if api_ok else "API OFFLINE"
m_cls  = "lp-g" if model_ready else "lp-r"
m_dot  = "dot-g" if model_ready else "dot-r"
m_txt  = "MODEL READY"       if model_ready else "MODEL OFFLINE"
lat_v  = f"PING {health_latency_ms} ms" if health_latency_ms else "PING N/A"
lat_cls = "lp-g" if (health_latency_ms and health_latency_ms < 1000) else "lp-y"
lat_dot = "dot-g" if (health_latency_ms and health_latency_ms < 1000) else "dot-y"

pill_col, refresh_col = st.columns([5, 1])
with pill_col:
    st.markdown(f"""<div style='display:flex;flex-wrap:wrap;gap:.7rem;margin-bottom:1.1rem'>
      <div class='lpill {a_cls}'><span class='dot {a_dot}'></span>{a_txt}</div>
      <div class='lpill {m_cls}'><span class='dot {m_dot}'></span>{m_txt}</div>
      <div class='lpill lp-b'><span class='dot dot-b'></span>AUTO RETRY</div>
      <div class='lpill {lat_cls}'><span class='dot {lat_dot}'></span>{lat_v}</div>
    </div>""", unsafe_allow_html=True)
with refresh_col:
    st.markdown("<div style='padding-top:.2rem'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh API", use_container_width=True, key="refresh_api"):
        st.rerun()

# ── KPI strip ─────────────────────────────────────────────────────────────────
avg_inf = f"{st.session_state.last_latency_ms} ms" if st.session_state.last_latency_ms else "N/A"
ping_v  = f"{health_latency_ms} ms" if health_latency_ms else "N/A"
sr      = f"{api_metrics.get('success_rate_percent',100):.0f}%" if api_metrics else "N/A"
uptime_m = f"{api_metrics.get('uptime_seconds',0)//60} min" if api_metrics else "N/A"

st.markdown(f"""<div class='kpi-grid'>
  <div class='kpi' style='--ka:#2dd4bf'><div class='kpi-label'>🎯 Dice Score</div>
    <div class='kpi-value'>{dice_score}</div><div class='kpi-sub'>{_metrics_src}</div></div>
  <div class='kpi' style='--ka:#38bdf8'><div class='kpi-label'>📐 IoU Score</div>
    <div class='kpi-value'>{iou_score}</div><div class='kpi-sub'>Intersection over union</div></div>
  <div class='kpi' style='--ka:#a78bfa'><div class='kpi-label'>⚡ Last Inference</div>
    <div class='kpi-value'>{avg_inf}</div><div class='kpi-sub'>Single-call latency</div></div>
  <div class='kpi' style='--ka:#f59e0b'><div class='kpi-label'>🛰️ API Ping</div>
    <div class='kpi-value'>{ping_v}</div><div class='kpi-sub'>Health check round-trip</div></div>
  <div class='kpi' style='--ka:#22c55e'><div class='kpi-label'>✅ API Success Rate</div>
    <div class='kpi-value' style='color:#22c55e'>{sr}</div><div class='kpi-sub'>Uptime {uptime_m}</div></div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='gap'></div>", unsafe_allow_html=True)

# ── Workflow strip ────────────────────────────────────────────────────────────
st.markdown("""<div class='shead'>Inference Workspace</div>
<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:.8rem;margin-bottom:1.2rem'>""" +
"".join([f"""<div class='wf-step'>
  <div class='wf-num'>{n}</div>
  <div class='wf-label'>{lbl}</div></div>"""
for n, lbl in [("1","📤 Upload"),("2","🏥 DICOM/RGB"),("3","🧼 Preprocess"),
               ("4","🫁 Segment"),("5","🩻 Overlay"),("6","📊 Analyse")]]) +
"</div>", unsafe_allow_html=True)

# ── Upload + inference ────────────────────────────────────────────────────────
up_col, status_col = st.columns([1.2, 1], gap="large")

with up_col:
    st.markdown("<div class='card'><div class='sec-head'>📤 Image Submission</div>"
                "<div style='font-size:.92rem;color:rgba(236,254,255,.75);line-height:1.65'>"
                "Upload a chest radiograph (PNG · JPG · JPEG · DCM). The system runs a single "
                "optimised inference call returning mask, overlay and full diagnostics.</div></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload radiograph",
                                     type=["png","jpg","jpeg","dcm"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        cur_bytes = uploaded_file.getvalue()
        sig = f"{uploaded_file.name}_{len(cur_bytes)}"
        if sig != st.session_state.last_uploaded_signature:
            for k in ["prediction_result","timeline_steps","stream_done","last_error","last_latency_ms"]:
                st.session_state[k] = [] if k == "timeline_steps" else (False if k == "stream_done" else None)
            st.session_state.last_uploaded_signature = sig
            st.session_state.uploaded_bytes = cur_bytes
            st.session_state.uploaded_type  = uploaded_file.type or "image/png"
            st.session_state.uploaded_name  = uploaded_file.name
            try:
                st.session_state.original_image = load_uploaded_image(cur_bytes, uploaded_file.name)
            except Exception as e:
                st.session_state.original_image = None
                st.session_state.last_error = str(e)

    can_run = (st.session_state.uploaded_bytes is not None
               and st.session_state.original_image is not None
               and api_ok and model_ready)
    run_btn = st.button("🚀 Run Segmentation", disabled=not can_run, use_container_width=True)

with status_col:
    a_sc = "status-ready" if api_ok else "status-offline"
    m_sc = "status-ready" if model_ready else "status-offline"
    g_style = lambda ok: ("linear-gradient(135deg,rgba(18,120,74,.92),rgba(13,98,60,.92))" if ok
                          else "linear-gradient(135deg,rgba(70,80,90,.9),rgba(50,60,70,.9))")
    st.markdown(f"""<div class='card'>
      <div class='sec-head'>🧾 System Status</div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin:.7rem 0'>
        <div style='background:{g_style(api_ok)};border-radius:14px;padding:.9rem'>
          <div style='font-size:.8rem;font-weight:700;color:rgba(240,253,244,.8)'>API</div>
          <div style='font-size:1.6rem;font-weight:900;color:#fff'>{'Ready' if api_ok else 'Offline'}</div>
        </div>
        <div style='background:{g_style(model_ready)};border-radius:14px;padding:.9rem'>
          <div style='font-size:.8rem;font-weight:700;color:rgba(240,253,244,.8)'>Model</div>
          <div style='font-size:1.6rem;font-weight:900;color:#fff'>{'Ready' if model_ready else 'Offline'}</div>
        </div>
      </div>
      <div style='font-size:.88rem;line-height:1.75;color:rgba(236,254,255,.78)'>
        <b>Architecture:</b> {model_name}<br>
        <b>Framework:</b> {framework}<br>
        <b>Dice:</b> {dice_score} &nbsp;|&nbsp; <b>IoU:</b> {iou_score}<br>
        <b>Endpoint:</b> /predict-full (single-call)<br>
        <b>Formats:</b> PNG · JPG · JPEG · DCM
      </div></div>""", unsafe_allow_html=True)

if st.session_state.last_error:
    st.markdown(f"<div class='err-box'>⚠️ {st.session_state.last_error}</div>",
                unsafe_allow_html=True)

# ── Inference execution ───────────────────────────────────────────────────────
if run_btn:
    with st.status("🚀 Running PulmoVision AI inference pipeline...", expanded=True) as _inf_status:
        _inf_status.write("📤 Upload received — radiograph queued for inference")
        _inf_status.write("🛰️ Connecting to /predict-full endpoint...")
        try:
            data, req_ms = call_predict_full(
                st.session_state.uploaded_name,
                st.session_state.uploaded_bytes,
                st.session_state.uploaded_type,
            )
            _inf_status.write("🧼 Preprocessing — resize · normalise · channel alignment")
            _inf_status.write("🫁 Attention U-Net — segmentation mask generated")
            _inf_status.write("🩻 Overlay rendered — clinical teal fill + cyan boundary")
            _inf_status.write("📊 Diagnostics computed — coverage · confidence · QC check")

            mask_img    = b64_to_pil(data["mask_png_b64"], "L")
            overlay_img = b64_to_pil(data["overlay_png_b64"], "RGB")
            _, binary_mask, comp_cov = normalise_mask(mask_img)

            if not overlay_img:
                overlay_img = create_teal_overlay(st.session_state.original_image, binary_mask)

            cov     = float(data.get("mask_coverage_percent", comp_cov))
            conf    = data.get("confidence_label", "Low")
            c_score = float(data.get("confidence_score", 0.43))

            from src.report_generator import build_json_report
            report = build_json_report(
                prediction_id=data.get("prediction_id","N/A"),
                filename=st.session_state.uploaded_name,
                mask_coverage=cov,
                confidence_label=conf,
                confidence_score=c_score,
                left_lung_percent=float(data.get("left_lung_percent",0)),
                right_lung_percent=float(data.get("right_lung_percent",0)),
                anatomy_balance=data.get("anatomy_balance","N/A"),
                quality_check=data.get("quality_check","N/A"),
                processing_time_ms=int(data.get("processing_time_ms", req_ms)),
                input_resolution=data.get("input_resolution",[256,256]),
                pipeline_stages=data.get("pipeline_stages",[]),
                inference_message=data.get("message",""),
                dice_score=str(dice_score), iou_score=str(iou_score),
            )

            result_obj = {
                "data": data,
                "original_image": st.session_state.original_image,
                "binary_mask": binary_mask,
                "clinical_overlay": overlay_img,
                "mask_coverage": round(cov, 1),
                "confidence": conf,
                "confidence_score": c_score,
                "left_percent": float(data.get("left_lung_percent",0)),
                "right_percent": float(data.get("right_lung_percent",0)),
                "anatomy": data.get("anatomy_balance","N/A"),
                "qc": data.get("quality_check","N/A"),
                "elapsed_ms": int(data.get("processing_time_ms", req_ms)),
                "request_latency_ms": req_ms,
                "prediction_id": data.get("prediction_id","N/A"),
                "pipeline_stages": data.get("pipeline_stages",[]),
                "filename": st.session_state.uploaded_name,
                "report": report,
            }
            st.session_state.prediction_result = result_obj
            st.session_state.last_latency_ms   = req_ms
            st.session_state.stream_done        = True
            st.session_state.last_error         = None
            st.session_state.history.append({
                "filename":      result_obj["filename"],
                "mask_coverage": result_obj["mask_coverage"],
                "confidence":    result_obj["confidence"],
                "elapsed_ms":    result_obj["elapsed_ms"],
                "dice":          str(dice_score),
                "timestamp":     time.time(),
            })
            _inf_status.update(
                label=f"✅ Segmentation complete!  Coverage {round(cov,1)}%  ·  {req_ms} ms",
                state="complete", expanded=False,
            )
        except Exception as e:
            _inf_status.update(label="❌ Inference failed — is the FastAPI server running?", state="error")
            st.session_state.prediction_result = None
            st.session_state.last_error = str(e)
            st.session_state.error_logs.append({"ts": time.time(), "msg": str(e)})

if st.session_state.last_error and not st.session_state.prediction_result:
    st.markdown(f"<div class='err-box'><b>Inference error:</b> {st.session_state.last_error}</div>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS — 6 TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='gap'></div>", unsafe_allow_html=True)
st.markdown("<div class='shead'>Segmentation Results</div>", unsafe_allow_html=True)

result = st.session_state.prediction_result
tabs = st.tabs(["🖼️ Original","🫁 Binary Mask","🩻 Clinical Overlay",
                "📊 Diagnostics","🧠 AI Analysis","📥 Export",
                "🔄 Compare","📈 Training Curve"])

# ── Tab 0: Original ───────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### Input Radiograph")
    st.caption("Source image · auto-contrast enhanced for display")
    if result and result.get("original_image"):
        c1, c2 = st.columns([2,1])
        with c1:
            st.image(result["original_image"])
        with c2:
            fn = result.get("filename","—")
            ext = fn.rsplit(".",1)[-1].upper() if "." in fn else "—"
            ori = result["original_image"]
            st.markdown(f"""<div class='card'>
              <div class='sec-head'>🏷️ Image Info</div>
              <div style='font-size:.9rem;line-height:1.8;color:rgba(236,254,255,.82)'>
                <b>File:</b> {fn}<br><b>Format:</b> {ext}<br>
                <b>Size:</b> {ori.width} × {ori.height} px<br>
                <b>Mode:</b> {ori.mode}
              </div></div>""", unsafe_allow_html=True)
    elif st.session_state.original_image:
        st.image(st.session_state.original_image)
    else:
        st.info("Upload a chest radiograph to preview it here.")

# ── Tab 1: Binary Mask ────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Predicted Lung Mask")
    st.caption("Binary segmentation mask · white = lung region")
    if result and result.get("binary_mask"):
        c1, c2 = st.columns([2,1])
        with c1:
            st.image(result["binary_mask"])
        with c2:
            cov = result["mask_coverage"]
            conf = result["confidence"]
            cc = conf_color(conf)
            st.markdown(f"""<div class='card'>
              <div class='sec-head'>📏 Mask Stats</div>
              <div style='font-size:.9rem;line-height:1.85;color:rgba(236,254,255,.82)'>
                <b>Coverage:</b> <span style='color:#2dd4bf;font-weight:900'>{cov}%</span><br>
                <b>Confidence:</b> <span style='color:{cc};font-weight:800'>{conf}</span><br>
                <b>QC Status:</b> {result['qc']}<br>
                <b>Left Lung:</b> {result['left_percent']}%<br>
                <b>Right Lung:</b> {result['right_percent']}%<br>
                <b>Symmetry:</b> {result['anatomy']}
              </div></div>""", unsafe_allow_html=True)
    else:
        st.info("Run segmentation to generate the lung mask.")

# ── Tab 2: Clinical Overlay ───────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Clinical Overlay")
    st.caption("Teal fill · cyan boundary emphasis · backend-rendered or local fallback")
    if result and result.get("clinical_overlay"):
        c1, c2 = st.columns([2,1])
        with c1:
            st.image(result["clinical_overlay"])
        with c2:
            pid = str(result.get("prediction_id","N/A"))
            st.markdown(f"""<div class='card'>
              <div class='sec-head'>🩺 Prediction Info</div>
              <div style='font-size:.88rem;line-height:1.85;color:rgba(236,254,255,.82)'>
                <b>ID:</b> <span style='font-family:monospace;font-size:.82rem'>{pid[:8]}</span><br>
                <b>Processing:</b> {result['elapsed_ms']} ms<br>
                <b>API Latency:</b> {result['request_latency_ms']} ms<br>
                <b>Coverage:</b> {result['mask_coverage']}%<br>
                <b>Stages:</b> {len(result['pipeline_stages'])} completed
              </div></div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            for s in result.get("pipeline_stages", []):
                st.markdown(f"<div style='font-size:.82rem;color:rgba(45,212,191,.9);padding:.2rem 0'>"
                            f"✓ {str(s).replace('_',' ').title()}</div>", unsafe_allow_html=True)
    else:
        st.info("Clinical overlay will appear here after segmentation.")

# ── Tab 3: Diagnostics (Plotly) ───────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### Clinical Diagnostics")
    if result:
        cov  = result["mask_coverage"]
        left = result["left_percent"]
        rght = result["right_percent"]
        cscore = result["confidence_score"]

        # Row 1: Gauge + Bar + Radar
        gc1, gc2, gc3 = st.columns(3, gap="medium")
        with gc1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(cov, "Lung Coverage %", color=conf_color(result["confidence"])),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with gc2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.plotly_chart(bar_chart(left, rght),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with gc3:
            dice_v = float(str(dice_score).replace("~","").replace(">","")) if dice_score else 0.95
            iou_v  = float(str(iou_score).replace("~","").replace(">","")) if iou_score else 0.90
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.plotly_chart(radar_chart(dice_v, iou_v, cov, cscore),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Row 2: Full metric grid
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        mc = [
            ("🆔","Prediction ID", str(result['prediction_id'])[:8]),
            ("🎯","Coverage",      f"{cov}%"),
            ("💡","Confidence",    result['confidence']),
            ("⚖️","Symmetry",      result['anatomy']),
            ("✅","QC Status",     result['qc']),
            ("⬅️","Left Lung",     f"{left}%"),
            ("➡️","Right Lung",    f"{rght}%"),
            ("⏱️","Inference",     f"{result['elapsed_ms']} ms"),
        ]
        cols = st.columns(4, gap="medium")
        for i, (ico, lbl, val) in enumerate(mc):
            with cols[i % 4]:
                st.markdown(f"""<div class='card' style='margin-bottom:.7rem'>
                  <div class='kpi-label'>{ico} {lbl}</div>
                  <div class='kpi-value' style='font-size:1.35rem'>{val}</div>
                </div>""", unsafe_allow_html=True)

        # Confidence score bar
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        conf_pct = int(cscore * 100)
        bar_col = conf_color(result["confidence"])
        # Gradient fill: red → amber → green based on confidence level
        if conf_pct >= 70:
            bar_grad = "linear-gradient(90deg,#ef4444 0%,#f59e0b 40%,#22c55e 100%)"
        elif conf_pct >= 40:
            bar_grad = "linear-gradient(90deg,#ef4444 0%,#f59e0b 100%)"
        else:
            bar_grad = "#ef4444"
        st.markdown(f"""<div class='card'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem'>
            <span style='font-weight:800;font-size:.95rem'>Model Confidence Score</span>
            <span style='font-weight:900;font-size:1.25rem;color:{bar_col}'>{conf_pct}%</span>
          </div>
          <div style='background:rgba(255,255,255,.07);border-radius:999px;height:12px;overflow:hidden'>
            <div style='width:{conf_pct}%;height:12px;border-radius:999px;
            background:{bar_grad};box-shadow:0 0 12px rgba(45,212,191,.35);transition:width .6s ease-out'></div>
          </div>
          <div style='display:flex;justify-content:space-between;margin-top:.45rem;
            font-size:.7rem;color:rgba(236,254,255,.35);padding:0 2px'>
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
          </div>
          <div style='font-size:.82rem;color:rgba(236,254,255,.5);margin-top:.4rem'>
            Mask coverage heuristics · <b style='color:{bar_col}'>{result["confidence"]}</b> confidence tier
          </div></div>""", unsafe_allow_html=True)

        # ── Clinical interpretation ───────────────────────────────────────────
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        cov_interp = (
            "within normal physiological range (15–45%)"
            if 15 <= cov <= 45 else
            "below expected range — may indicate small/excluded lung field"
            if cov < 15 else
            "above expected range — may include non-lung structures"
        )
        sym_map = {
            "Symmetric":      "✅ Both lung fields appear evenly represented",
            "Mild asymmetry": "⚠️ Slight asymmetry — may be positional or anatomical",
            "Asymmetric":     "🔴 Significant asymmetry — clinical correlation recommended",
            "Indeterminate":  "❓ Symmetry could not be determined from mask",
        }
        sym_interp = sym_map.get(result['anatomy'], result['anatomy'])
        qc_col2    = "#22c55e" if result['qc'] == "Passed" else ("#f59e0b" if "review" in result['qc'].lower() else "#ef4444")
        st.markdown(f"""<div class='card'>
          <div class='sec-head'>🩺 Clinical Interpretation</div>
          <div style='font-size:.9rem;line-height:1.95;color:rgba(236,254,255,.82)'>
            <b>Coverage ({cov}%):</b> <span style='color:#2dd4bf'>{cov_interp}</span><br>
            <b>Bilateral anatomy:</b> {sym_interp}<br>
            <b>Left / Right split:</b> {result['left_percent']}% · {result['right_percent']}%<br>
            <b>QC outcome:</b> <span style='color:{qc_col2};font-weight:800'>{result['qc']}</span><br>
            <b>Confidence basis:</b> Coverage heuristic — High (18–45%), Medium (10–55%), Low (outside)
          </div>
          <div style='margin-top:.7rem;padding:.55rem .8rem;background:rgba(239,68,68,.06);
            border-left:3px solid rgba(239,68,68,.3);border-radius:0 8px 8px 0;
            font-size:.78rem;color:rgba(236,254,255,.48)'>
            ⚕️ Research demonstration only · Not for direct clinical use without medical validation.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Run segmentation to populate Plotly diagnostics charts.")

# ── Tab 4: AI Analysis ────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("#### AI Model Analysis")
    if result:
        ac1, ac2 = st.columns(2, gap="large")
        with ac1:
            st.markdown(f"""<div class='card'>
              <div class='sec-head'>🧠 Model Explainability</div>
              <div style='font-size:.9rem;line-height:1.85;color:rgba(236,254,255,.82)'>
                <b>Architecture:</b> Attention U-Net<br>
                <b>Attention Gates:</b> 4 decoder stages with spatial attention<br>
                <b>Dropout Rate:</b> 0.3 (training regularisation)<br>
                <b>Filter Progression:</b> 32 → 64 → 128 → 256 → 512<br>
                <b>Input Shape:</b> 256 × 256 × 1 (grayscale)<br>
                <b>Output:</b> 1 channel sigmoid probability map<br>
                <b>Loss Function:</b> BCE + Dice (composite)<br>
                <b>Optimiser:</b> Adam with LR scheduling<br>
                <b>Framework:</b> {framework}<br>
                <b>Version:</b> {model_version}
              </div></div>""", unsafe_allow_html=True)
        with ac2:
            st.markdown(f"""<div class='card'>
              <div class='sec-head'>📈 Inference Diagnostics</div>
              <div style='font-size:.9rem;line-height:1.85;color:rgba(236,254,255,.82)'>
                <b>Inference Message:</b> {result['data'].get('message','—')}<br>
                <b>Pipeline Stages:</b> {len(result['pipeline_stages'])}<br>
                <b>Input Resolution:</b> {result['data'].get('input_resolution','—')}<br>
                <b>Mask Resolution:</b> {result['data'].get('mask_resolution','—')}<br>
                <b>Processing Time:</b> {result['elapsed_ms']} ms<br>
                <b>API Round-trip:</b> {result['request_latency_ms']} ms<br>
                <b>QC Outcome:</b> {result['qc']}<br>
                <b>Anatomy Balance:</b> {result['anatomy']}<br>
                <b>Prediction ID:</b> <span style='font-family:monospace;font-size:.82rem'>
                {result['prediction_id']}</span>
              </div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        # Pipeline stages timeline
        st.markdown("<div class='card'><div class='sec-head'>⏱️ Inference Timeline</div>"
                    "<div style='display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.5rem'>",
                    unsafe_allow_html=True)
        for i, stage in enumerate(result.get("pipeline_stages", []), 1):
            lbl = str(stage).replace("_"," ").title()
            st.markdown(f"""<div style='background:rgba(45,212,191,.1);border:1px solid rgba(45,212,191,.2);
              border-radius:10px;padding:.4rem .75rem;font-size:.83rem;font-weight:700;color:#a7f3d0'>
              {i}. {lbl}</div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Postprocessing steps applied
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        pp_steps = result["data"].get("postprocessing", [])
        if pp_steps:
            st.markdown("<div class='card'><div class='sec-head'>🔬 Postprocessing Applied</div>",
                        unsafe_allow_html=True)
            for step in pp_steps:
                st.markdown(f"<div style='font-size:.88rem;color:rgba(236,254,255,.8);padding:.18rem 0'>"
                            f"✓ {step}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run segmentation to view AI model analysis.")

# ── Tab 5: Export ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("#### Export Artefacts")
    if result:
        st.markdown(f"""<div class='success-box' style='margin-bottom:1rem'>
          <b>✅ Prediction Ready for Export</b> · ID: 
          <span style='font-family:monospace'>{str(result['prediction_id'])[:16]}</span><br>
          <span style='font-size:.9rem'>Download individual artefacts or the full ZIP package below.</span>
        </div>""", unsafe_allow_html=True)

        from src.report_generator import report_to_json_bytes, report_to_csv_bytes
        report_json  = report_to_json_bytes(result["report"])
        report_csv   = report_to_csv_bytes(result["report"])
        zip_bytes    = make_zip(result)
        pid_short    = str(result["prediction_id"])[:8]

        # Row 1 — images
        dc1, dc2, dc3 = st.columns(3, gap="medium")
        with dc1:
            st.download_button("⬇️ Binary Mask (PNG)",
                               data=pil_to_png_bytes(result["binary_mask"]),
                               file_name=f"mask_{pid_short}.png", mime="image/png",
                               use_container_width=True)
        with dc2:
            st.download_button("⬇️ Clinical Overlay (PNG)",
                               data=pil_to_png_bytes(result["clinical_overlay"]),
                               file_name=f"overlay_{pid_short}.png", mime="image/png",
                               use_container_width=True)
        with dc3:
            st.download_button("⬇️ Original Image (PNG)",
                               data=pil_to_png_bytes(result["original_image"]),
                               file_name=f"original_{pid_short}.png", mime="image/png",
                               use_container_width=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        # Row 2 — reports + ZIP
        re1, re2, re3 = st.columns(3, gap="medium")
        with re1:
            st.download_button("📄 JSON Diagnostic Report",
                               data=report_json,
                               file_name=f"report_{pid_short}.json",
                               mime="application/json",
                               use_container_width=True)
        with re2:
            st.download_button("📊 CSV Summary Row",
                               data=report_csv,
                               file_name=f"summary_{pid_short}.csv",
                               mime="text/csv",
                               use_container_width=True)
        with re3:
            st.download_button("🗜️ Full ZIP Package",
                               data=zip_bytes,
                               file_name=f"pulmovision_{pid_short}.zip",
                               mime="application/zip",
                               use_container_width=True)

        # Report preview
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Preview JSON Diagnostic Report"):
            st.json(result["report"])
    else:
        st.info("Run segmentation to unlock export artefacts.")

# ── Tab 6: Compare ────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("#### Side-by-Side Comparison")
    st.caption("Original radiograph · Clinical overlay · Binary lung mask — aligned for direct visual comparison")
    if result and result.get("original_image"):
        _c1, _c2, _c3 = st.columns(3, gap="medium")
        with _c1:
            st.markdown("<div style='text-align:center;font-weight:700;color:#2dd4bf;margin-bottom:.4rem'>📷 Original Radiograph</div>", unsafe_allow_html=True)
            st.image(result["original_image"])
        with _c2:
            st.markdown("<div style='text-align:center;font-weight:700;color:#38bdf8;margin-bottom:.4rem'>🩻 Clinical Overlay</div>", unsafe_allow_html=True)
            if result.get("clinical_overlay"):
                st.image(result["clinical_overlay"])
        with _c3:
            st.markdown("<div style='text-align:center;font-weight:700;color:#a78bfa;margin-bottom:.4rem'>🫁 Binary Mask</div>", unsafe_allow_html=True)
            if result.get("binary_mask"):
                st.image(result["binary_mask"])
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        _s1, _s2, _s3, _s4 = st.columns(4)
        with _s1:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>🎯 Coverage</div><div class='kpi-value' style='color:#2dd4bf'>{result['mask_coverage']}%</div></div>", unsafe_allow_html=True)
        with _s2:
            _cc2 = conf_color(result["confidence"])
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>💡 Confidence</div><div class='kpi-value' style='color:{_cc2}'>{result['confidence']}</div></div>", unsafe_allow_html=True)
        with _s3:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>⚖️ Symmetry</div><div class='kpi-value' style='font-size:1rem'>{result['anatomy']}</div></div>", unsafe_allow_html=True)
        with _s4:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>✅ QC</div><div class='kpi-value' style='font-size:1rem'>{result['qc']}</div></div>", unsafe_allow_html=True)
    else:
        st.info("Run segmentation to enable side-by-side comparison.")

# ── Tab 7: Training Curve ─────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("#### Model Training History")
    st.caption("Dice score over epochs — shows learning progression and the train/val generalisation gap")
    _training_data = load_training_data()
    if _training_data:
        _fig_train = training_curve_chart(_training_data)
        if _fig_train:
            st.plotly_chart(_fig_train, use_container_width=True, config={"displayModeBar": False})
        _best_val  = max(r["val_dice"] for r in _training_data)
        _best_trn  = max(r["dice"]     for r in _training_data)
        _gap       = _best_trn - _best_val
        _t1, _t2, _t3, _t4 = st.columns(4)
        with _t1:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>📊 Epochs Trained</div><div class='kpi-value' style='color:#2dd4bf'>{len(_training_data)}</div></div>", unsafe_allow_html=True)
        with _t2:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>🏆 Best Train Dice</div><div class='kpi-value' style='color:#38bdf8'>{_best_trn:.4f}</div></div>", unsafe_allow_html=True)
        with _t3:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>📐 Best Val Dice</div><div class='kpi-value' style='color:#f59e0b'>{_best_val:.4f}</div></div>", unsafe_allow_html=True)
        with _t4:
            st.markdown(f"<div class='card' style='text-align:center'><div class='kpi-label'>⚠️ Overfit Gap</div><div class='kpi-value' style='color:#ef4444'>{_gap:.4f}</div></div>", unsafe_allow_html=True)
        st.markdown("""<div class='card' style='margin-top:.8rem'>
          <div class='sec-head'>📝 Training Configuration</div>
          <div style='font-size:.9rem;line-height:1.9;color:rgba(236,254,255,.82)'>
            <b>Dataset:</b> Indiana University Chest X-Ray — 55 images (70/15/15 split)<br>
            <b>Architecture:</b> Attention U-Net · filter progression 32→64→128→256→512<br>
            <b>Loss function:</b> Combined BCE + Dice (0.5 each weight)<br>
            <b>Optimiser:</b> Adam · LR 1e-4 with ReduceLROnPlateau (factor=0.5, patience=5)<br>
            <b>Regularisation:</b> Dropout 0.3 · BatchNormalization at every conv block<br>
            <b>Early stopping:</b> Patience=12 on val_dice_coefficient (best weights restored)<br>
            <b>Augmentation:</b> Horizontal/vertical flip · 90° rotation · brightness/contrast jitter<br>
            <b>Overfit note:</b> Small dataset (55 imgs) causes train/val divergence.
            Training on full Indiana dataset (~3,955 images) expected to push Dice to >0.85.
          </div></div>""", unsafe_allow_html=True)
    else:
        st.info("Training log not found. Run: venv310\\Scripts\\python.exe train_lung_unet.py")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div class='gap'></div>", unsafe_allow_html=True)
st.markdown(f"""<div class='footer'>
  <div style='margin-bottom:.5rem'>
    <span style='font-size:1.3rem;font-weight:900;
      background:linear-gradient(135deg,#ffffff,#a7f3d0,#67e8f9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text'>
      🫁 PulmoVision AI
    </span>
    <span style='font-size:.8rem;font-weight:700;color:rgba(45,212,191,.6);margin-left:.55rem'>{model_version}</span>
  </div>
  <div style='font-size:.87rem;color:rgba(236,254,255,.6);line-height:1.8;margin-bottom:1.1rem'>
    Attention U-Net Lung Segmentation · Built for AI/ML internship portfolio demonstration
  </div>
  <div style='display:flex;justify-content:center;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem'>
    <span style='background:rgba(45,212,191,.1);border:1px solid rgba(45,212,191,.2);
      border-radius:8px;padding:.3rem .75rem;font-size:.74rem;font-weight:700;color:#5eead4'>🧠 TensorFlow/Keras</span>
    <span style='background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.2);
      border-radius:8px;padding:.3rem .75rem;font-size:.74rem;font-weight:700;color:#7dd3fc'>⚡ Streamlit</span>
    <span style='background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.2);
      border-radius:8px;padding:.3rem .75rem;font-size:.74rem;font-weight:700;color:#c4b5fd'>🚀 FastAPI</span>
    <span style='background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.2);
      border-radius:8px;padding:.3rem .75rem;font-size:.74rem;font-weight:700;color:#fcd34d'>📊 Plotly</span>
    <span style='background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.2);
      border-radius:8px;padding:.3rem .75rem;font-size:.74rem;font-weight:700;color:#86efac'>🔢 NumPy/PIL</span>
    <span style='background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);
      border-radius:8px;padding:.3rem .75rem;font-size:.74rem;font-weight:700;color:#fca5a5'>🏥 DICOM</span>
  </div>
  <div style='font-size:.74rem;color:rgba(236,254,255,.3)'>
    ⚕️ Research Use Only · Not intended for direct clinical decision-making without proper medical validation and regulatory review.
  </div>
</div>""", unsafe_allow_html=True)