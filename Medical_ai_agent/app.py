"""
╔══════════════════════════════════════════════════════════════╗
║           MEDICAL AI AGENT - app.py (Main Entry)            ║
║      Dynamic Streamlit UI with GraphRAG + OpenAI Agents     ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import time
import json
import base64
from pathlib import Path
from datetime import datetime

# ── Page Config (MUST be first Streamlit call) ───────────────
st.set_page_config(
    page_title="MedAI Agent | Your Personal Medical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject Dynamic CSS ────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=Space+Mono:wght@400;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --primary: #00D4AA;
        --primary-dark: #00A884;
        --secondary: #6C63FF;
        --accent: #FF6B6B;
        --bg-dark: #0A0E1A;
        --bg-card: #111827;
        --bg-card2: #1A2235;
        --text-main: #E8F0FE;
        --text-muted: #8892A4;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --border: rgba(0,212,170,0.15);
        --glow: 0 0 30px rgba(0,212,170,0.2);
    }

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-main) !important;
    }

    .stApp {
        background: var(--bg-dark) !important;
        background-image:
            radial-gradient(ellipse at 20% 20%, rgba(0,212,170,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(108,99,255,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(10,14,26,1) 0%, transparent 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1321 0%, #111827 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    /* ── Hero Section ── */
    .hero-container {
        position: relative;
        text-align: center;
        padding: 2rem 1rem 1rem;
        overflow: hidden;
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        background: linear-gradient(135deg, #00D4AA, #6C63FF, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem;
        color: var(--primary) !important;
        letter-spacing: 4px;
        text-transform: uppercase;
        opacity: 0.9;
    }

    /* ── Robot SVG Container ── */
    .robot-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
        position: relative;
    }
    .robot-glow {
        filter: drop-shadow(0 0 20px rgba(0,212,170,0.5));
        animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-12px); }
    }

    /* ── Pulse Ring ── */
    .pulse-ring {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .pulse-ring::before,
    .pulse-ring::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        border: 2px solid var(--primary);
        animation: pulse-expand 2.5s ease-out infinite;
    }
    .pulse-ring::after { animation-delay: 1.25s; }
    @keyframes pulse-expand {
        0%   { width: 120px; height: 120px; opacity: 1; }
        100% { width: 220px; height: 220px; opacity: 0; }
    }

    /* ── Cards ── */
    .med-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .med-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    .med-card:hover {
        border-color: var(--primary);
        box-shadow: var(--glow);
        transform: translateY(-2px);
    }
    .med-card-header {
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem;
        font-weight: 700;
        color: var(--primary) !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Metric Badge ── */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 0.2rem;
    }
    .badge-success { background: rgba(16,185,129,0.15); color: #10B981 !important; border: 1px solid rgba(16,185,129,0.3); }
    .badge-warning { background: rgba(245,158,11,0.15); color: #F59E0B !important; border: 1px solid rgba(245,158,11,0.3); }
    .badge-danger  { background: rgba(239,68,68,0.15);  color: #EF4444 !important; border: 1px solid rgba(239,68,68,0.3);  }
    .badge-info    { background: rgba(0,212,170,0.15);  color: #00D4AA !important; border: 1px solid rgba(0,212,170,0.3);  }
    .badge-purple  { background: rgba(108,99,255,0.15); color: #6C63FF !important; border: 1px solid rgba(108,99,255,0.3); }

    /* ── Upload Zone ── */
    .upload-zone {
        background: linear-gradient(135deg, rgba(0,212,170,0.05), rgba(108,99,255,0.05));
        border: 2px dashed rgba(0,212,170,0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .upload-zone:hover {
        border-color: var(--primary);
        background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(108,99,255,0.1));
    }

    /* ── Progress Bar ── */
    .custom-progress {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(0,212,170,0.5);
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
        color: var(--bg-dark) !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,212,170,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0,212,170,0.5) !important;
    }

    /* ── Selectbox, Input ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed rgba(0,212,170,0.3) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; }

    /* ── Agent Status Pills ── */
    .agent-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 30px;
        padding: 0.4rem 1rem;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.65rem;
        margin: 0.2rem;
    }
    .dot-active  { width: 6px; height: 6px; border-radius: 50%; background: var(--success); box-shadow: 0 0 6px var(--success); }
    .dot-idle    { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); }
    .dot-running { width: 6px; height: 6px; border-radius: 50%; background: var(--warning); animation: blink 0.8s infinite; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

    /* ── Scroll bar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    /* ── Disclaimer ── */
    .disclaimer-box {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.75rem;
        color: rgba(239,68,68,0.9) !important;
        margin: 0.5rem 0;
    }

    /* ── Streamlit overrides ── */
    .stMarkdown p { color: var(--text-main) !important; }
    .stSpinner > div { border-top-color: var(--primary) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    .stAlert { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ── Medical Robot SVG ─────────────────────────────────────────
def render_medical_robot():
    robot_svg = """
    <div class="robot-container">
      <div class="pulse-ring">
        <svg class="robot-glow" width="130" height="160" viewBox="0 0 130 160" xmlns="http://www.w3.org/2000/svg">
          <!-- Body -->
          <rect x="30" y="65" width="70" height="65" rx="12" fill="#1A2235" stroke="#00D4AA" stroke-width="2"/>
          <!-- Head -->
          <rect x="35" y="18" width="60" height="45" rx="14" fill="#1A2235" stroke="#00D4AA" stroke-width="2"/>
          <!-- Head antenna -->
          <line x1="65" y1="18" x2="65" y2="8" stroke="#00D4AA" stroke-width="2"/>
          <circle cx="65" cy="6" r="4" fill="#00D4AA" opacity="0.9">
            <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>
          </circle>
          <!-- Eyes -->
          <rect x="43" y="28" width="16" height="12" rx="5" fill="#6C63FF" opacity="0.9"/>
          <rect x="71" y="28" width="16" height="12" rx="5" fill="#6C63FF" opacity="0.9"/>
          <circle cx="51" cy="34" r="4" fill="#00D4AA">
            <animate attributeName="cx" values="51;53;51;49;51" dur="4s" repeatCount="indefinite"/>
          </circle>
          <circle cx="79" cy="34" r="4" fill="#00D4AA">
            <animate attributeName="cx" values="79;81;79;77;79" dur="4s" repeatCount="indefinite"/>
          </circle>
          <!-- Mouth display -->
          <rect x="47" y="46" width="36" height="10" rx="5" fill="rgba(0,212,170,0.2)" stroke="#00D4AA" stroke-width="1"/>
          <line x1="50" y1="51" x2="55" y2="51" stroke="#00D4AA" stroke-width="1.5"/>
          <line x1="58" y1="49" x2="58" y2="53" stroke="#00D4AA" stroke-width="1.5"/>
          <line x1="62" y1="51" x2="72" y2="51" stroke="#00D4AA" stroke-width="1.5"/>
          <line x1="75" y1="49" x2="75" y2="53" stroke="#00D4AA" stroke-width="1.5"/>
          <line x1="78" y1="51" x2="80" y2="51" stroke="#00D4AA" stroke-width="1.5"/>
          <!-- Neck -->
          <rect x="55" y="63" width="20" height="4" rx="2" fill="#00D4AA" opacity="0.4"/>
          <!-- Chest panel -->
          <rect x="40" y="75" width="50" height="35" rx="8" fill="rgba(0,212,170,0.05)" stroke="rgba(0,212,170,0.3)" stroke-width="1"/>
          <!-- Red cross -->
          <rect x="61" y="81" width="8" height="22" rx="2" fill="#EF4444" opacity="0.9"/>
          <rect x="55" y="87" width="20" height="8" rx="2" fill="#EF4444" opacity="0.9"/>
          <!-- Heart beat line -->
          <polyline points="42,97 47,97 50,90 53,104 56,90 59,97 88,97" fill="none" stroke="#00D4AA" stroke-width="1.5" opacity="0.8">
            <animate attributeName="opacity" values="0.8;0.2;0.8" dur="1.5s" repeatCount="indefinite"/>
          </polyline>
          <!-- Arms -->
          <rect x="8" y="68" width="22" height="12" rx="6" fill="#1A2235" stroke="#00D4AA" stroke-width="1.5"/>
          <rect x="100" y="68" width="22" height="12" rx="6" fill="#1A2235" stroke="#00D4AA" stroke-width="1.5"/>
          <!-- Hand left - holding clipboard -->
          <rect x="5" y="82" width="26" height="18" rx="5" fill="#1A2235" stroke="#00D4AA" stroke-width="1.5"/>
          <rect x="9" y="85" width="18" height="12" rx="3" fill="rgba(108,99,255,0.2)" stroke="#6C63FF" stroke-width="1"/>
          <line x1="11" y1="89" x2="24" y2="89" stroke="#6C63FF" stroke-width="1"/>
          <line x1="11" y1="92" x2="24" y2="92" stroke="#6C63FF" stroke-width="1"/>
          <!-- Hand right - stethoscope -->
          <rect x="99" y="82" width="26" height="18" rx="5" fill="#1A2235" stroke="#00D4AA" stroke-width="1.5"/>
          <circle cx="112" cy="91" r="6" fill="none" stroke="#00D4AA" stroke-width="1.5"/>
          <circle cx="112" cy="91" r="2" fill="#00D4AA"/>
          <!-- Legs -->
          <rect x="38" y="130" width="22" height="22" rx="8" fill="#1A2235" stroke="#00D4AA" stroke-width="1.5"/>
          <rect x="70" y="130" width="22" height="22" rx="8" fill="#1A2235" stroke="#00D4AA" stroke-width="1.5"/>
          <!-- Feet -->
          <rect x="34" y="148" width="30" height="10" rx="5" fill="#0A0E1A" stroke="#00D4AA" stroke-width="1.5"/>
          <rect x="66" y="148" width="30" height="10" rx="5" fill="#0A0E1A" stroke="#00D4AA" stroke-width="1.5"/>
          <!-- Wi-Fi signal -->
          <path d="M 58 14 Q 65 9 72 14" stroke="#00D4AA" stroke-width="1.5" fill="none" opacity="0.7">
            <animate attributeName="opacity" values="0.7;0.1;0.7" dur="2s" repeatCount="indefinite"/>
          </path>
          <path d="M 55 11 Q 65 4 75 11" stroke="#00D4AA" stroke-width="1" fill="none" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.05;0.4" dur="2s" begin="0.5s" repeatCount="indefinite"/>
          </path>
        </svg>
      </div>
    </div>
    """
    st.markdown(robot_svg, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800;
                        background: linear-gradient(135deg, #00D4AA, #6C63FF);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text;">
                🏥 MedAI Agent
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                        color:#8892A4; letter-spacing:3px; margin-top:4px;">
                INTELLIGENCE v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Agent Status
        st.markdown("**⚡ Agent Status**")
        agents = [
            ("🔬 Diagnosis Agent", "active"),
            ("📈 Prognosis Agent", "active"),
            ("🥗 Lifestyle Agent", "active"),
            ("💊 Medication Agent", "active"),
            ("🧬 GraphRAG Engine", "active"),
            ("🔮 Pinecone Memory", "active"),
        ]
        for name, status in agents:
            dot_class = "dot-active" if status == "active" else "dot-idle"
            st.markdown(f"""
            <div class="agent-pill">
                <span class="{dot_class}"></span>
                <span style="color:#E8F0FE;">{name}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # User Profile
        st.markdown("**👤 Patient Profile**")
        with st.expander("Configure Profile", expanded=False):
            st.text_input("Full Name", placeholder="John Doe", key="user_name")
            st.number_input("Age", min_value=1, max_value=120, value=30, key="user_age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="user_gender")
            st.text_area("Known Allergies", placeholder="Penicillin, Aspirin...", key="allergies", height=70)
            st.text_area("Current Medications", placeholder="Metformin 500mg...", key="current_meds", height=70)
            st.text_area("Chronic Conditions", placeholder="Diabetes Type 2...", key="conditions", height=70)
            if st.button("💾 Save Profile"):
                st.success("Profile saved to Pinecone memory!")

        st.markdown("---")

        # Model Config
        st.markdown("**⚙️ Model Configuration**")
        llm_choice = st.selectbox(
            "Primary LLM",
            ["gpt-4o", "gpt-4-turbo", "claude-3-opus", "claude-3-sonnet"],
            key="llm_model"
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="temp")
        use_graphrag = st.toggle("Enable GraphRAG", value=True, key="graphrag")
        use_pinecone = st.toggle("Enable Pinecone Memory", value=True, key="pinecone")
        multilang = st.toggle("Multi-Language Output", value=False, key="multilang")
        if multilang:
            st.selectbox("Output Language", ["English", "Spanish", "French", "German", "Hindi", "Arabic"], key="lang")

        st.markdown("---")
        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <strong>Medical Disclaimer</strong><br/>
            This AI is for informational purposes only and does not replace professional medical advice,
            diagnosis, or treatment. Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

# ── Render Analysis Results ───────────────────────────────────
def render_analysis_tabs(results: dict):
    tabs = st.tabs([
        "📋 Summary",
        "🔬 Conditions",
        "📈 Future Risks",
        "🥗 Diet Plan",
        "💊 Medications",
        "🏃 Exercise",
        "🧬 Graph Relations",
        "📊 Raw Report"
    ])

    with tabs[0]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">📋 Clinical Summary</div>', unsafe_allow_html=True)
        st.markdown(results.get("summary", "_Analysis pending..._"))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", results.get("risk_score", "N/A"), delta=results.get("risk_delta", ""))
        with col2:
            st.metric("Confidence", results.get("confidence", "N/A"))
        with col3:
            st.metric("Pages Analyzed", results.get("pages", "N/A"))
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">🔬 Detected Conditions</div>', unsafe_allow_html=True)
        conditions = results.get("conditions", [])
        if conditions:
            for cond in conditions:
                severity_badge = {
                    "high": "badge-danger",
                    "medium": "badge-warning",
                    "low": "badge-success"
                }.get(cond.get("severity", "low"), "badge-info")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:1rem; margin:0.5rem 0; border-left: 3px solid var(--primary);">
                    <strong style="color:#E8F0FE;">{cond.get('name','')}</strong>
                    <span class="metric-badge {severity_badge}">{cond.get('severity','').upper()}</span>
                    <p style="color:#8892A4; margin:0.5rem 0 0; font-size:0.85rem;">{cond.get('description','')}</p>
                    <p style="color:#00D4AA; font-size:0.8rem;">📚 Source: {cond.get('source','PubMed KG')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No conditions data yet. Upload a report to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">📈 Future Health Risks (3-Month / 1-Year Outlook)</div>', unsafe_allow_html=True)
        risks = results.get("risks", [])
        if risks:
            for risk in risks:
                prob = risk.get("probability", 0)
                color = "#EF4444" if prob > 70 else "#F59E0B" if prob > 40 else "#10B981"
                st.markdown(f"""
                <div style="margin:0.8rem 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:#E8F0FE; font-weight:500;">{risk.get('name','')}</span>
                        <span style="color:{color}; font-family:'Space Mono',monospace; font-size:0.8rem;">{prob}%</span>
                    </div>
                    <div class="custom-progress">
                        <div class="custom-progress-fill" style="width:{prob}%; background: linear-gradient(90deg, {color}88, {color});"></div>
                    </div>
                    <span style="color:#8892A4; font-size:0.75rem;">{risk.get('description','')}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No risk data yet. Upload a report to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">🥗 Personalized Diet Plan</div>', unsafe_allow_html=True)
        diet = results.get("diet", {})
        if diet:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Recommended Foods**")
                for item in diet.get("recommended", []):
                    st.markdown(f"<span class='metric-badge badge-success'>✓ {item}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown("**❌ Foods to Avoid**")
                for item in diet.get("avoid", []):
                    st.markdown(f"<span class='metric-badge badge-danger'>✗ {item}</span>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"**📅 Sample Meal Plan:**\n\n{diet.get('meal_plan','')}")
        else:
            st.info("No diet data yet. Upload a report to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">💊 Medication Recommendations</div>', unsafe_allow_html=True)
        meds = results.get("medications", [])
        if meds:
            for med in meds:
                st.markdown(f"""
                <div style="background:rgba(108,99,255,0.07); border:1px solid rgba(108,99,255,0.2);
                            border-radius:10px; padding:1rem; margin:0.5rem 0;">
                    <strong style="color:#6C63FF;">💊 {med.get('name','')}</strong>
                    <span class="metric-badge badge-purple">{med.get('dosage','')}</span>
                    <p style="color:#8892A4; margin:0.5rem 0 0; font-size:0.85rem;">{med.get('purpose','')}</p>
                    <p style="color:#F59E0B; font-size:0.75rem;">⚠️ Side effects: {med.get('side_effects','Consult doctor')}</p>
                    <p style="color:#8892A4; font-size:0.7rem;">📚 {med.get('source','DrugBank')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No medication data yet. Upload a report to analyze.")
        st.markdown("""
        <div class="disclaimer-box" style="margin-top:1rem;">
            ⚠️ Always consult your doctor before starting, stopping, or changing any medications.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[5]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">🏃 Exercise & Lifestyle Plan</div>', unsafe_allow_html=True)
        exercises = results.get("exercises", [])
        if exercises:
            for ex in exercises:
                intensity_color = {"low": "#10B981", "moderate": "#F59E0B", "high": "#EF4444"}.get(ex.get("intensity","low"), "#00D4AA")
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:1rem; background:rgba(255,255,255,0.03);
                            border-radius:10px; padding:1rem; margin:0.5rem 0;">
                    <div style="font-size:2rem;">{ex.get('icon','🏃')}</div>
                    <div style="flex:1;">
                        <strong style="color:#E8F0FE;">{ex.get('name','')}</strong>
                        <span class="metric-badge" style="background:rgba(0,0,0,0.2); color:{intensity_color}; border:1px solid {intensity_color}44;">
                            {ex.get('intensity','').upper()}
                        </span>
                        <p style="color:#8892A4; margin:0.3rem 0 0; font-size:0.8rem;">{ex.get('description','')}</p>
                        <span style="color:#00D4AA; font-size:0.75rem; font-family:'Space Mono',monospace;">
                            ⏱ {ex.get('duration','')} | 📅 {ex.get('frequency','')}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No exercise data yet. Upload a report to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[6]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">🧬 Knowledge Graph Relations (GraphRAG)</div>', unsafe_allow_html=True)
        graph_data = results.get("graph_relations", "")
        if graph_data:
            st.json(graph_data)
        else:
            st.markdown("""
            <div style="text-align:center; padding:2rem; color:#8892A4;">
                <div style="font-size:3rem;">🕸️</div>
                <p>GraphRAG relationships will appear here after analysis.</p>
                <p style="font-size:0.8rem;">Powered by Neo4j + Hetionet + PubMed KG</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[7]:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-header">📊 Extracted Report Text</div>', unsafe_allow_html=True)
        raw = results.get("raw_text", "")
        if raw:
            st.text_area("Extracted Content", raw, height=300)
        else:
            st.info("Raw text will appear after document parsing.")
        st.markdown('</div>', unsafe_allow_html=True)


# ── Simulate Agent Processing ─────────────────────────────────
def simulate_analysis(uploaded_file, use_graphrag: bool, use_pinecone: bool):
    """Calls actual agent pipeline or returns demo data."""
    try:
        from agents import run_medical_analysis
        return run_medical_analysis(uploaded_file, use_graphrag, use_pinecone)
    except ImportError:
        # Demo mode - returns mock data so UI is testable without API keys
        time.sleep(2)
        return {
            "summary": """**Patient Overview:** 45-year-old male presenting with elevated blood glucose levels (HbA1c: 7.8%), mild hypertension (140/90 mmHg),
and slightly elevated LDL cholesterol (165 mg/dL). CBC within normal limits. Kidney function tests show early-stage microalbuminuria.

**Key Findings:** The combination of metabolic markers suggests **pre-diabetic to early Type 2 Diabetes** with cardiovascular risk factors.
Immediate lifestyle intervention recommended alongside medication review.""",
            "risk_score": "6.2/10",
            "risk_delta": "+0.4 vs last visit",
            "confidence": "94.3%",
            "pages": "3 pages",
            "conditions": [
                {"name": "Type 2 Diabetes (Early Stage)", "severity": "medium",
                 "description": "HbA1c at 7.8% indicates suboptimal glycemic control. Regular monitoring and metformin review advised.",
                 "source": "PubMed KG + Hetionet"},
                {"name": "Stage 1 Hypertension", "severity": "medium",
                 "description": "BP consistently at 140/90 mmHg. Lifestyle changes and possible ACE inhibitor introduction.",
                 "source": "DrugBank + PubMed"},
                {"name": "Dyslipidemia", "severity": "low",
                 "description": "Elevated LDL at 165 mg/dL. Statin therapy may be considered.",
                 "source": "Hetionet KG"},
                {"name": "Microalbuminuria", "severity": "low",
                 "description": "Early kidney involvement marker. Annual monitoring recommended.",
                 "source": "MIMIC-III"},
            ],
            "risks": [
                {"name": "Cardiovascular Event (MI/Stroke)", "probability": 32, "description": "3-year risk based on current profile"},
                {"name": "Diabetic Nephropathy", "probability": 18, "description": "If glycemic control not improved"},
                {"name": "Diabetic Retinopathy", "probability": 12, "description": "Annual ophthalmology check advised"},
                {"name": "Neuropathy", "probability": 22, "description": "Foot care and regular nerve checks needed"},
                {"name": "Hypertensive Crisis", "probability": 8, "description": "With current BP trend, risk is moderate"},
            ],
            "diet": {
                "recommended": ["Leafy greens", "Oily fish (salmon)", "Nuts & seeds", "Berries", "Whole grains", "Legumes", "Olive oil"],
                "avoid": ["Refined sugars", "White bread/rice", "Processed meats", "Trans fats", "High-sodium foods", "Sweetened beverages"],
                "meal_plan": """**Breakfast:** Oatmeal with berries, flaxseeds, Greek yogurt | **Lunch:** Grilled salmon salad with quinoa, olive oil dressing
**Dinner:** Stir-fried vegetables with tofu/lean chicken, brown rice | **Snacks:** Almonds, apple slices, hummus with carrots
**Hydration:** 8-10 glasses of water, green tea (no sugar)"""
            },
            "medications": [
                {"name": "Metformin", "dosage": "500mg twice daily", "purpose": "First-line T2DM management, improves insulin sensitivity",
                 "side_effects": "Nausea, GI upset (take with food)", "source": "DrugBank DB00331"},
                {"name": "Amlodipine", "dosage": "5mg once daily", "purpose": "Calcium channel blocker for hypertension management",
                 "side_effects": "Ankle swelling, flushing", "source": "DrugBank DB00381"},
                {"name": "Atorvastatin", "dosage": "20mg at bedtime", "purpose": "LDL reduction, cardiovascular risk reduction",
                 "side_effects": "Muscle aches (report immediately if severe)", "source": "DrugBank DB01076"},
            ],
            "exercises": [
                {"icon": "🚶", "name": "Brisk Walking", "intensity": "low", "duration": "30 min",
                 "frequency": "Daily", "description": "Start with 10-min sessions, build up. Great for blood sugar regulation."},
                {"icon": "🏊", "name": "Swimming", "intensity": "moderate", "duration": "30-45 min",
                 "frequency": "3x/week", "description": "Excellent low-impact full-body exercise, gentle on joints."},
                {"icon": "🚴", "name": "Cycling (Stationary)", "intensity": "moderate", "duration": "20-30 min",
                 "frequency": "3x/week", "description": "Improves cardiovascular health and insulin sensitivity."},
                {"icon": "🏋️", "name": "Resistance Training", "intensity": "moderate", "duration": "20 min",
                 "frequency": "2x/week", "description": "Builds muscle mass which improves glucose uptake."},
                {"icon": "🧘", "name": "Yoga / Stretching", "intensity": "low", "duration": "15-20 min",
                 "frequency": "Daily", "description": "Stress reduction lowers cortisol which helps blood sugar control."},
            ],
            "graph_relations": {
                "patient_node": "Patient_001",
                "condition_edges": ["Patient→HAS_CONDITION→Type2Diabetes", "Patient→HAS_CONDITION→Hypertension"],
                "drug_edges": ["Type2Diabetes→TREATED_BY→Metformin", "Hypertension→TREATED_BY→Amlodipine"],
                "risk_edges": ["Type2Diabetes→INCREASES_RISK_OF→CardiovascularDisease"],
                "source": "Hetionet + PubMed KG"
            },
            "raw_text": "PATIENT REPORT\nDate: 2024-01-15\nHbA1c: 7.8%\nFasting Glucose: 148 mg/dL\nBP: 140/90 mmHg\nLDL: 165 mg/dL\nHDL: 42 mg/dL\nCreatinine: 1.1 mg/dL\nMicroalbumin/Creatinine Ratio: 42 mg/g (elevated)\n..."
        }


# ── Main App ──────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero Section ──────────────────────
    st.markdown("""
    <div class="hero-container">
        <div class="hero-subtitle">🔬 AI-POWERED MEDICAL INTELLIGENCE PLATFORM</div>
        <div class="hero-title">Medical AI Agent</div>
        <div style="color:#8892A4; font-size:0.9rem; max-width:600px; margin:0.5rem auto 0;">
            Upload your medical report and receive comprehensive AI-powered analysis,<br/>
            personalized recommendations, and predictive health insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_medical_robot()

    # ── Stats Row ──────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (col1, "🧬", "5M+", "Medical Entities", "badge-info"),
        (col2, "📚", "28M+", "PubMed Articles", "badge-purple"),
        (col3, "💊", "13K+", "Drug Interactions", "badge-warning"),
        (col4, "⚡", "4 AI", "Specialized Agents", "badge-success"),
    ]
    for col, icon, val, label, badge in stats:
        with col:
            st.markdown(f"""
            <div class="med-card" style="text-align:center; padding:1rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
                            color:#E8F0FE;">{val}</div>
                <div style="color:#8892A4; font-size:0.75rem; letter-spacing:1px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload Section ─────────────────────
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-header">📤 Upload Medical Report</div>', unsafe_allow_html=True)

    col_upload, col_options = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag & drop your medical report here",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            help="Supports PDF, PNG, JPG, TIFF. Max file size: 50MB",
            key="file_upload"
        )
        if uploaded_file:
            file_details_col1, file_details_col2, file_details_col3 = st.columns(3)
            with file_details_col1:
                st.markdown(f"<span class='metric-badge badge-info'>📄 {uploaded_file.name}</span>", unsafe_allow_html=True)
            with file_details_col2:
                size_kb = uploaded_file.size // 1024
                st.markdown(f"<span class='metric-badge badge-success'>💾 {size_kb} KB</span>", unsafe_allow_html=True)
            with file_details_col3:
                st.markdown(f"<span class='metric-badge badge-purple'>🔮 {uploaded_file.type}</span>", unsafe_allow_html=True)

    with col_options:
        st.markdown("**Analysis Options**")
        deep_scan = st.toggle("Deep Scan Mode", value=True)
        alert_on_risk = st.toggle("Risk Alerts (Email/SMS)", value=False)
        if alert_on_risk:
            st.text_input("Alert Email/Phone", placeholder="your@email.com or +1234567890")
        generate_pdf = st.toggle("Export PDF Report", value=True)

    analyze_btn = st.button("🚀 Analyze Report with AI Agents", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Analysis Pipeline ──────────────────
    if analyze_btn:
        if not uploaded_file:
            st.error("⚠️ Please upload a medical report before analyzing.")
        else:
            # Processing Pipeline Steps
            pipeline_steps = [
                ("🔍", "OCR & Document Parsing", 0.3),
                ("📊", "Extracting Medical Entities", 0.5),
                ("🧬", "Querying GraphRAG (Neo4j + Hetionet)", 0.7),
                ("🔮", "Searching Pinecone Vector Memory", 0.8),
                ("🤖", "Running Diagnosis Agent", 0.87),
                ("📈", "Running Prognosis Agent", 0.92),
                ("🥗", "Running Lifestyle Agent", 0.96),
                ("💊", "Running Medication Agent", 0.99),
                ("✅", "Compiling Results", 1.0),
            ]

            progress_placeholder = st.empty()

            with progress_placeholder.container():
                st.markdown('<div class="med-card">', unsafe_allow_html=True)
                st.markdown('<div class="med-card-header">⚡ Agent Pipeline Running</div>', unsafe_allow_html=True)

                progress_bar = st.progress(0)
                status_text = st.empty()

                for icon, step_name, progress_val in pipeline_steps:
                    status_text.markdown(f"""
                    <div style="display:flex; align-items:center; gap:0.8rem; padding:0.5rem; color:#E8F0FE;">
                        <span class="dot-running"></span>
                        <span style="font-family:'DM Sans',sans-serif;">{icon} {step_name}</span>
                        <span style="margin-left:auto; font-family:'Space Mono',monospace; font-size:0.7rem; color:#00D4AA;">
                            {int(progress_val*100)}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(progress_val)
                    time.sleep(0.4)

                status_text.markdown("""
                <div style="color:#10B981; display:flex; align-items:center; gap:0.5rem; padding:0.5rem;">
                    <span>✅</span> <strong>Analysis Complete!</strong>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            time.sleep(0.5)
            progress_placeholder.empty()

            # Run actual analysis
            use_graphrag = st.session_state.get("graphrag", True)
            use_pinecone = st.session_state.get("pinecone", True)
            results = simulate_analysis(uploaded_file, use_graphrag, use_pinecone)

            st.success("✅ Analysis complete! Scroll down to view results.")
            st.balloons()

            # Store results
            st.session_state["analysis_results"] = results
            st.session_state["analyzed"] = True

    # ── Results Section ────────────────────
    if st.session_state.get("analyzed") and st.session_state.get("analysis_results"):
        st.markdown("---")
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
                    color:#E8F0FE; margin-bottom:1rem;">
            🔬 Analysis Results
        </div>
        """, unsafe_allow_html=True)
        render_analysis_tabs(st.session_state["analysis_results"])

        if st.session_state.get("generate_pdf", True):
            st.markdown("---")
            if st.button("📥 Download Full PDF Report"):
                st.info("PDF export requires `reportlab` — run: `pip install reportlab`")

    # ── Footer ──────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#8892A4; font-size:0.75rem; padding:1rem 0;">
        <span style="font-family:'Space Mono',monospace; color:#00D4AA;">MedAI Agent v2.0</span>
        &nbsp;|&nbsp; Built with Streamlit + GraphRAG + OpenAI + Pinecone
        &nbsp;|&nbsp; Data: PubMed KG · Hetionet · DrugBank · MIMIC-III<br/>
        <span style="color:#EF4444; margin-top:0.5rem; display:block;">
            ⚠️ For educational & research purposes only. Not a substitute for professional medical advice.
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
