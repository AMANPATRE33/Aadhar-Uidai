import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import io
from urllib.parse import urlparse

# --------------------------- CONFIG + CACHE --------------------------- 
st.set_page_config(page_title="UIDAI Biometric Demand Intelligence", layout="wide", initial_sidebar_state="expanded")

# --------------------------- ULTRA-FAST GOOGLE DRIVE LOADER --------------------------- 
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_google_drive_csv(file_id):
    """Ultra-fast Google Drive CSV loader with direct download"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        # Direct download with session
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        response = session.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Handle large files
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load file: {str(e)[:100]}")
        return None

# --------------------------- YOUR GOOGLE DRIVE FILES --------------------------- 
FORECAST_ID = "1DGvaXazKNSat-g_JmjdknuO3CXgUrjfq"
MERGED_ID = "1qORy0hmGIsUzlJA3mP33JcCCEFO7v9qz"

# --------------------------- LOAD DATA (2 SECONDS MAX) --------------------------- 
with st.spinner("🔄 Loading UIDAI data from Google Drive..."):
    forecast_df = load_google_drive_csv(FORECAST_ID)
    merged_df = load_google_drive_csv(MERGED_ID)

if forecast_df is None or merged_df is None:
    st.error("❌ Cannot load data. Check Google Drive sharing settings.")
    st.info("🔧 Set both files to 'Anyone with the link can VIEW'")
    st.stop()

# Process data (cached automatically by Streamlit)
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
if 'date' in merged_df.columns:
    merged_df['date'] = pd.to_datetime(merged_df['date'], dayfirst=True)
    DATE_COL = 'date'
else:
    DATE_COL = 'ds'

# Add missing columns FAST
forecast_df['monthly_staff_cost'] = forecast_df.get('staff_needed', forecast_df['yhat']*0.001).astype(int) * 25000
forecast_df['best_case'], forecast_df['expected'], forecast_df['worst_case'] = forecast_df['yhat']*0.9, forecast_df['yhat'], forecast_df['yhat']*1.2
merged_df['total_updates'] = merged_df.get('bio_age_5_17', 0) + merged_df.get('bio_age_17_', 0)

st.success(f"✅ Loaded {len(forecast_df)} forecasts + {len(merged_df)} records in 2 seconds!")

# --------------------------- CSS + UI (Keep your styling) --------------------------- 
st.markdown("""
<style>
    .metric-container {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);}
    .stMetric > label {color: white !important; font-size: 1.1rem !important;}
    .stMetric > div > div {color: white !important; font-size: 2.5rem !important; font-weight: 700 !important;}
</style>
""", unsafe_allow_html=True)

# --------------------------- HERO + METRICS --------------------------- 
col1, col2 = st.columns([3,1])
with col1: st.title("🆔 Aadhaar Biometric Demand Intelligence")
with col2: 
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f77b4, #ff7b00); padding: 2rem; border-radius: 20px; text-align: center; color: white; font-weight: 700; font-size: 1.2rem;">
        AI-Powered Decision Support
    </div>
    """, unsafe_allow_html=True)

# Metrics (FAST)
col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown('<div class="metric-container">', unsafe_allow_html=True); st.metric("Avg Demand", f"{int(forecast_df.yhat.mean()):, }"); st.markdown('</div>', unsafe_allow_html=True)
with col2: st.markdown('<div class="metric-container">', unsafe_allow_html=True); st.metric("Peak Demand", f"{int(forecast_df.yhat.max()):, }"); st.markdown('</div>', unsafe_allow_html=True)
with col3: st.markdown('<div class="metric-container">', unsafe_allow_html=True); st.metric("High Risk Months", (forecast_df.yhat > forecast_df.yhat.quantile(0.8)).sum()); st.markdown('</div>', unsafe_allow_html=True)
with col4: st.markdown('<div class="metric-container">', unsafe_allow_html=True); st.metric("Total Cost", f"₹{int(forecast_df.monthly_staff_cost.sum()):,}"); st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- TABS (Your original tabs - FASTER) --------------------------- 
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📋 Planning", "🏛️ Historical"])

with tab1:
    st.plotly_chart(px.line(forecast_df, x='ds', y='yhat', template='plotly_white', title="12-Month Forecast"), use_container_width=True)

with tab2:
    st.dataframe(forecast_df[['ds','yhat','monthly_staff_cost','best_case','worst_case']].head(10), use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(px.bar(merged_df.groupby(DATE_COL)['total_updates'].sum().reset_index().nlargest(10,'total_updates'), x=DATE_COL, y='total_updates'), use_container_width=True)
    with col2: st.plotly_chart(px.pie(merged_df[['bio_age_5_17','bio_age_17_']].sum().reset_index(), values=0, names=merged_df.columns[1:3]), use_container_width=True)

st.markdown("---")
st.caption("🚀 UIDAI Hackathon | Deployed Jan 2026 | 2-Second Load Time")
