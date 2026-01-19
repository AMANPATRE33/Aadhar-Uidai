import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import requests
from streamlit_option_menu import option_menu

# --------------------------- CONFIG + ULTIMATE GLOWING DARK MODE --------------------------- 
st.set_page_config(
    page_title="🆔 UIDAI Dark Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- ULTIMATE GLOWING CSS --------------------------- 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.stApp { 
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1e1e2f 0%, #252540) !important;
    border-right: 1px solid rgba(96,165,250,0.3) !important;
    box-shadow: 5px 0 25px rgba(96,165,250,0.1) !important;
}

/* GLOWING HEADINGS */
.main-title { 
    font-size: 4.2rem !important; 
    font-weight: 800 !important; 
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1.8rem;
    text-shadow: 0 0 40px rgba(96, 165, 250, 0.6), 0 0 80px rgba(167, 139, 250, 0.3) !important;
}

.section-title { 
    font-size: 2.8rem !important; 
    font-weight: 800 !important; 
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
    text-shadow: 0 0 20px rgba(96, 165, 250, 0.5) !important;
}

/* GLOWING METRIC CARDS */
.metric-card {
    background: linear-gradient(145deg, rgba(31, 41, 55, 0.95), rgba(51, 65, 85, 0.98)) !important;
    backdrop-filter: blur(25px);
    border: 1px solid rgba(96, 165, 250, 0.4);
    border-radius: 24px !important;
    padding: 2.2rem !important;
    box-shadow: 
        0 25px 50px rgba(0,0,0,0.5),
        0 0 0 1px rgba(96, 165, 250, 0.3),
        0 0 30px rgba(96, 165, 250, 0.2),
        inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.3s ease !important;
}
.metric-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 
        0 35px 70px rgba(0,0,0,0.6),
        0 0 50px rgba(96, 165, 250, 0.4) !important;
}

.metric-label { color: #cbd5e1 !important; font-size: 1.4rem !important; font-weight: 600 !important; margin-bottom: 0.8rem !important; }
.metric-value { color: #f8fafc !important; font-size: 3.8rem !important; font-weight: 800 !important; line-height: 1.1 !important; }

.hero-card {
    background: linear-gradient(145deg, rgba(31, 119, 180, 0.95), rgba(59, 130, 246, 0.95));
    backdrop-filter: blur(30px);
    border-radius: 28px;
    padding: 2.5rem;
    border: 1px solid rgba(96, 165, 250, 0.6);
    box-shadow: 
        0 30px 60px rgba(31, 119, 180, 0.4),
        0 0 40px rgba(59, 130, 246, 0.3) !important;
}

.content-card {
    background: linear-gradient(145deg, rgba(30, 30, 47, 0.95), rgba(37, 37, 64, 0.95));
    backdrop-filter: blur(25px);
    border-radius: 20px;
    border: 1px solid rgba(60, 165, 250, 0.3);
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3), 0 0 20px rgba(96,165,250,0.1) !important;
}

.plotly-chart { 
    border-radius: 20px !important; 
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 25px rgba(96,165,250,0.2) !important; 
}

/* GLOWING BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.5rem !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.4), 0 0 20px rgba(96,165,250,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(59,130,246,0.6), 0 0 30px rgba(96,165,250,0.5) !important;
}

.filter-card {
    background: linear-gradient(145deg, rgba(59,130,246,0.2), rgba(96,165,250,0.15));
    border: 1px solid rgba(96,165,250,0.4);
    border-radius: 16px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    box-shadow: 0 8px 25px rgba(59,130,246,0.15) !important;
}

p, div, label, .stMarkdown, .stText { font-size: 1.3rem !important; line-height: 1.7 !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------- INITIALIZE SESSION STATE --------------------------- 
if 'selected_states' not in st.session_state:
    st.session_state.selected_states = []
if 'selected_ages' not in st.session_state:
    st.session_state.selected_ages = ['👶 5-17 years', '🧑 18+ years']

# --------------------------- DATA LOADER --------------------------- 
@st.cache_data(ttl=3600)
def load_uidai_data():
    FORECAST_ID = "1DGvaXazKNSat-g_JmjdknuO3CXgUrjfq"
    MERGED_ID = "1qORy0hmGIsUzlJA3mP33JcCCEFO7v9qz"
    
    def load_csv(file_id):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    
    forecast_df = load_csv(FORECAST_ID)
    merged_df = load_csv(MERGED_ID)
    
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    merged_df['date'] = pd.to_datetime(merged_df['date'], dayfirst=True)
    
    forecast_df['staff_needed'] = (forecast_df['yhat'] * 0.001).astype(int)
    forecast_df['monthly_staff_cost'] = forecast_df['staff_needed'] * 25000
    forecast_df['best_case'] = forecast_df['yhat'] * 0.9
    forecast_df['worst_case'] = forecast_df['yhat'] * 1.2
    forecast_df['demand_risk'] = np.where(forecast_df['yhat'] > forecast_df['yhat'].quantile(0.8), '🔴 High', '🟢 Low')
    
    merged_df['total_updates'] = merged_df.get('bio_age_5_17', 0) + merged_df.get('bio_age_17_', 0)
    
    if 'state' in merged_df.columns:
        state_age_df = merged_df.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
        state_age_df['total'] = state_age_df['bio_age_5_17'] + state_age_df['bio_age_17_']
        state_age_df['child_pct'] = (state_age_df['bio_age_5_17'] / state_age_df['total'] * 100).round(1)
        state_age_df = state_age_df.sort_values('total', ascending=False)
        all_states = state_age_df['state'].tolist()
    else:
        state_age_df = pd.DataFrame()
        all_states = []
    
    return forecast_df, merged_df, state_age_df, all_states

forecast_df, merged_df, state_age_df, all_states = load_uidai_data()

# --------------------------- PERFECT WORKING FILTERS --------------------------- 
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(145deg, rgba(31,119,180,0.3), rgba(59,130,246,0.3)); border-radius: 20px; margin: 1rem 0; border: 1px solid rgba(96,165,250,0.5); box-shadow: 0 10px 30px rgba(59,130,246,0.2);'>
        <div style='font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🆔 UIDAI</div>
        <div style='color: #94a3b8; font-size: 0.9rem;'>Biometric Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/06/UIDAI_Logo.png", width=160)
    
    selected = option_menu(
        menu_title=None,
        options=["🏠 Dashboard", "📈 Forecast", "📊 Planning", "📋 Historical", "⚙️ Scenarios", "👥 Demographics"],
        icons=["activity", "graph-up-arrow", "clipboard-data", "clock", "sliders", "people"],
        menu_icon="cpu-fill",
        default_index=0,
        styles={
            "container": {"padding": "0.8rem", "background": "transparent"},
            "nav-link": {"font-size": "1.2rem", "font-weight": "600", "text-align": "left", "margin": "0.2rem 0", "--hover-color": "#3b82f6"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #1f77b4, #3b82f6)", "border-radius": "12px", "font-weight": "700", "box-shadow": "0 0 20px rgba(59,130,246,0.5)"},
            "icon": {"color": "#60a5fa", "font-size": "1.4rem"},
        }
    )
    
    st.markdown("### 🔥 **Interactive Filters**")
    
    # 🌍 STATE FILTER - PERFECTLY WORKING
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🌍 **Select All States**", key="select_all_states"):
            st.session_state.selected_states = all_states.copy()
            st.rerun()
    with col2:
        if st.button("🗑️ **Clear All**", key="clear_all_states"):
            st.session_state.selected_states = []
            st.rerun()
    
    selected_states = st.multiselect(
        "🌍 States",
        options=all_states,
        default=st.session_state.selected_states,
        key="state_selector"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 👥 AGE FILTER - PERFECTLY WORKING
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👥 **Select All Ages**", key="select_all_ages"):
            st.session_state.selected_ages = ['👶 5-17 years', '🧑 18+ years']
            st.rerun()
    with col2:
        if st.button("🗑️ **Clear Ages**", key="clear_ages"):
            st.session_state.selected_ages = []
            st.rerun()
    
    selected_ages = st.multiselect(
        "👥 Age Groups",
        options=['👶 5-17 years', '🧑 18+ years'],
        default=st.session_state.selected_ages,
        key="age_selector"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; color: #94a3b8; font-size: 1rem;'>
        📊 <strong>{len(forecast_df):,}</strong> forecasts<br>
        🧬 <strong>{len(merged_df):,}</strong> records<br>
        🗺️ <strong>{len(all_states):,}</strong> states
    </div>
    """, unsafe_allow_html=True)

# --------------------------- APPLY FILTERS --------------------------- 
filtered_merged = merged_df.copy()
if selected_states and len(selected_states) > 0 and len(selected_states) < len(all_states):
    filtered_merged = filtered_merged[filtered_merged['state'].isin(selected_states)]

# --------------------------- ALL PAGES - FULLY WORKING --------------------------- 
if selected == "🏠 Dashboard":
    st.markdown('<h1 class="main-title">Aadhaar Biometric Demand Intelligence</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="content-card">
            <div style='font-size: 1.6rem; color: #e2e8f0; font-weight: 600; margin-bottom: 1rem;'>✨ AI-Powered Workforce Planning</div>
            <div style='font-size: 1.3rem; color: #94a3b8; line-height: 1.6;'>Real-time demand forecasting with state-wise demographic insights</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 0.5rem;'>🚀 Live Analytics</div>
            <div style='font-size: 1.1rem; color: rgba(255,255,255,0.9);'>Demand • Risk • States</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">📊 Executive Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><p class="metric-label">Avg Monthly Demand</p><p class="metric-value">{int(forecast_df["yhat"].mean()):,}</p></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><p class="metric-label">Peak Demand</p><p class="metric-value">{int(forecast_df["yhat"].max()):,}</p></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><p class="metric-label">High Risk Months</p><p class="metric-value">{(forecast_df["demand_risk"] == "🔴 High").sum()}</p></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><p class="metric-label">Total Cost</p><p class="metric-value">₹{int(forecast_df["monthly_staff_cost"].sum()):,}</p></div>', unsafe_allow_html=True)

elif selected == "📈 Forecast":
    st.markdown('<h2 class="section-title">📈 12-Month Demand Forecast</h2>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    fig = px.line(forecast_df, x='ds', y='yhat', template='plotly_dark', line_shape='spline', 
                  title="Future Biometric Updates", labels={'yhat': 'Expected Updates'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "📊 Planning":
    st.markdown('<h2 class="section-title">📊 Risk-Based Action Plan</h2>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    table_data = forecast_df[['ds','yhat','demand_risk','staff_needed','monthly_staff_cost']].copy()
    table_data['action'] = np.where(table_data['demand_risk'] == '🔴 High', '🚨 Recruit Now', '✅ Monitor')
    st.dataframe(table_data.rename(columns={
        'ds':'📅 Month','yhat':'🎯 Expected','demand_risk':'⚠️ Risk','staff_needed':'👥 Staff',
        'action':'✅ Action','monthly_staff_cost':'💰 Cost ₹'
    }), use_container_width=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "📋 Historical":
    st.markdown('<h2 class="section-title">📋 Historical Analysis</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        hist_data = filtered_merged.groupby('date')['total_updates'].sum().nlargest(10).reset_index()
        if not hist_data.empty:
            fig = px.bar(hist_data, x='date', y='total_updates', template='plotly_dark', 
                        title="🔥 Top Peak Months", color='total_updates')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        if 'state' in filtered_merged.columns:
            state_hist = filtered_merged.groupby('state')['total_updates'].sum().nlargest(10)
            fig = px.bar(state_hist.reset_index(), x='total_updates', y='state', 
                        orientation='h', template='plotly_dark', title="🏛️ Top States")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif selected == "⚙️ Scenarios":
    st.markdown('<h2 class="section-title">⚙️ Scenario Planning</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        scenario_df = forecast_df[['ds', 'best_case', 'yhat', 'worst_case']].melt(id_vars='ds')
        scenario_df['variable'] = scenario_df['variable'].replace({
            'best_case': '🟢 Best', 'yhat': '📊 Expected', 'worst_case': '🔴 Worst'
        })
        fig = px.line(scenario_df, x='ds', y='value', color='variable', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        fig = px.area(forecast_df, x='ds', y='monthly_staff_cost', template='plotly_dark',
                     title="💸 Cost Projection")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif selected == "👥 Demographics":
    st.markdown('<h2 class="section-title">👥 Demographics Analysis</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        if 'bio_age_5_17' in filtered_merged.columns:
            age_data = {}
            if '👶 5-17 years' in selected_ages: age_data['👶 5-17'] = filtered_merged['bio_age_5_17'].sum()
            if '🧑 18+ years' in selected_ages: age_data['🧑 18+'] = filtered_merged['bio_age_17_'].sum()
            if age_data:
                fig = px.pie(values=list(age_data.values()), names=list(age_data.keys()), 
                           hole=0.4, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        if not state_age_df.empty:
            filtered_states = state_age_df[state_age_df['state'].isin(selected_states)] if selected_states else state_age_df.head(10)
            fig = px.bar(filtered_states, x='total', y='state', orientation='h', 
                        color='child_pct', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- GLOWING FOOTER --------------------------- 
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #94a3b8; font-size: 1.4rem; font-weight: 600; 
    background: linear-gradient(145deg, rgba(30,30,47,0.8), rgba(37,37,64,0.8)); 
    border-radius: 20px; border: 1px solid rgba(96,165,250,0.2); 
    box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 30px rgba(96,165,250,0.1);'>
    🌟 ULTIMATE UIDAI DASHBOARD | All Filters ✅ | Glowing Effects ✨ | January 2026
</div>
""", unsafe_allow_html=True)
