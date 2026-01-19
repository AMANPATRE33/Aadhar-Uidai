import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io

# --------------------------- CSS STYLING --------------------------- 
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #333 !important;
        font-weight: 700;
        margin-top: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stMetric > label {
        color: white !important;
        font-size: 1.1rem !important;
    }
    .stMetric > div > div {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    .plotly-chart {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    }
    .stDataFrame {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    }
    .upload-area {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- CONFIG --------------------------- 
st.set_page_config(
    page_title="UIDAI Biometric Demand Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- MANUAL UPLOAD DATA --------------------------- 
def load_uploaded_data(forecast_file, merged_file):
    """Process uploaded CSV files with full data validation"""
    if forecast_file is not None and merged_file is not None:
        # Read CSV files
        forecast_df = pd.read_csv(forecast_file)
        merged_df = pd.read_csv(merged_file)
        
        # Fix dates
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'], dayfirst=True)
            DATE_COL = 'date'
        elif 'ds' in merged_df.columns:
            merged_df['ds'] = pd.to_datetime(merged_df['ds'])
            DATE_COL = 'ds'
        else:
            st.error("❌ No date column found in biometric dataset. Expected 'date' or 'ds'.")
            st.stop()
        
        # Add missing columns for forecast
        if 'monthly_staff_cost' not in forecast_df.columns:
            if 'staff_needed' in forecast_df.columns:
                forecast_df['monthly_staff_cost'] = forecast_df['staff_needed'] * 25000
            else:
                st.warning("⚠️ 'staff_needed' column missing. Using default cost calculation.")
                forecast_df['staff_needed'] = (forecast_df['yhat'] * 0.001).astype(int)
                forecast_df['monthly_staff_cost'] = forecast_df['staff_needed'] * 25000
        
        if 'best_case' not in forecast_df.columns:
            forecast_df['best_case'] = forecast_df['yhat'] * 0.9
            forecast_df['expected'] = forecast_df['yhat']
            forecast_df['worst_case'] = forecast_df['yhat'] * 1.2
        
        # Ensure required columns exist
        if 'demand_risk' not in forecast_df.columns:
            forecast_df['demand_risk'] = pd.cut(forecast_df['yhat'], 
                                              bins=[0, forecast_df['yhat'].quantile(0.5), forecast_df['yhat'].max()], 
                                              labels=['Low', 'High'])
        
        if 'recommended_action' not in forecast_df.columns:
            forecast_df['recommended_action'] = np.where(forecast_df['demand_risk'] == 'High', 
                                                       'Recruit Now', 'Monitor')
        
        # Historical total updates
        if 'bio_age_5_17' in merged_df.columns and 'bio_age_17_' in merged_df.columns:
            merged_df['total_updates'] = merged_df['bio_age_5_17'] + merged_df['bio_age_17_']
        else:
            st.warning("⚠️ Age group columns missing. Using total_updates if available.")
            if 'total_updates' not in merged_df.columns:
                merged_df['total_updates'] = merged_df.iloc[:, 1:].sum(axis=1)
        
        st.success(f"✅ Loaded {len(forecast_df):,} forecast records & {len(merged_df):,} biometric records")
        return forecast_df, merged_df, DATE_COL
    
    return None, None, None

# --------------------------- SIDEBAR UPLOAD --------------------------- 
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/06/UIDAI_Logo.png", width=200)
    st.title("📁 Upload Datasets")
    
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    forecast_file = st.file_uploader("**Forecast Output CSV**", type="csv", 
                                   help="forecast_output.csv from your UIDAI project")
    merged_file = st.file_uploader("**Biometric Merged CSV**", type="csv", 
                                 help="biometric_merged.csv from your UIDAI project")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if forecast_file is not None and merged_file is not None:
        # Preview files
        st.subheader("📋 Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.read_csv(forecast_file).head(), use_container_width=True)
        with col2:
            st.dataframe(pd.read_csv(merged_file).head(), use_container_width=True)

# --------------------------- MAIN DATA LOADING --------------------------- 
@st.cache_data
def cached_load(forecast_file, merged_file):
    return load_uploaded_data(forecast_file, merged_file)

if 'forecast_file' not in st.session_state:
    st.session_state.forecast_file = None
if 'merged_file' not in st.session_state:
    st.session_state.merged_file = None

# Update session state when files change
if forecast_file != st.session_state.forecast_file or merged_file != st.session_state.merged_file:
    st.session_state.forecast_file = forecast_file
    st.session_state.merged_file = merged_file
    st.cache_data.clear()

forecast_df, merged_df, DATE_COL = cached_load(st.session_state.forecast_file, st.session_state.merged_file)

if forecast_df is None or merged_df is None:
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown("""
        <h3>👆 Upload Both CSV Files to Start</h3>
        <p><strong>Required Files:</strong></p>
        <ul>
            <li>📊 <code>forecast_output.csv</code></li>
            <li>📈 <code>biometric_merged.csv</code></li>
        </ul>
        <p>Files will be processed automatically once uploaded!</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --------------------------- HERO HEADER --------------------------- 
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<h1 class="main-header">🆔 Aadhaar Biometric Demand Intelligence</h1>', unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f77b4, #ff7b00); padding: 2rem; border-radius: 20px; text-align: center; color: white; font-weight: 700; font-size: 1.2rem;">
        AI-Powered Decision Support<br>for UIDAI Operations
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --------------------------- EXECUTIVE SUMMARY --------------------------- 
st.markdown('<h2 class="sub-header">📊 Executive Summary</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    avg_demand = int(forecast_df['yhat'].mean())
    st.metric("Avg Monthly Demand", f"{avg_demand:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    peak_demand = int(forecast_df['yhat'].max())
    st.metric("Peak Demand", f"{peak_demand:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    high_risk = (forecast_df['demand_risk'] == "High").sum()
    st.metric("High-Risk Months", high_risk)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    total_cost = int(forecast_df['monthly_staff_cost'].sum())
    st.metric("Total Staff Cost", f"₹{total_cost:,}")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- SIDEBAR FILTERS (Now uses uploaded data) --------------------------- 
with st.sidebar:
    st.subheader("🔍 Filters")
    
    if merged_df is not None and 'state' in merged_df.columns:
        states = st.multiselect(
            "Select States", 
            options=sorted(merged_df['state'].unique()), 
            default=sorted(merged_df['state'].unique())[:5]
        )
    else:
        states = []
    
    age_groups = st.multiselect(
        "Age Groups", 
        options=['5-17 years', '18+ years'],
        default=['5-17 years', '18+ years']
    )

# --------------------------- TABS (All your original content remains the same) --------------------------- 
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📋 Planning", "🏛️ Historical", "⚙️ Scenarios"])

with tab1:
    st.markdown('<h3 class="sub-header">12-Month Demand Forecast</h3>', unsafe_allow_html=True)
    
    fig_forecast = px.line(
        forecast_df, x='ds', y='yhat',
        title="Future Biometric Update Demand",
        labels={'yhat': 'Expected Updates', 'ds': 'Month'},
        template='plotly_white',
        line_shape='spline'
    )
    fig_forecast.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab2:
    st.markdown('<h3 class="sub-header">Risk-Based Action Plan</h3>', unsafe_allow_html=True)
    
    table_data = forecast_df[[
        'ds','yhat','demand_risk','staff_needed',
        'recommended_action','monthly_staff_cost'
    ]].rename(columns={
        'ds':'Month','yhat':'Expected','demand_risk':'Risk',
        'staff_needed':'Staff','recommended_action':'Action',
        'monthly_staff_cost':'Cost (₹)'
    })
    
    st.data_editor(
        table_data,
        column_config={
            "Risk": st.column_config.SelectboxColumn("Risk Level", options=["Low", "Medium", "High"]),
            "Cost (₹)": st.column_config.NumberColumn(format="₹%,.0f")
        },
        use_container_width=True,
        hide_index=True
    )

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 class="sub-header">Peak Load Months</h4>', unsafe_allow_html=True)
        monthly_hist = merged_df.groupby(DATE_COL)['total_updates'].sum().reset_index()
        peak_months = monthly_hist.nlargest(10, 'total_updates')
        
        fig_peak = px.bar(
            peak_months.rename(columns={DATE_COL: 'Month'}),
            x='Month', y='total_updates',
            title="Top 10 Peak Months",
            color='total_updates',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_peak, use_container_width=True)
    
    with col2:
        st.markdown('<h4 class="sub-header">Top States</h4>', unsafe_allow_html=True)
        filtered_merged = merged_df[merged_df['state'].isin(states)]
        state_demand = filtered_merged.groupby('state')['total_updates'].sum().sort_values(ascending=False)
        
        fig_states = px.bar(
            state_demand.reset_index(),
            x='total_updates', y='state',
            orientation='h',
            title="State-wise Demand",
            color='total_updates',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_states, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 class="sub-header">Demand Scenarios</h4>', unsafe_allow_html=True)
        scenario_df = forecast_df[['ds', 'best_case', 'expected', 'worst_case']].melt(
            id_vars='ds', var_name='Scenario', value_name='Demand'
        )
        fig_scenarios = px.line(
            scenario_df, x='ds', y='Demand', color='Scenario',
            title="Best/Expected/Worst Case Scenarios"
        )
        st.plotly_chart(fig_scenarios, use_container_width=True)
    
    with col2:
        st.markdown('<h4 class="sub-header">Staffing Costs</h4>', unsafe_allow_html=True)
        fig_cost = px.area(
            forecast_df, x='ds', y='monthly_staff_cost',
            title="Monthly Staffing Costs",
            labels={'monthly_staff_cost': 'Cost (₹)'}
        )
        st.plotly_chart(fig_cost, use_container_width=True)

# --------------------------- AGE ANALYSIS --------------------------- 
st.markdown('<h2 class="sub-header">👥 Demographics Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age_demand = merged_df[['bio_age_5_17','bio_age_17_']].sum()
    age_df = pd.DataFrame({
        "Age Group": ["5–17 years", "18+ years"],
        "Updates": age_demand.values
    })
    
    fig_age = px.pie(age_df, values='Updates', names='Age Group', 
                     title="Age Group Distribution",
                     hole=0.4, color_discrete_sequence=['#ff7f0e', '#1f77b4'])
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    filtered_merged = merged_df[merged_df['state'].isin(states)]
    if age_groups:
        age_data = {}
        for group in age_groups:
            if group == '5-17 years' and 'bio_age_5_17' in filtered_merged.columns:
                age_data[group] = filtered_merged['bio_age_5_17'].sum()
            elif group == '18+ years' and 'bio_age_17_' in filtered_merged.columns:
                age_data[group] = filtered_merged['bio_age_17_'].sum()
        
        if age_data:
            fig_age_bar = px.bar(
                pd.DataFrame(list(age_data.items()), columns=['Age Group', 'Updates']),
                x='Updates', y='Age Group', orientation='h',
                title="Filtered Age Analysis"
            )
            st.plotly_chart(fig_age_bar, use_container_width=True)

# --------------------------- FOOTER --------------------------- 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;'>
    🚀 Built for UIDAI Hackathon | AI-Driven Biometric Service Planning | Jan 2026
</div>
""", unsafe_allow_html=True)
