import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import data_loader

# 1. Page Configuration
st.set_page_config(
    page_title="Kerala Flood Warning & Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark-theme styling helper (Streamlit overrides)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Controls (Simulation Inputs)
st.sidebar.header("🌊 Simulator Parameters")
st.sidebar.write("Adjust the environmental conditions below to predict the flood risk:")

# Input 1: Antecedent Rainfall (Soil Saturation)
antecedent_rain = st.sidebar.slider(
    "Antecedent Rainfall (June-July) (mm)",
    min_value=100,
    max_value=1200,
    value=857,  # July 2018 actual
    step=50,
    help="Cumulative rain over the previous month. High values saturate the soil, preventing further infiltration."
)

# Input 2: 3-Day Forecasted Rainfall
forecast_rain = st.sidebar.slider(
    "Forecasted 3-Day Rainfall (mm)",
    min_value=0,
    max_value=500,
    value=250,  # Medium-heavy spell
    step=10,
    help="Expected rainfall volume from incoming weather systems."
)

# Input 3: Initial Reservoir Storage
initial_storage = st.sidebar.slider(
    "Initial Reservoir Storage Capacity (%)",
    min_value=30,
    max_value=100,
    value=95,  # August 2018 actual
    step=5,
    help="Combined water levels in major dams prior to the storm."
)

# Input 4: Dam Release Strategy
release_strategy = st.sidebar.selectbox(
    "Dam Release Strategy",
    options=["Delayed Release (Hold water until FRL)", "Proactive Pre-Release (Controlled release early)"],
    index=0,
    help="Delayed release creates sudden massive spills during peak rains. Proactive release clears storage room early."
)

# Input 5: High Tide Level
tide_level = st.sidebar.slider(
    "Coastal Tide Level (meters)",
    min_value=0.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Sea level height. High tides create backpressure, blocking river discharge into the ocean."
)

# 3. Risk Calculation Engine (SCS-CN Runoff & Reservoir Logic)
def calculate_flood_risk(antecedent, forecast, storage, strategy, tide):
    # Step A: Soil Curve Number (CN) modification based on soil saturation
    # Kerala soil is clayey (base CN ~80). Antecedent moisture adjusts this.
    if antecedent < 300:
        cn = 68.0  # Dry soil (Class I)
    elif antecedent > 800:
        cn = 92.0  # Fully saturated soil (Class III)
    else:
        # Linear interpolation
        cn = 68.0 + (92.0 - 68.0) * ((antecedent - 300) / 500)
        
    # Potential maximum retention after runoff begins (S in mm)
    S = (25400.0 / cn) - 254.0
    
    # SCS-CN Runoff formula: Q = (P - 0.2S)^2 / (P + 0.8S)
    P = forecast
    if P > 0.2 * S:
        runoff = ((P - 0.2 * S) ** 2) / (P + 0.8 * S)
    else:
        runoff = 0.0
        
    runoff_coefficient = runoff / P if P > 0 else 0.0
    
    # Step B: Reservoir Inflow & Spill Simulation
    # Assume aggregate catchment area translates to inflow MCM
    inflow_mcm = 16.0 * P * runoff_coefficient
    total_capacity_mcm = 2000.0
    available_capacity_mcm = total_capacity_mcm * (100.0 - storage) / 100.0
    
    if strategy == "Proactive Pre-Release (Controlled release early)":
        # Proactive release creates a 20% additional buffer in storage
        available_capacity_mcm += total_capacity_mcm * 0.20
        # Reduce incoming storage load by early discharge
        spill_mcm = max(0.0, inflow_mcm - available_capacity_mcm)
        spill_risk = min(spill_mcm / 400.0, 1.0) * 40.0 # Weighted spill risk
    else:
        # Delayed release (like 2018) leads to sudden heavy emergency spill
        spill_mcm = max(0.0, inflow_mcm - available_capacity_mcm)
        spill_risk = min(spill_mcm / 300.0, 1.0) * 100.0
        
    # Step C: Tidal Backpressure contribution (up to 15% absolute addition)
    tide_contribution = (tide / 3.0) * 15.0
    
    # Step D: Aggregate Flood Risk Percentage Calculation
    runoff_risk = min(runoff / 200.0, 1.0) * 100.0
    
    # Overall risk calculation (weighted average normalized to 100%)
    overall_risk = (0.50 * runoff_risk) + (0.35 * spill_risk) + (0.15 * tide_contribution)
    overall_risk = min(max(overall_risk, 0.0), 100.0)
    
    return round(overall_risk, 1), round(runoff_coefficient, 2), round(spill_mcm, 1), cn

overall_risk, runoff_coeff, spill_vol, final_cn = calculate_flood_risk(
    antecedent_rain, forecast_rain, initial_storage, release_strategy, tide_level
)

# 4. Main UI Layout
st.markdown("<h1 class='main-title'>🌊 Kerala Flood Early Warning & Simulation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>A predictive model dashboard based on the Ministry of Earth Sciences (MoES) study of the 2018 Kerala Flood disaster.</p>", unsafe_allow_html=True)

# Risk Meter and Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Big dynamic percentage display
    st.metric(
        label="Predicted Flood Risk",
        value=f"{overall_risk}%"
    )

with col2:
    st.metric(
        label="Runoff Coefficient",
        value=f"{runoff_coeff}",
        help="The fraction of rainfall that turns into direct surface runoff. Higher values mean the soil is too saturated to absorb water."
    )

with col3:
    st.metric(
        label="Est. Reservoir Spill",
        value=f"{spill_vol} MCM",
        help="Million Cubic Meters of emergency water spillway release needed from reservoirs during the storm."
    )

with col4:
    st.metric(
        label="Soil Curve Number (CN)",
        value=f"{int(final_cn)}",
        help="A value from 0-100 indicating runoff potential. Saturated clayey soil yields a high CN close to 100."
    )

# 5. Alert Level Banners and Warnings
st.subheader("📢 Warning System & Recommended Actions")

if overall_risk >= 80.0:
    st.error(f"🔴 **RED ALERT (CRITICAL RISK: {overall_risk}%)**")
    st.markdown("""
        * **Situation:** High forecasted rainfall on fully saturated soil coupled with maximum reservoir capacities. Severe basin-wide inundation is imminent.
        * **Operational Directives:**
            1. **Evacuate:** Immediate evacuation of citizens in low-lying areas of Thrissur, Alappuzha, and downstream Periyar (Ernakulam).
            2. **Dam Operations:** Maximize spillway gates immediately to manage head levels; coordinate downstream notifications.
            3. **Fisheries & Coast:** Total ban on marine activities due to tidal backpressure blocking river mouth discharges.
    """)
elif overall_risk >= 60.0:
    st.warning(f"🟠 **ORANGE ALERT (HIGH RISK: {overall_risk}%)**")
    st.markdown("""
        * **Situation:** Soil moisture is saturated, and reservoir buffers are highly limited. Flood conditions are developing.
        * **Operational Directives:**
            1. Prepare rescue and relief camps near high-risk zones.
            2. Initiate controlled pre-releases from dams to clear capacity before the storm peak.
            3. Issue public warnings to avoid riverbanks and water-logged roads.
    """)
elif overall_risk >= 30.0:
    st.info(f"🟡 **YELLOW ALERT (MODERATE RISK: {overall_risk}%)**")
    st.markdown("""
        * **Situation:** Soil is partially wet, and dams can absorb most of the inflow if managed carefully. Low-lying areas may experience minor flooding.
        * **Operational Directives:**
            1. Monitor weather radar and quantitative precipitation forecasts (QPF) from IMD.
            2. Maintain standard operating procedures for reservoir buffers.
    """)
else:
    st.success(f"🟢 **GREEN STATUS (LOW RISK: {overall_risk}%)**")
    st.markdown("""
        * **Situation:** Environmental parameters are within safe thresholds. High soil infiltration capacity and ample reservoir buffer volumes.
        * **Operational Directives:** Normal monitoring; no active flood warnings.
    """)

# 6. Tabbed Workspace for Analysis and Charts
st.write("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Simulation Mechanics", 
    "📅 2018 Rainfall Timeline", 
    "🛢️ Reservoir Case Study (Idukki)", 
    "🔍 NWP Model Performance"
])

with tab1:
    st.subheader("⚙️ Understanding the Physics of the Simulation")
    st.write("""
        This simulator runs calculations based on the **SCS Curve Number (SCS-CN) method** 
        for runoff and hydrologic routing parameters detailed in the MoES report.
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Simulation Flow Chart")
        st.markdown("""
            1. **Antecedent Rainfall** determines the initial **Soil Saturation state** (Antecedent Moisture Condition - AMC Class I, II, or III).
            2. **Soil Class** selects the **effective Curve Number (CN)**. Clayey soils in Kerala reach a CN of **92** when wet, leading to extreme runoff.
            3. **Forecasted Rain** is routed through the SCS-CN formula to output the **Runoff Coefficient**.
            4. **Dam storage capacity** absorbs the runoff. If dams are already full and releases are **delayed**, a massive emergency spill is generated.
            5. **Tide height** adds backpressure, preventing water from draining, raising risk.
        """)
        
    with col_b:
        st.write("### Sensitivity Analysis")
        # Generate a small sensitivity chart showing how risk increases with forecast rain
        rain_range = np.linspace(0, 500, 50)
        temp_cn = final_cn
        temp_S = (25400.0 / temp_cn) - 254.0
        
        sim_runoff = []
        for r in rain_range:
            if r > 0.2 * temp_S:
                q = ((r - 0.2 * temp_S) ** 2) / (r + 0.8 * temp_S)
            else:
                q = 0.0
            sim_runoff.append(q)
            
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=rain_range, y=sim_runoff, name="Direct Runoff (mm)", line=dict(color="#EF4444", width=3)))
        fig_sens.update_layout(
            title="Rainfall vs. Direct Runoff (Current Soil State)",
            xaxis_title="Rainfall (mm)",
            yaxis_title="Direct Runoff (mm)",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_sens, use_container_width=True)

with tab2:
    st.subheader("🌧️ August 2018 Actual Daily Rainfall Curve")
    st.write("""
        This interactive plot shows the actual daily area-weighted rainfall recorded over Kerala 
        during the peak periods of August 2018. Note the two back-to-back heavy spells 
        (Aug 8-10 and Aug 14-17).
    """)
    
    df_rain = data_loader.get_august_daily_rainfall_2018()
    
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(
        x=df_rain["Date"], 
        y=df_rain["Rainfall (mm)"], 
        name="Realized Rainfall (mm)",
        marker_color="#3B82F6"
    ))
    fig_rain.add_trace(go.Scatter(
        x=df_rain["Date"], 
        y=df_rain["Normal Daily (mm)"], 
        name="Normal Daily Average (mm)",
        line=dict(color="#EF4444", dash="dash", width=2)
    ))
    
    fig_rain.update_layout(
        title="Kerala Daily Area-Weighted Rainfall (August 1-20, 2018)",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_rain, use_container_width=True)

with tab3:
    st.subheader("🛢️ Reservoir Spill vs Inflow (Idukki Basin)")
    st.write("""
        The 2018 disaster was aggravated by the timing of reservoir releases. As shown below, 
        water releases (spills) were kept very low while storage capacity was filling up. 
        On August 15th, as inflow surged to 390 MCM, the dam hit capacity, forcing a massive, 
        sudden emergency spill.
    """)
    
    df_idukki = data_loader.get_idukki_reservoir_2018()
    
    fig_idukki = go.Figure()
    fig_idukki.add_trace(go.Bar(
        x=df_idukki["Date"], 
        y=df_idukki["Inflow Volume (MCM)"], 
        name="Reservoir Inflow (MCM)",
        marker_color="#10B981"
    ))
    fig_idukki.add_trace(go.Bar(
        x=df_idukki["Date"], 
        y=df_idukki["Spill Release (MCM)"], 
        name="Spill Release (MCM)",
        marker_color="#EF4444"
    ))
    fig_idukki.add_trace(go.Scatter(
        x=df_idukki["Date"], 
        y=df_idukki["Storage Capacity (%)"], 
        name="Storage Level (%)",
        yaxis="y2",
        line=dict(color="#F59E0B", width=3)
    ))
    
    fig_idukki.update_layout(
        title="Idukki Reservoir Operations (August 1-19, 2018)",
        xaxis_title="Date",
        yaxis_title="Water Volumes (Million Cubic Meters - MCM)",
        yaxis2=dict(
            title="Storage Capacity (%)",
            overlaying="y",
            side="right",
            range=[85, 101]
        ),
        template="plotly_white",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_idukki, use_container_width=True)

with tab4:
    st.subheader("🔍 NWP Model Underestimation Analysis")
    st.write("""
        The MoES report highlights that operational weather forecasting models (such as GFS V14) 
        substantially underestimated the rainfall volume. Longer lead-time forecasts (Day 3 & Day 5) 
        missed the extreme nature of the Aug 15-16 spell completely, which delayed emergency dam releases.
    """)
    
    df_fc = data_loader.get_forecast_vs_actual_2018()
    
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(x=df_fc["Date"], y=df_fc["Observed (IMD-GPM)"], name="Observed Actual (IMD-GPM)", marker_color="#1E3A8A"))
    fig_fc.add_trace(go.Scatter(x=df_fc["Date"], y=df_fc["Day 1 Forecast (GFS V14)"], name="Day 1 Forecast", line=dict(color="#10B981", width=2)))
    fig_fc.add_trace(go.Scatter(x=df_fc["Date"], y=df_fc["Day 3 Forecast (GFS V14)"], name="Day 3 Forecast", line=dict(color="#F59E0B", width=2)))
    fig_fc.add_trace(go.Scatter(x=df_fc["Date"], y=df_fc["Day 5 Forecast (GFS V14)"], name="Day 5 Forecast", line=dict(color="#EF4444", width=2)))
    
    fig_fc.update_layout(
        title="NWP Forecast (GFS V14) vs. Realized Rainfall (Aug 14-18, 2018)",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    
    st.info("""
        💡 **College Project Note:** This chart highlights the scientific importance of data-driven 
        flood models. If meteorologists had a hybrid ML model correcting GFS forecast outputs 
        based on lower-level moisture convergence trends, dam pre-releases could have started 3 days earlier.
    """)
