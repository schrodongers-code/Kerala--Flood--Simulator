import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import data_loader

# 1. Page Configuration
st.set_page_config(
    page_title="Kerala Flood Warning & ML Simulator",
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

# 2. Machine Learning Model Training (Cached for performance)
@st.cache_resource
def train_ml_model():
    # A. Generate/load 10 years of simulated monsoon records
    df = data_loader.generate_historical_monsoon_dataset(size=1200)
    
    # B. Define features (X) and target variable (y)
    feature_cols = [
        "antecedent_rain", 
        "forecast_rain", 
        "reservoir_storage", 
        "release_strategy", 
        "tide_level"
    ]
    X = df[feature_cols]
    y = df["flood_risk_pct"]
    
    # C. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # D. Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # E. Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": ["Antecedent Rain", "Forecast Rain", "Reservoir Storage", "Release Strategy", "Tide Level"],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # Return trained assets
    test_results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    return model, round(mae, 2), round(r2, 3), importance_df, test_results

# Load the trained model assets
model, model_mae, model_r2, feat_importance, test_df = train_ml_model()


# 3. Sidebar Controls (Live Simulation Inputs)
st.sidebar.header("🌊 Environment Simulator")
st.sidebar.write("Configure the inputs to feed into the trained Random Forest model:")

# Input 1: Antecedent Rainfall (Soil Saturation)
antecedent_rain = st.sidebar.slider(
    "Antecedent Rainfall (June-July) (mm)",
    min_value=100,
    max_value=1200,
    value=857,  # July 2018 actual
    step=50,
    help="Cumulative rain over the previous month. Saturates the soil, which triggers high runoff."
)

# Input 2: 3-Day Forecasted Rainfall
forecast_rain = st.sidebar.slider(
    "Forecasted 3-Day Rainfall (mm)",
    min_value=0,
    max_value=500,
    value=250,  # Medium-heavy spell
    step=10,
    help="Incoming storm forecast volume."
)

# Input 3: Initial Reservoir Storage
initial_storage = st.sidebar.slider(
    "Initial Reservoir Storage Capacity (%)",
    min_value=30,
    max_value=100,
    value=95,  # August 2018 actual
    step=5,
    help="Water storage levels in dams prior to the storm."
)

# Input 4: Dam Release Strategy
release_strategy_str = st.sidebar.selectbox(
    "Dam Release Strategy",
    options=["Delayed Release (Hold water until FRL)", "Proactive Pre-Release (Controlled release early)"],
    index=0,
    help="Delayed release creates sudden massive spills during peak rains. Proactive release clears storage room early."
)
# Map to numeric binary features (0 = Delayed, 1 = Proactive)
release_strategy = 0 if release_strategy_str == "Delayed Release (Hold water until FRL)" else 1

# Input 5: High Tide Level
tide_level = st.sidebar.slider(
    "Coastal Tide Level (meters)",
    min_value=0.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Sea tide height. High tides create backpressure at river mouths."
)


# 4. Live ML Inference / Prediction
input_data = pd.DataFrame([{
    "antecedent_rain": antecedent_rain,
    "forecast_rain": forecast_rain,
    "reservoir_storage": initial_storage,
    "release_strategy": release_strategy,
    "tide_level": tide_level
}])

# Predict using the trained Random Forest model
predicted_risk = model.predict(input_data)[0]
predicted_risk = round(np.clip(predicted_risk, 0.0, 100.0), 1)


# 5. Main UI Layout
st.markdown("<h1 class='main-title'>🌊 Kerala Flood Early Warning ML Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>A Machine Learning predictive simulator based on environmental data from the 2018 Kerala Flood report.</p>", unsafe_allow_html=True)

# Risk Meter and Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🤖 Random Forest Prediction",
        value=f"{predicted_risk}%",
        help="Predicted flood risk probability output by the trained Random Forest regressor."
    )

with col2:
    st.metric(
        label="Model R² Score",
        value=f"{model_r2}",
        help="Coefficient of determination. E.g., 0.98 means the model explains 98% of the flood risk variance."
    )

with col3:
    st.metric(
        label="Model Mean Absolute Error (MAE)",
        value=f"{model_mae}%",
        help="Average prediction error range on test records."
    )

with col4:
    st.metric(
        label="ML Model Algorithm",
        value="Random Forest",
        help="Random Forest Regressor (Ensemble Decision Trees) trained on 1200 historical daily records."
    )

# 6. Alert Level Banners and Warnings
st.subheader("📢 ML Prediction Warnings & Action Protocols")

if predicted_risk >= 80.0:
    st.error(f"🔴 **RED ALERT (CRITICAL RISK: {predicted_risk}%)**")
    st.markdown("""
        * **Situation:** Random Forest model predicts critical flood levels.
        * **Operational Directives:**
            1. **Evacuate:** Immediate evacuation of citizens in low-lying areas of Thrissur, Alappuzha, and Ernakulam.
            2. **Dam Operations:** Maximize spillway gates immediately; delayed dam operations represent the single highest risk parameter.
            3. **Fisheries & Coast:** Total ban on marine activities due to tidal backpressure blocking river mouths.
    """)
elif predicted_risk >= 60.0:
    st.warning(f"🟠 **ORANGE ALERT (HIGH RISK: {predicted_risk}%)**")
    st.markdown("""
        * **Situation:** High runoff and limited dam buffer volume. Flooding is highly likely.
        * **Operational Directives:**
            1. Prepare rescue and relief camps near high-risk zones.
            2. Initiate controlled pre-releases from dams to clear capacity before the storm peak.
            3. Issue public warnings to avoid riverbanks and water-logged roads.
    """)
elif predicted_risk >= 30.0:
    st.info(f"🟡 **YELLOW ALERT (MODERATE RISK: {predicted_risk}%)**")
    st.markdown("""
        * **Situation:** Moderate soil saturation and manageable reservoir buffers. Minor localized flooding expected.
        * **Operational Directives:**
            1. Monitor weather radar and quantitative precipitation forecasts (QPF) from IMD.
            2. Maintain standard operating procedures for reservoir buffers.
    """)
else:
    st.success(f"🟢 **GREEN STATUS (LOW RISK: {predicted_risk}%)**")
    st.markdown("""
        * **Situation:** Low flood risk forecast. Soils can absorb rainfall and dams have sufficient buffer space.
        * **Operational Directives:** Normal monitoring; no active flood warnings.
    """)

# 7. Tabbed Workspace for Analysis and Charts
st.write("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 ML Model Explainability",
    "⚙️ Simulation Mechanics", 
    "📅 2018 Rainfall Timeline", 
    "🛢️ Reservoir Case Study (Idukki)", 
    "🔍 NWP Model Performance"
])

with tab1:
    st.subheader("📊 Machine Learning Model Analytics")
    st.write("""
        Demonstrating **model evaluation metrics** and **feature explainability** for the trained Random Forest model.
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Model Feature Importance")
        st.write("Calculated feature contribution values showing which factors the ML model weights most heavily:")
        
        fig_feat = px.bar(
            feat_importance, 
            x="Importance", 
            y="Feature", 
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Feature": "Input Parameter", "Importance": "Weight Importance"}
        )
        fig_feat.update_layout(template="plotly_white", height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_feat, width="stretch")
        
    with col_b:
        st.write("### Test Set: Actual vs. Predicted")
        st.write("Evaluates model predictions vs. test target labels to demonstrate fit:")
        
        # Take a subset of 100 points for a clean scatter plot
        scatter_sample = test_df.sample(150, random_state=42)
        fig_scatter = px.scatter(
            scatter_sample, 
            x="Actual", 
            y="Predicted", 
            trendline="ols",
            labels={"Actual": "Actual Risk %", "Predicted": "Predicted Risk %"}
        )
        fig_scatter.update_traces(marker=dict(size=6, color="#10B981", opacity=0.7))
        fig_scatter.update_layout(template="plotly_white", height=300)
        st.plotly_chart(fig_scatter, width="stretch")

with tab2:
    st.subheader("⚙️ Understanding the Physics Behind the Data")
    st.write("""
        The synthetic data used to train the machine learning model was generated using **SCS Curve Number (SCS-CN) hydrology equations**
        mixed with Gaussian noise to simulate the typical noise found in physical environmental sensors.
    """)
    st.markdown("""
        * **Soil Wetness Index**: Saturated clayey soil in Kerala reaches a Curve Number of **92** when wet, causing rainfall-runoff rates to spike.
        * **Reservoir Spill**: If initial capacity is high and releases are **delayed**, a massive emergency release spike is added.
        * **Tidal Backpressure**: Prevented drainage capacity at river-mouth junctions due to high tides.
    """)

with tab3:
    st.subheader("🌧️ August 2018 Actual Daily Rainfall Curve")
    st.write("""
        This interactive plot shows the actual daily area-weighted rainfall recorded over Kerala 
        during the peak periods of August 2018. Note the two back-to-back heavy spells (Aug 8-10 and Aug 14-17).
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
    st.plotly_chart(fig_rain, width="stretch")

with tab4:
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
    st.plotly_chart(fig_idukki, width="stretch")

with tab5:
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
    st.plotly_chart(fig_fc, width="stretch")
    
    st.info("""
        💡 **College Project Note:** This chart highlights the scientific importance of data-driven 
        flood models. If meteorologists had a hybrid ML model correcting GFS forecast outputs 
        based on lower-level moisture convergence trends, dam pre-releases could have started 3 days earlier.
    """)
