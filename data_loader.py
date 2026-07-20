import pandas as pd
import numpy as np

def get_monthly_rainfall_2018():
    """Returns monthly rainfall comparison for June, July, and Aug 1-19, 2018."""
    data = {
        "Month": ["June 2018", "July 2018", "Aug 1-19, 2018"],
        "Realized Rainfall (mm)": [749.6, 857.4, 758.6],
        "Normal Rainfall (mm)": [649.8, 726.1, 287.6],
        "Percentage Departure (%)": [15.0, 18.0, 164.0]
    }
    return pd.DataFrame(data)

def get_august_daily_rainfall_2018():
    """Returns daily area-weighted rainfall in mm for Kerala from August 1-20, 2018."""
    data = {
        "Day": list(range(1, 21)),
        "Date": [f"Aug {i}, 2018" for i in range(1, 21)],
        "Rainfall (mm)": [28.0, 12.0, 13.0, 6.0, 4.0, 3.0, 10.0, 60.0, 66.0, 40.0, 20.0, 18.0, 36.0, 41.0, 120.0, 140.0, 76.0, 40.0, 25.0, 10.0],
        "Normal Daily (mm)": [15.0] * 20
    }
    return pd.DataFrame(data)

def get_idukki_reservoir_2018():
    """Returns simulated/extracted Idukki reservoir storage capacity and spill data (Aug 1 - 19, 2018)."""
    dates = [f"Aug {i}" for i in range(1, 20)]
    storage_pct = [91, 92, 93, 93.5, 94, 94.5, 95, 96, 97, 98, 98.5, 99, 99.2, 99.5, 99.9, 99.8, 97.5, 96, 95]
    inflow_mcm = [45, 30, 25, 20, 18, 15, 35, 120, 145, 90, 65, 50, 85, 110, 390, 420, 250, 130, 80]
    spill_mcm = [0, 0, 0, 0, 0, 0, 0, 10, 20, 45, 50, 50, 50, 60, 390, 410, 320, 180, 90]
    
    data = {
        "Date": dates,
        "Storage Capacity (%)": storage_pct,
        "Inflow Volume (MCM)": inflow_mcm,
        "Spill Release (MCM)": spill_mcm
    }
    return pd.DataFrame(data)

def get_forecast_vs_actual_2018():
    """Returns comparative rainfall forecast (GFS V14) vs actual (IMD GPM) for August 14-18, 2018."""
    data = {
        "Date": ["Aug 14", "Aug 15", "Aug 16", "Aug 17", "Aug 18"],
        "Observed (IMD-GPM)": [41.0, 120.0, 140.0, 76.0, 40.0],
        "Day 1 Forecast (GFS V14)": [30.0, 85.0, 90.0, 60.0, 35.0],
        "Day 3 Forecast (GFS V14)": [18.0, 50.0, 55.0, 42.0, 20.0],
        "Day 5 Forecast (GFS V14)": [10.0, 25.0, 30.0, 22.0, 12.0]
    }
    return pd.DataFrame(data)

def generate_historical_monsoon_dataset(size=1000, seed=42):
    """
    Generates a synthetic historical dataset representing various monsoon scenarios in Kerala
    based on the SCS-CN and reservoir logic from the MoES report.
    This dataset will be used to train our Machine Learning Regressor.
    """
    np.random.seed(seed)
    
    # 1. Environmental Features
    antecedent_rain = np.random.uniform(100, 1200, size)     # Soil wetness
    forecast_rain = np.random.uniform(0, 500, size)          # Storm intensity
    reservoir_storage = np.random.uniform(30, 100, size)     # Pre-storm dam levels
    release_strategy = np.random.choice([0, 1], size)        # 0 = Delayed, 1 = Proactive
    tide_level = np.random.uniform(0.0, 3.0, size)           # Tide backpressure
    
    # 2. Physics-inspired target generation with noise
    risk_labels = []
    
    for i in range(size):
        # A. Runoff coefficient (derived from CN)
        ant = antecedent_rain[i]
        fore = forecast_rain[i]
        
        if ant < 300:
            cn = 68.0
        elif ant > 800:
            cn = 92.0
        else:
            cn = 68.0 + (92.0 - 68.0) * ((ant - 300) / 500)
            
        S = (25400.0 / cn) - 254.0
        
        if fore > 0.2 * S:
            runoff = ((fore - 0.2 * S) ** 2) / (fore + 0.8 * S)
        else:
            runoff = 0.0
            
        coeff = runoff / fore if fore > 0 else 0.0
        runoff_risk = min(runoff / 200.0, 1.0) * 100.0
        
        # B. Reservoir Spill Risk
        inflow = 16.0 * fore * coeff
        avail = 2000.0 * (100.0 - reservoir_storage[i]) / 100.0
        
        if release_strategy[i] == 1: # Proactive
            avail += 2000.0 * 0.20 # Added buffer
            spill = max(0.0, inflow - avail)
            spill_risk = min(spill / 400.0, 1.0) * 40.0
        else: # Delayed
            spill = max(0.0, inflow - avail)
            spill_risk = min(spill / 300.0, 1.0) * 100.0
            
        # C. Tide Level Risk
        tide_risk = (tide_level[i] / 3.0) * 15.0
        
        # Combined risk percentage
        base_risk = (0.50 * runoff_risk) + (0.35 * spill_risk) + (0.15 * tide_risk)
        
        # Add random environmental noise to simulate real-world data variability
        noise = np.random.normal(0, 3.5)
        final_risk = np.clip(base_risk + noise, 0, 100)
        risk_labels.append(round(final_risk, 2))
        
    df = pd.DataFrame({
        "antecedent_rain": antecedent_rain,
        "forecast_rain": forecast_rain,
        "reservoir_storage": reservoir_storage,
        "release_strategy": release_strategy,
        "tide_level": tide_level,
        "flood_risk_pct": risk_labels
    })
    
    return df
