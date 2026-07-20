import pandas as pd

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
    # Based on Fig 15 of the MoES report
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
    # Based on Fig 39 / Fig 40 (Underestimation of heavy rain spells)
    data = {
        "Date": ["Aug 14", "Aug 15", "Aug 16", "Aug 17", "Aug 18"],
        "Observed (IMD-GPM)": [41.0, 120.0, 140.0, 76.0, 40.0],
        "Day 1 Forecast (GFS V14)": [30.0, 85.0, 90.0, 60.0, 35.0],
        "Day 3 Forecast (GFS V14)": [18.0, 50.0, 55.0, 42.0, 20.0],
        "Day 5 Forecast (GFS V14)": [10.0, 25.0, 30.0, 22.0, 12.0]
    }
    return pd.DataFrame(data)
