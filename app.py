import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_point, predict_by_coordinates, batch_predict_csv, CROP_CLASSES

st.set_page_config(
    page_title="West Bengal Precision Soil & Crop Advisory System",
    page_icon="🌱",
    layout="wide"
)

# Header
st.title("🌱 West Bengal Soil Fertility, Crop Suitability & Fertilizer Advisory System")
st.markdown("""
**Academic Project:** Final Year B.Tech Computer Science (MAKAUT)  
**Study Area:** West Bengal Agro-Ecological Zones (Alluvial, Terai, Lateritic, Coastal, Mountain)  
**Machine Learning Architecture:** Leak-Free Preprocessing Pipeline with Extra Trees / Random Forest / HistGBM Inference
""")

tabs = st.tabs(["📝 Manual Soil Parameter Entry", "📍 Geospatial Coordinate Lookup", "📁 Batch CSV Upload", "ℹ️ System Documentation & Disclaimers"])

# Tab 1: Manual Entry
with tabs[0]:
    st.subheader("1. Enter Field Soil & Environmental Measurements")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("##### Core Soil Reaction")
        pH = st.number_input("Soil pH (1:2.5 H2O)", min_value=3.5, max_value=9.5, value=6.2, step=0.1)
        EC = st.number_input("EC (dS/m at 25°C)", min_value=0.01, max_value=4.0, value=0.45, step=0.05)
        OC = st.number_input("Organic Carbon (%)", min_value=0.1, max_value=2.5, value=0.65, step=0.05)
        
    with col2:
        st.markdown("##### Primary Macronutrients (kg/ha)")
        N = st.number_input("Available Nitrogen (N)", min_value=30.0, max_value=600.0, value=220.0, step=10.0)
        P = st.number_input("Available Phosphorus (P)", min_value=2.0, max_value=120.0, value=24.5, step=2.0)
        K = st.number_input("Available Potassium (K)", min_value=20.0, max_value=500.0, value=160.0, step=10.0)
        S = st.number_input("Available Sulphur (mg/kg)", min_value=1.0, max_value=80.0, value=18.0, step=2.0)
        
    with col3:
        st.markdown("##### DTPA Micronutrients (mg/kg)")
        Zn = st.number_input("Zinc (Zn)", min_value=0.05, max_value=5.0, value=0.75, step=0.05)
        Fe = st.number_input("Iron (Fe)", min_value=1.0, max_value=80.0, value=18.5, step=1.0)
        Mn = st.number_input("Manganese (Mn)", min_value=0.5, max_value=40.0, value=8.2, step=0.5)
        Cu = st.number_input("Copper (Cu)", min_value=0.05, max_value=5.0, value=0.90, step=0.05)
        B = st.number_input("Boron (B)", min_value=0.05, max_value=3.0, value=0.48, step=0.05)

    with col4:
        st.markdown("##### Meso-Climate & Geography")
        temp = st.number_input("Mean Annual Temp (°C)", min_value=8.0, max_value=35.0, value=25.2, step=0.5)
        rain = st.number_input("Annual Rainfall (mm)", min_value=200.0, max_value=5000.0, value=1580.0, step=50.0)
        rh = st.number_input("Relative Humidity (%)", min_value=30.0, max_value=100.0, value=75.0, step=2.0)
        season = st.selectbox("Cropping Season", ["Kharif", "Rabi", "Mixed", "Perennial"])
        soil_zone = st.selectbox("Agro-Ecological Zone", ["Alluvial", "Terai", "Lateritic", "Coastal", "Mountain"])
        lat = st.number_input("Latitude (°N)", value=23.24, step=0.1)
        lon = st.number_input("Longitude (°E)", value=87.86, step=0.1)

    if st.button("🚀 Generate Precision Agronomic Advisory", type="primary"):
        soil_input = {
            'soil_pH': pH, 'soil_EC_dS_m': EC, 'soil_organic_carbon_pct': OC,
            'available_N_kg_ha': N, 'available_P_kg_ha': P, 'available_K_kg_ha': K,
            'available_S_mg_kg': S, 'Zn_mg_kg': Zn, 'Fe_mg_kg': Fe, 'Mn_mg_kg': Mn,
            'Cu_mg_kg': Cu, 'B_mg_kg': B, 'annual_mean_temperature_C': temp,
            'annual_rainfall_mm': rain, 'relative_humidity_pct': rh,
            'latitude_approx': lat, 'longitude_approx': lon
        }
        with st.spinner("Executing leak-free feature transformation and ML inference..."):
            res = predict_point(soil_input, season=season, soil_zone=soil_zone)
            
        st.success("Advisory successfully generated!")
        
        # Results View
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.metric("Soil Fertility Index (SFI)", f"{res['Soil_Fertility_Index']} / 100", res['Fertility_Classification'])
            st.info(f"**Primary Limiting Nutrient:** {res['Primary_Limiting_Nutrient']}")
            
            st.markdown("##### 🌾 Top-3 Suitable Crop Recommendations")
            for c_info in res['Top_3_Suitable_Crops']:
                st.write(f"**{c_info['Crop']}** — {c_info['Confidence_Pct']}% Confidence")
                st.progress(c_info['Confidence_Pct'] / 100.0)
                
        with res_col2:
            st.markdown(f"##### 🧪 Site-Specific Fertilizer Prescription for {res['Top_3_Suitable_Crops'][0]['Crop']}")
            fert = res['Recommended_Fertilizer_kg_ha']
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("Nitrogen (N)", f"{fert['N (Urea basis)']} kg/ha")
            f_col2.metric("Phosphorus (P2O5)", f"{fert['P2O5 (SSP/DAP basis)']} kg/ha")
            f_col3.metric("Potassium (K2O)", f"{fert['K2O (MOP basis)']} kg/ha")
            
            st.markdown("##### 📊 Edaphic Nutrient Status Breakdown")
            st.json(res['Nutrient_Status'])
            
        if res['OOD_Uncertainty_Warning']:
            st.warning("⚠️ **Out-of-Distribution (OOD) Warning:** One or more edaphic values fall outside empirical training envelopes. Treat predictions with caution and seek official block agricultural testing.")

# Tab 2: Geospatial Lookup
with tabs[1]:
    st.subheader("2. Query Soil & Climate by Geographical Coordinates")
    geo_c1, geo_c2 = st.columns(2)
    with geo_c1:
        query_lat = st.number_input("Query Latitude (°N)", min_value=21.5, max_value=27.5, value=22.98, step=0.01)
        query_lon = st.number_input("Query Longitude (°E)", min_value=85.5, max_value=90.0, value=88.42, step=0.01)
        geo_season = st.selectbox("Target Season", ["Kharif", "Rabi", "Mixed"], key="geo_season")
        
        if st.button("📍 Interpolate & Predict Location Advisory"):
            with st.spinner("Querying spatial nearest-neighbor profile..."):
                geo_res = predict_by_coordinates(query_lat, query_lon, season=geo_season)
            st.success(f"Matched nearest profile in **{geo_res['Nearest_District']} District** ({geo_res['Nearest_Soil_Zone']} zone) at distance {geo_res['Interpolation_Distance_Deg']}°.")
            st.json(geo_res)

# Tab 3: Batch CSV
with tabs[2]:
    st.subheader("3. Batch Processing via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV containing soil parameters", type=['csv'])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Preview (First 5 records):")
        st.dataframe(batch_df.head(5))
        if st.button("⚡ Run Batch Advisory Inference"):
            temp_in = "temp_batch_input.csv"
            temp_out = "temp_batch_output.csv"
            batch_df.to_csv(temp_in, index=False)
            batch_predict_csv(temp_in, temp_out)
            res_df = pd.read_csv(temp_out)
            st.success(f"Successfully processed {len(res_df)} observations!")
            st.dataframe(res_df)
            st.download_button("📥 Download Batch Advisory Results", res_df.to_csv(index=False), "advisory_batch_results.csv")

# Tab 4: Documentation
with tabs[3]:
    st.subheader("System Documentation, Literature Foundations & Legal Disclaimers")
    st.markdown("""
    ### Scientific Anchors
    1. **Malo & Saha (2025):** Spatial variability of soil fertility and site-specific nutrient management in Barind Tract, West Bengal.
    2. **Singha et al. (2023):** Land suitability analysis for rice and potato using Fuzzy-AHP and ML in Hooghly, West Bengal.
    3. **Chowdhuri et al. (2026):** Toposequence-based assessment for sustainable agriculture in Bankura, West Bengal.

    ### Legal & Advisory Notice
    This software is a scientific machine learning prototype designed for academic benchmark evaluation and clinical decision-support prototyping. Prescriptions do not constitute certified state extension orders. Always validate recommendations against physical wet-chemistry Soil Health Card laboratory testing issued by the West Bengal Department of Agriculture.
    """)
