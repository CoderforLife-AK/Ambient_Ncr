import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import kagglehub
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. PAGE LAYOUT CONFIGURATION
st.set_page_config(
    page_title="Ambient-NCR: Air Quality Assistant",
    page_icon="🍃",
    layout="wide"
)

# Custom Slate UI CSS styling injection
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    div[data-testid="stForm"] { background-color: #1e293b; border-color: #334155; border-radius: 1rem; }
    h1, h2, h3, h4 { color: #34d399 !important; }
    </style>
""", unsafe_allow_html=True)

# 2. CACHED CLOUD TRAINING PIPELINE (Runs only once when app boots up)
@st.cache_resource
def initialize_and_train_pipeline():
    try:
        # Download real Delhi AQI dataset straight from Kaggle in the cloud
        path = kagglehub.dataset_download("vishardmehta/delhi-pollution-aqi-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded repository.")
            
        df_raw = pd.read_csv(csv_files[0])
        df_raw.columns = df_raw.columns.str.strip()
        
        # Build features safely matching training sequences
        df = pd.DataFrame()
        df['PM2_5'] = df_raw.get('PM2.5', df_raw.get('pm25', np.random.uniform(20, 450, len(df_raw))))
        df['PM10'] = df_raw.get('PM10', df_raw.get('pm10', np.random.uniform(40, 500, len(df_raw))))
        df['NO2'] = df_raw.get('NO2', df_raw.get('no2', np.random.uniform(10, 120, len(df_raw))))
        df['Temp'] = df_raw.get('Temp', df_raw.get('temp', np.random.uniform(15, 40, len(df_raw))))
        df['Humidity'] = df_raw.get('Humidity', df_raw.get('humidity', np.random.uniform(30, 80, len(df_raw))))
        
        if 'AQI' in df_raw.columns:
            df['AQI'] = df_raw['AQI']
        else:
            df['AQI'] = (df['PM2_5'] * 0.7) + (df['PM10'] * 0.4) + (df['NO2'] * 0.3)
        df['AQI'] = df['AQI'].fillna(df['AQI'].median()).astype(int)
        
        if 'Region' in df_raw.columns:
            df['Region'] = df_raw['Region']
        elif 'city' in df_raw.columns:
            df['Region'] = df_raw['city']
        else:
            # Fallback regional arrays
            df['Region'] = np.random.choice(['Central Delhi', 'Noida', 'Gurugram', 'Ghaziabad'], len(df_raw))
            
        df_encoded = pd.get_dummies(df, columns=['Region'], drop_first=False)
        
        X = df_encoded.drop(columns=['AQI'])
        y = df_encoded['AQI']
        
        # Train model in cloud container memory
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Pull distinct target regions dynamically
        regions_list = [col.replace('Region_', '') for col in X.columns if col.startswith('Region_')]
        if not regions_list:
            regions_list = ['Central Delhi', 'Noida', 'Gurugram', 'Ghaziabad']
            
        return model, X.columns.tolist(), regions_list
    except Exception as e:
        st.error(f"Cloud setup crash: {e}")
        return None, [], []

# Human-readable warning thresholds mapping
def get_health_advisory(aqi: int) -> dict:
    if aqi <= 50:
        return {"cat": "Good", "color": "#00e400", "adv": "Air quality is satisfactory. Perfect day for outdoor physical activities."}
    elif aqi <= 100:
        return {"cat": "Satisfactory / Moderate", "color": "#e5c100", "adv": "Acceptable air quality. However, unusually sensitive individuals should minimize heavy outdoor exertion."}
    elif aqi <= 200:
        return {"cat": "Poor", "color": "#ff7e00", "adv": "Breathing discomfort possible for children, elderly, and those with lung/heart disease. Consider wearing an N95 mask outside."}
    elif aqi <= 300:
        return {"cat": "Very Poor", "color": "#ff0000", "adv": "Health alert! Everyone may experience health effects. Avoid prolonged outdoor exposure; keep windows closed."}
    else:
        return {"cat": "Severe / Hazardous", "color": "#7e0023", "adv": "Emergency conditions. High risk of respiratory impact for the entire population. Avoid all outdoor physical activity."}

# Run cloud initialization spinner
with st.spinner("Downloading dataset and initializing predictive brain framework..."):
    model, model_features, regions_trained = initialize_and_train_pipeline()

# 3. GRAPHICAL USER INTERFACE
st.title("Ambient-NCR 🍃")
st.caption("AI-Powered Regional Air Quality Tracker & Forecasting Assistant | 1M1B Sustainability Internship Project")
st.markdown("---")

left_panel, right_panel = st.columns([5, 7], gap="large")

with left_panel:
    st.subheader("Environmental Parameters")
    with st.form("aqi_form"):
        # Automatically lists all regions found in your dataset!
        selected_region = st.selectbox("Target Monitoring Region", options=regions_trained)
        
        c1, c2 = st.columns(2)
        with c1:
            pm2_5 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=65.0)
        with c2:
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=120.0)
            
        no2 = st.number_input("NO2 Concentration (ppb)", min_value=0.0, value=35.0)
        
        c3, c4 = st.columns(2)
        with c3:
            temp = st.number_input("Temperature (°C)", value=28.0)
        with c4:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)
            
        submit = st.form_submit_with_ui_button("Analyze & Forecast AQI", type="primary")

with right_panel:
    if submit and model is not None:
        try:
            # Build inference vector matrix template
            input_dict = {col: 0.0 for col in model_features}
            input_dict['PM2_5'] = float(pm2_5)
            input_dict['PM10'] = float(pm10)
            input_dict['NO2'] = float(no2)
            input_dict['Temp'] = float(temp)
            input_dict['Humidity'] = float(humidity)
            
            target_col = f"Region_{selected_region}"
            if target_col in input_dict:
                input_dict[target_col] = 1.0
                
            input_df = pd.DataFrame([input_dict])[model_features]
            predicted_aqi = int(np.clip(model.predict(input_df)[0], 0, 500))
            advisory = get_health_advisory(predicted_aqi)
            
            # Interactive metric template layout card injection
            st.markdown(f"""
                <div style="border: 2px solid {advisory['color']}; background-color: {advisory['color']}15; padding: 25px; border-radius: 15px; margin-bottom: 20px;">
                    <p style="text-transform: uppercase; font-size: 0.75rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em; margin: 0;">Predicted Output Index</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                        <div>
                            <h2 style="color: {advisory['color']} !important; margin: 0; font-size: 2rem; font-weight: 700;">{advisory['cat']}</h2>
                            <p style="font-size: 0.9rem; color: #cbd5e1; margin-top: 4px; margin-bottom: 0;">Forecasted location zone: <b>{selected_region}</b></p>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 4rem; font-weight: 900; line-height: 1; color: #f8fafc;">{predicted_aqi}</span>
                            <p style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: bold; margin: 0;">AQI Value</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"**AI-Generated Health Advisory:** {advisory['adv']}")
            
            st.markdown("---")
            st.caption("⚡ Framework Infrastructure: Dynamic On-Demand Scikit-Learn Container Pipeline | SDG 11.6 Target Compliance")
        except Exception as eval_err:
            st.error(f"Inference error: {eval_err}")
    else:
        st.markdown("""
            <div style="border: 1px dashed #475569; padding: 40px; text-align: center; border-radius: 15px; margin-top: 30px; background-color: #1e293b20;">
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">Enter environmental metrics and launch the forecasting pipeline to view insights.</p>
            </div>
        """, unsafe_allow_html=True)
