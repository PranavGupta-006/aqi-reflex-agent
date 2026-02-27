from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import logging
import os

app = Flask(__name__)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- GLOBAL OBJECTS ---
model = None
state_encoder = LabelEncoder()
county_encoder = LabelEncoder()

FEATURES = [
    'State_Enc',
    'County_Enc',
    'Prev_Year_AQI',
    'Prev_Year_Wind',
    'Temperature',
    'Humidity',
    'Wind Speed',
    'PM2.5'
]

DATA_PATH = "merged_annual_data.csv"

def simple_aqi(pm25):
    if pm25 <= 12: return pm25 * 4.16
    elif pm25 <= 35.4: return 51 + (pm25 - 12.1) * 2.09
    elif pm25 <= 55.4: return 101 + (pm25 - 35.5) * 2.46
    else: return 151 + (pm25 - 55.5) * 1.05

def get_aqi_category(val):
    if val <= 50: return {"label": "Good", "color": "#00e400"}
    if val <= 100: return {"label": "Moderate", "color": "#ffff00"}
    if val <= 150: return {"label": "Unhealthy for Sensitive Groups", "color": "#ff7e00"}
    if val <= 200: return {"label": "Unhealthy", "color": "#ff0000"}
    return {"label": "Hazardous", "color": "#7e0023"}

def train_model_on_startup():
    global model, state_encoder, county_encoder
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"{DATA_PATH} not found in Render environment.")

        df = pd.read_csv(DATA_PATH)

        # Encode
        df['State_Enc'] = state_encoder.fit_transform(df['State Name'])
        df['County_Enc'] = county_encoder.fit_transform(df['County Name'])

        # Sort & clean
        df = df.sort_values(by=['State Name', 'County Name', 'Year'])
        cols = ['PM2.5', 'Temperature', 'Humidity', 'Wind Speed']
        df[cols] = df.groupby(['State Name', 'County Name'])[cols].ffill()
        df = df.dropna(subset=['PM2.5'])

        # Feature engineering
        df['Current_AQI'] = df['PM2.5'].apply(simple_aqi)
        df['Prev_Year_AQI'] = df.groupby(['State Name', 'County Name'])['Current_AQI'].shift(1)
        df['Prev_Year_Wind'] = df.groupby(['State Name', 'County Name'])['Wind Speed'].shift(1)
        df['Target_Next_Year_AQI'] = df.groupby(['State Name', 'County Name'])['Current_AQI'].shift(-1)

        df_model = df.dropna(
            subset=[
                'Prev_Year_AQI',
                'Target_Next_Year_AQI',
                'Temperature',
                'Humidity',
                'Wind Speed'
            ]
        )

        if df_model.empty:
            raise ValueError("Dataset after preprocessing is empty.")

        X = df_model[FEATURES]
        y = df_model['Target_Next_Year_AQI']

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)
        print("✅ Model trained successfully.")

    except Exception as e:
        print(f"❌ CRITICAL STARTUP ERROR: {e}")
        model = None

train_model_on_startup()

# ---------------- ROUTES ---------------- #

@app.route('/')
def health():
    return "Server is running"

@app.route('/locations', methods=['GET'])
def get_locations():
    try:
        df = pd.read_csv(DATA_PATH)
        location_map = df.groupby('State Name')['County Name'].unique().to_dict()
        return jsonify({
            state: sorted(counties.tolist())
            for state, counties in location_map.items()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not available. Check server logs.'
        }), 500

    try:
        data = request.get_json()

        s_enc = state_encoder.transform([data.get('state')])[0]
        c_enc = county_encoder.transform([data.get('county')])[0]

        pm25 = float(data.get('pm25'))
        temp = float(data.get('temp'))
        hum  = float(data.get('humidity'))
        wind = float(data.get('wind'))

        curr_aqi = simple_aqi(pm25)

        input_vars = [[
            s_enc,
            c_enc,
            curr_aqi,
            wind,
            temp,
            hum,
            wind,
            pm25
        ]]

        prediction = model.predict(input_vars)[0]

        return jsonify({
            'success': True,
            'prediction': round(prediction, 1),
            'category': get_aqi_category(prediction)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# Needed for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
