from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
model = joblib.load('aqi_model.pkl')

OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY)

def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "#00e400"
    elif aqi <= 100: return "Moderate", "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200: return "Unhealthy", "#ff0000"
    elif aqi <= 300: return "Very Unhealthy", "#8f3f97"
    else: return "Hazardous", "#7e0023"

def get_health_advice(aqi, category, city, age=25, condition="none"):
    prompt = f"""
    You are a public health expert. A person in {city}, India has asked for health advice.
    Current Air Quality Index (AQI): {aqi}
    AQI Category: {category}
    Person's age: {age}
    Health condition: {condition}
    
    Give specific, practical health advice in 4-5 bullet points.
    Be direct and actionable. Include outdoor activity recommendations.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        float(data['pm25']),
        float(data['pm10']),
        float(data['no2']),
        float(data['co']),
        float(data['so2']),
        float(data['o3'])
    ]])
    aqi = model.predict(features)[0]
    aqi = round(float(aqi), 2)
    category, color = get_aqi_category(aqi)
    advice = get_health_advice(
        aqi, category,
        data.get('city', 'your city'),
        data.get('age', 25),
        data.get('condition', 'none')
    )
    return jsonify({
        'aqi': aqi,
        'category': category,
        'color': color,
        'advice': advice
    })

@app.route('/city', methods=['POST'])
def city_aqi():
    data = request.json
    city = data.get('city', 'Chennai')

    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    geo_response = requests.get(geo_url).json()

    if not geo_response:
        return jsonify({'error': 'City not found'}), 404

    lat = geo_response[0]['lat']
    lon = geo_response[0]['lon']

    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    aqi_data = requests.get(aqi_url).json()

    components = aqi_data['list'][0]['components']

    features = np.array([[
        components.get('pm2_5', 0),
        components.get('pm10', 0),
        components.get('no2', 0),
        components.get('co', 0) / 1000,
        components.get('so2', 0),
        components.get('o3', 0)
    ]])

    aqi = model.predict(features)[0]
    aqi = round(float(aqi), 2)
    category, color = get_aqi_category(aqi)
    advice = get_health_advice(
        aqi, category, city,
        data.get('age', 25),
        data.get('condition', 'none')
    )

    return jsonify({
        'aqi': aqi,
        'category': category,
        'color': color,
        'advice': advice,
        'components': components
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)