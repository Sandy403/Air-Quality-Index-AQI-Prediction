# 🌬️ AQI Health Advisor

An AI-powered Air Quality Index (AQI) prediction system that provides personalized health advice based on real-time air pollution data.

> **SDG 3: Good Health and Well-being** — Helping people make informed decisions about their health based on air quality conditions.

---

## 🚀 Features

- 📊 **ML-based AQI Prediction** — Random Forest model trained on Indian city air quality data
- 🌍 **Live Pollutant Data** — Real-time data fetched from OpenWeatherMap API
- 🤖 **AI Health Advice** — Personalized recommendations powered by Groq LLM (LLaMA 3.1)
- 🏙️ **City Search** — Enter any Indian city and get instant AQI + health advice
- 🐳 **Dockerized** — Fully containerized for easy deployment anywhere

---

## 🏗️ Architecture
User Input (City Name)
↓
OpenWeatherMap API → Real Pollutant Data (PM2.5, PM10, NO2, CO, SO2, O3)
↓
Random Forest ML Model → AQI Score Prediction
↓
Groq LLM (LLaMA 3.1) → Personalized Health Advice
↓
Flask Web App → Clean UI Response

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | Scikit-learn, Random Forest |
| Dataset | Indian City Air Quality (city_day.csv) |
| LLM | Groq API — LLaMA 3.1 8B |
| Live Data | OpenWeatherMap Air Pollution API |
| Containerization | Docker |
| Version Control | Git, GitHub |

---

## 🐳 Run with Docker

```bash
docker pull nightfury192/aqi-health-advisor
docker run -p 5000:5000 --env-file .env nightfury192/aqi-health-advisor
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Sandy403/Air-Quality-Index-AQI-Prediction.git
cd Air-Quality-Index-AQI-Prediction
pip install -r requirements.txt
python app.py
```

---

## 🔑 Environment Variables

Create a `.env` file with:

OPENWEATHER_API_KEY=your_openweather_key_here
GROQ_API_KEY=your_groq_key_here

---

## 📦 DockerHub

→ https://hub.docker.com/r/nightfury192/aqi-health-advisor

---

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 31.20 | 59.10 | 0.809 |
| Random Forest | 20.59 | 40.48 | 0.910 |
| XGBoost | 21.83 | 42.38 | 0.901 |

✅ Random Forest selected as best model (R² = 0.91)

---

## 👨‍💻 Author

**Santhosh Kumar**
CSE AIML A
RA2311026050192
