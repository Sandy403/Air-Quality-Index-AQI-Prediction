#!/bin/bash
echo "Building Docker image..."
docker build -t aqi-health-advisor .

echo "Running container..."
docker run -d -p 5000:5000 --name aqi-app \
  -e OPENWEATHER_API_KEY=$OPENWEATHER_API_KEY \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  aqi-health-advisor

echo "App running at http://localhost:5000"