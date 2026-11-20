"""Standalone Flask API: trains once at startup, serves all three brief endpoints."""
from datetime import date, datetime
import json
import numpy as np
import requests
from flask import Flask, jsonify, request
from train import train, ROOT

app = Flask(__name__)
model, metrics = train()
locations = [
    {'id': 'taj-mahal', 'name': 'Taj Mahal', 'lat': 27.1751, 'lng': 78.0421, 'capacity': 1200},
    {'id': 'manali', 'name': 'Manali', 'lat': 32.2432, 'lng': 77.1892, 'capacity': 800},
    {'id': 'varanasi', 'name': 'Dashashwamedh Ghat', 'lat': 25.3077, 'lng': 83.0102, 'capacity': 1000},
]

@app.get('/api/locations')
def get_locations():
    return jsonify(locations=locations)

@app.get('/api/metrics')
def get_metrics():
    return jsonify(metrics)

@app.post('/api/predict')
def predict():
    data = request.get_json(silent=True) or {}
    try:
        selected_date = date.fromisoformat(data['date'])
        hour = data['hour']
        if type(hour) is not int or not 0 <= hour <= 23:
            raise ValueError()
        idx = next(i for i, l in enumerate(locations) if l['id'] == data['locationId'])
    except (KeyError, ValueError, StopIteration):
        return jsonify(error='Valid locationId, date and hour (0–23) required'), 400
    loc = locations[idx]
    temps, rains, source = [None] * 24, [None] * 24, 'Weather unavailable · neutral model inputs'
    try:
        response = requests.get('https://api.open-meteo.com/v1/forecast', params={'latitude': loc['lat'], 'longitude': loc['lng'], 'hourly': 'temperature_2m,precipitation', 'timezone': 'Asia/Kolkata', 'start_date': selected_date.isoformat(), 'end_date': selected_date.isoformat()}, timeout=6)
        response.raise_for_status()
        weather = response.json()['hourly']
        temps, rains, source = weather['temperature_2m'], weather['precipitation'], 'Open-Meteo forecast'
    except (requests.RequestException, KeyError):
        pass
    features = [[idx, h, selected_date.weekday(), selected_date.month, temps[h] if temps[h] is not None else 23, rains[h] if rains[h] is not None else 0] for h in range(24)]
    counts = model.predict(np.array(features))
    hours = [{'hour': h, 'count': max(0, round(float(counts[h]))), 'percent': round(max(0, float(counts[h])) / loc['capacity'] * 100), 'temperature': temps[h], 'rain': rains[h], 'source': source} for h in range(24)]
    current = hours[hour]['count']
    windows = [{**h, 'reduction': max(0, round((1 - h['count'] / current) * 100)) if current else 0} for h in sorted(hours, key=lambda h: h['count'])[:3]]
    return jsonify(destination=loc, date=data['date'], hour=hour, hours=hours, recommendations=windows, metrics=metrics, synthetic=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
