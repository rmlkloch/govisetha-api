import os
from flask import Flask, jsonify, request
import requests
import json
import math
from collections import Counter

# LIGHTWEIGHT APP - NO PANDAS, NO SKLEARN (To prevent crashing)
app = Flask(__name__)

GOOGLE_SHEET_URL = "https://script.google.com/macros/s/AKfycbwuTc2PTZnIyCCXx0oDjWm285Gxf2O-TR0ntKPYGkEfNNPAFR6SdbAo4QlZZmC8PQEZ/exec"

@app.route('/')
def home():
    return "Server is Online. Use /predict?id=100"

@app.route('/predict')
def predict():
    try:
        # 1. Fetch Data
        print("Step 1: Connecting to Google Sheet...")
        r = requests.get(GOOGLE_SHEET_URL, timeout=15)
        
        if r.status_code != 200:
            return jsonify({"error": f"Google Sheet returned Code {r.status_code}"}), 500
            
        # 2. Parse JSON
        try:
            raw_data = r.json()
        except:
            return jsonify({"error": "Google Sheet returned HTML, not JSON. Check Permissions."}), 500

        # Handle different JSON structures
        if isinstance(raw_data, dict) and 'data' in raw_data:
            data = raw_data['data']
        elif isinstance(raw_data, list):
            data = raw_data
        else:
            return jsonify({"error": "Unknown JSON structure"}), 500

        # 3. Process Data (Pure Python, no Pandas)
        print(f"Step 2: Loaded {len(data)} rows.")
        
        # Simple Logic to prove it works
        # (Replacing heavy ML with simple frequency for testing)
        all_numbers = []
        for row in data:
            # Try to find the numbers column
            val = row.get('Numbers') or row.get('numbers') or row.get('results')
            if val:
                # Clean string: "10, 20, 30" -> [10, 20, 30]
                cleaned = str(val).replace('"', '').replace("'", "").split(',')
                nums = [int(n.strip()) for n in cleaned if n.strip().isdigit()]
                all_numbers.extend(nums)

        # Calculate simple frequency
        counts = Counter(all_numbers)
        top_12 = counts.most_common(12)
        
        # Format for App
        pool_data = []
        for num, count in top_12:
            pool_data.append({
                "number": num,
                "score": count / 100.0,
                "confidence": 0.9,
                "agreement": 0.8
            })
            
        # Fake Tickets for testing
        tickets_data = [{
            "numbers": [1, 2, 3, 4],
            "sum": 10,
            "spread": 3,
            "odd_even": "2/2",
            "confidence": 0.99
        }]

        return jsonify({
            "draw_id": request.args.get('id', '0'),
            "elite_pool": pool_data,
            "champion_tickets": tickets_data
        })

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return jsonify({"error": "Server Logic Crash", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)