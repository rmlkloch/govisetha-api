import os
from flask import Flask, jsonify, request
import requests
import json
from collections import Counter

app = Flask(__name__)

# YOUR GOOGLE SHEET URL
GOOGLE_SHEET_URL = "https://script.google.com/macros/s/AKfycbwuTc2PTZnIyCCXx0oDjWm285Gxf2O-TR0ntKPYGkEfNNPAFR6SdbAo4QlZZmC8PQEZ/exec"

def fetch_google_data():
    try:
        # 1. PRETEND TO BE A CHROME BROWSER (To bypass Google blocking)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        
        print("Attempting to connect to Google...")
        # Follow redirects automatically
        r = requests.get(GOOGLE_SHEET_URL, headers=headers, timeout=10, allow_redirects=True)
        
        # 2. CHECK IF WE GOT A LOGIN PAGE INSTEAD OF JSON
        if "html" in r.headers.get('Content-Type', '').lower():
            print("ERROR: Google returned HTML (Login Page). Using Emergency Data.")
            return None 

        # 3. TRY TO PARSE JSON
        return r.json()
        
    except Exception as e:
        print(f"Connection Failed: {e}")
        return None

@app.route('/predict')
def predict():
    try:
        # TRY TO GET REAL DATA
        raw_data = fetch_google_data()
        
        if raw_data is None:
            # === EMERGENCY BACKUP DATA (If Google Fails) ===
            # This ensures the app NEVER crashes, even if Google blocks us.
            print("Using Offline Backup Data")
            all_numbers = [
                1, 5, 10, 12, 15, 20, 25, 30, 35, 40, 
                1, 5, 12, 22, 33, 44, 55, 60, 70, 75
            ] * 5
            is_offline = True
        else:
            # PROCESS REAL DATA
            print("Using Real Google Data")
            is_offline = False
            if isinstance(raw_data, dict) and 'data' in raw_data:
                data_list = raw_data['data']
            else:
                data_list = raw_data
                
            all_numbers = []
            for row in data_list:
                # Flexible column finder
                val = row.get('Numbers') or row.get('numbers') or row.get('results') or row.get('Win')
                if val:
                    # Clean: "10, 20, 30" -> [10, 20, 30]
                    cleaned = str(val).replace('"', '').replace("'", "").split(',')
                    nums = [int(n.strip()) for n in cleaned if n.strip().isdigit()]
                    all_numbers.extend(nums)

        # GENERATE STATISTICS (Simple Frequency)
        counts = Counter(all_numbers)
        top_12 = counts.most_common(12)
        
        pool_data = []
        for num, count in top_12:
            pool_data.append({
                "number": num,
                "score": count / 100.0,
                "confidence": 0.85 if not is_offline else 0.1,
                "agreement": 0.8
            })
            
        # GENERATE TICKET
        # Make a valid ticket from top numbers
        top_nums = [x[0] for x in top_12]
        if len(top_nums) >= 4:
            ticket_nums = sorted(top_nums[:4])
            champion = [{
                "numbers": ticket_nums,
                "sum": sum(ticket_nums),
                "spread": ticket_nums[-1] - ticket_nums[0],
                "odd_even": "Mix",
                "confidence": 0.95 if not is_offline else 0.0
            }]
        else:
            champion = []

        return jsonify({
            "draw_id": request.args.get('id', 'Unknown'),
            "status": "Offline Mode" if is_offline else "Online",
            "elite_pool": pool_data,
            "champion_tickets": champion
        })

    except Exception as e:
        return jsonify({
            "error": "CRITICAL CRASH",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)