# app.py
# DEPLOY THIS TO A CLOUD SERVER (e.g., Render, Heroku, or PythonAnywhere)

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import itertools
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re
import warnings

warnings.filterwarnings('ignore')
app = Flask(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GOOGLE_SHEET_URL = "https://script.google.com/macros/s/AKfycbwuTc2PTZnIyCCXx0oDjWm285Gxf2O-TR0ntKPYGkEfNNPAFR6SdbAo4QlZZmC8PQEZ/exec"
MAX_NUM = 80
TOP_N_NUMBERS = 12
TOP_N_TICKETS = 10

# MODEL WEIGHTS (EXACTLY AS PROVIDED)
CORE_WEIGHTS = {
    'frequency': 1.4, 'gap_recency': 1.5, 'pattern': 1.0, 'cyclical': 1.0,
    'ml_ensemble': 1.3, 'cluster': 0.9, 'monte_carlo': 0.8, 'neighbor': 0.8
}

# ==============================================================================
# 1. DATA MANAGER (MODIFIED FOR GOOGLE SHEETS)
# ==============================================================================
class LotteryDataManager:
    def __init__(self):
        self.url = GOOGLE_SHEET_URL

    def load_data(self):
        try:
            print(f"Fetching data from Google Sheet...")
            r = requests.get(self.url)
            if r.status_code == 200:
                data = r.json()
                # Ensure data matches DataFrame structure
                df = pd.DataFrame(data)
                # Standardize columns if necessary (assuming JSON keys match CSV headers)
                if 'Draw_ID' in df.columns:
                    df['Draw_ID'] = pd.to_numeric(df['Draw_ID'])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading Google Sheet: {e}")
            return pd.DataFrame()

# ==============================================================================
# CORE MODELS (EXACTLY AS PROVIDED - COPIED)
# ==============================================================================
def frequency_model(history):
    scores = Counter()
    total_weight = 0
    for idx, draw in enumerate(history[-100:]):
        weight = (idx + 1) / 100
        total_weight += weight
        for num in draw: scores[num] += weight
    if scores:
        max_s = max(scores.values())
        return {k: v/max_s for k, v in scores.items()}
    return {}

def gap_recency_model(history):
    scores = {}
    for num in range(1, MAX_NUM + 1):
        indices = [i for i, d in enumerate(history) if num in d]
        if not indices:
            scores[num] = 0.1
            continue
        recency = len(history) - indices[-1]
        recency_score = max(0, 60 - recency) / 60
        gap_score = 0
        if len(indices) >= 3:
            gaps = np.diff(indices)
            gap_mean = np.mean(gaps)
            gap_std = np.std(gaps) if len(gaps) > 1 else 1
            current_gap = len(history) - indices[-1]
            if current_gap > gap_mean + gap_std:
                gap_score = min(1.0, (current_gap - gap_mean) / (gap_mean + 1))
        scores[num] = (recency_score * 0.6 + gap_score * 0.4)
    return scores

def pattern_match_model(history):
    scores = Counter()
    if len(history) < 10: return {}
    curr_sig = np.array([sum(d) for d in history[-5:]])
    for i in range(len(history) - 6):
        cand_sig = np.array([sum(d) for d in history[i:i+5]])
        if euclidean(curr_sig, cand_sig) < 45:
            weight = 1.0 / (euclidean(curr_sig, cand_sig) + 1)
            for num in history[i + 5]: scores[num] += weight
    if scores:
        max_s = max(scores.values())
        return {k: v/max_s for k, v in scores.items()}
    return {}

def cyclical_model(history):
    scores = {}
    for num in range(1, MAX_NUM + 1):
        appearances = [i for i, d in enumerate(history) if num in d]
        if len(appearances) < 3:
            scores[num] = 0.2
            continue
        gaps = np.diff(appearances)
        gap_mean = np.mean(gaps)
        current_gap = len(history) - appearances[-1]
        proximity = 1.0 / (1 + abs(current_gap - gap_mean) / (gap_mean + 1))
        scores[num] = proximity
    return scores

def neighbor_model(history):
    scores = {}
    last_draw = set(history[-1]) if history else set()
    for num in range(1, MAX_NUM + 1):
        score = 0
        if (num - 1) in last_draw: score += 0.5
        if (num + 1) in last_draw: score += 0.5
        if (num - 2) in last_draw or (num + 2) in last_draw: score += 0.2
        scores[num] = min(score, 1.0)
    return scores

def ml_ensemble_model(history):
    try:
        if len(history) < 50: return {}
        X, y_dict = [], defaultdict(list)
        start_train = max(10, len(history) - 100)
        for i in range(start_train, len(history)):
            features = []
            for draw in history[i-10:i]:
                features.extend([sum(draw), np.mean(draw)])
            X.append(features)
            for num in range(1, 31):
                y_dict[num].append(1 if num in history[i] else 0)
        X = np.array(X)
        predictions = {}
        for num in range(1, 31):
            if num in y_dict and sum(y_dict[num]) > 2:
                rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                rf.fit(X, y_dict[num])
                predictions[num] = rf.predict_proba(X[-1:])[0][1]
        return predictions
    except: return {}

def cluster_model(history):
    try:
        if len(history) < 40: return {}
        features = []
        for draw in history[-50:]:
            features.append([sum(draw), max(draw) - min(draw)])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
        clusters = kmeans.fit_predict(X_scaled)
        recent = clusters[-1]
        similar = Counter()
        for i, c in enumerate(clusters[:-1]):
            if c == recent:
                idx = len(history) - len(features) + i
                if idx + 1 < len(history): similar.update(history[idx + 1])
        if similar:
            max_s = max(similar.values())
            return {k: v/max_s for k, v in similar.items()}
    except: pass
    return {}

def monte_carlo_model(history):
    all_nums = [n for draw in history for n in draw]
    freq = Counter(all_nums)
    if not freq: return {}
    probs = np.array([freq.get(i, 0.1) for i in range(1, MAX_NUM + 1)])
    probs = probs / probs.sum()
    sim_res = Counter()
    for _ in range(500):
        sim_res.update(np.random.choice(range(1, MAX_NUM + 1), size=4, replace=False, p=probs))
    max_c = max(sim_res.values())
    return {k: v/max_c for k, v in sim_res.items()}

# ==============================================================================
# FUSION ENGINE (EXACTLY AS PROVIDED)
# ==============================================================================
class PrecisionFusionEngine:
    def __init__(self, history_data):
        self.history = history_data

    def predict(self, verbose=False):
        preds = {}
        preds['frequency'] = frequency_model(self.history)
        preds['gap_recency'] = gap_recency_model(self.history)
        preds['pattern'] = pattern_match_model(self.history)
        preds['cyclical'] = cyclical_model(self.history)
        preds['neighbor'] = neighbor_model(self.history)
        preds['ml_ensemble'] = ml_ensemble_model(self.history)
        preds['cluster'] = cluster_model(self.history)
        preds['monte_carlo'] = monte_carlo_model(self.history)

        all_n = set()
        for p in preds.values(): all_n.update(p.keys())

        results = []
        n_models = len(preds)
        for num in all_n:
            sc_list = []
            for name, p in preds.items():
                w = CORE_WEIGHTS.get(name, 1.0)
                sc_list.append(p.get(num, 0) * w)
            mean_score = np.mean(sc_list)
            agreement = sum(1 for s in sc_list if s > 0.1) / n_models
            results.append((num, mean_score, agreement))

        results.sort(key=lambda x: x[1], reverse=True)
        final = []
        for r in results[:TOP_N_NUMBERS]:
             conf = (r[2] * 0.7) + (min(r[1], 1.0) * 0.3)
             final.append((r[0], r[1], conf, r[2]))
        return final

def generate_tickets(top_predictions):
    top_nums = [x[0] for x in top_predictions]
    confidences = {x[0]: x[2] for x in top_predictions}
    tickets = []
    filter_levels = [
        {'min_sum': 110, 'min_spread': 12},
        {'min_sum': 60,  'min_spread': 8},
        {'min_sum': 20,  'min_spread': 4}
    ]
    for level in filter_levels:
        if len(tickets) >= TOP_N_TICKETS: break
        current_batch = []
        for combo in itertools.combinations(top_nums, 4):
            ticket = sorted(list(combo))
            s = sum(ticket)
            spread = ticket[-1] - ticket[0]
            if s < level['min_sum'] or s > 280: continue
            if spread < level['min_spread']: continue
            if any(t['Numbers'] == ticket for t in tickets): continue
            odd = sum(1 for n in ticket if n % 2 == 1)
            conf = np.mean([confidences[n] for n in ticket])
            score = sum(sc for num, sc, _, _ in top_predictions if num in ticket)
            current_batch.append({
                'Numbers': ticket, 'Sum': s, 'Spread': spread,
                'Odd/Even': f"{odd}/{4-odd}", 'Confidence': conf, 'Score': score
            })
        current_batch.sort(key=lambda x: x['Confidence'], reverse=True)
        tickets.extend(current_batch)
    df = pd.DataFrame(tickets)
    if not df.empty:
        df = df.drop_duplicates(subset=['Sum', 'Spread'])
        df = df.sort_values(['Confidence', 'Score'], ascending=[False, False])
        return df.head(TOP_N_TICKETS)
    return pd.DataFrame()

# ==============================================================================
# API ENDPOINT (MOBILE APP CONNECTS HERE)
# ==============================================================================
@app.route('/predict', methods=['GET'])
def predict():
    target_id = request.args.get('id')
    
    dm = LotteryDataManager()
    df = dm.load_data()
    
    if df.empty:
        return jsonify({"error": "Failed to load database from Google Sheet"})

    # Parse Numbers
    df['Parsed'] = df['Numbers'].apply(lambda x: [int(n) for n in str(x).replace('"', '').replace("'", "").split(',') if n.strip().isdigit()] if not pd.isna(x) else [])
    
    # Run Engine
    engine = PrecisionFusionEngine(df['Parsed'].tolist())
    top_predictions = engine.predict(verbose=False)
    
    # Generate Pool Response
    pool_data = []
    for p in top_predictions:
        pool_data.append({
            "number": int(p[0]),
            "score": float(p[1]),
            "confidence": float(p[2]),
            "agreement": float(p[3])
        })

    # Generate Tickets
    tickets = generate_tickets(top_predictions)
    tickets_data = []
    if not tickets.empty:
        for _, row in tickets.iterrows():
            tickets_data.append({
                "numbers": row['Numbers'],
                "sum": int(row['Sum']),
                "spread": int(row['Spread']),
                "odd_even": row['Odd/Even'],
                "confidence": float(row['Confidence'])
            })

    return jsonify({
        "draw_id": target_id,
        "elite_pool": pool_data,
        "champion_tickets": tickets_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)