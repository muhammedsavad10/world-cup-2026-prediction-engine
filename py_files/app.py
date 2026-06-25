import streamlit as st
import datetime
import textwrap
import re
import numpy as np
import pandas as pd
import math

from const import WCGroups, data_dir_path
from preprocess import load_data, get_match_features
import preprocess
from tournament_simulator import TournamentSimulator
import importlib
import reasoning_agent
importlib.reload(reasoning_agent)
from reasoning_agent import generate_match_analysis, MatchState, get_match_state, MatchAnalysisContext
from calibration import get_calibration_metrics
from news_provider import fetch_live_team_news
import live_results_manager
from transparency_explainer import generate_recalibration_explanation
import probability_audit
import json
import os

def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.split("\n"))

# Set wide mode config first
st.set_page_config(layout="wide", page_title="2026 FIFA World Cup AI Predictor")

# Detect whether the app is running locally or in production
IS_LOCAL_DEV = os.path.abspath(__file__).startswith("c:\\Users\\Muhammed Savad T M") or os.environ.get("STREAMLIT_DEV") == "True"

# Initialize centralized session state variables early to prevent AttributeError
if "num_runs" not in st.session_state:
    st.session_state.num_runs = 10000 if not IS_LOCAL_DEV else 1000

# Inject stadium background and exact custom layout CSS at the top of the app
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

/* Apply Outfit font to the entire page */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif;
}

/* Global Stadium Canvas background with overlay */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(10, 15, 20, 0.85), rgba(10, 15, 20, 0.9)), url('https://images.unsplash.com/photo-1518605368461-1ee125225fa5?q=80&w=2500');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Force All Headings to be Bright and Glowing */
h1, h2, h3 {
    color: #ffffff !important;
    text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
    text-align: center;
}

/* Fixed-Row CSS Grid Flowchart Tree */
.flowchart-grid {
    display: grid !important;
    grid-template-columns: repeat(5, 1fr);
    grid-gap: 25px;
    width: 100%;
    min-height: 1200px;
    align-items: center;
    padding: 10px;
}

.round-column {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    height: 100%;
}

.match-card-btn {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 12px !important;
    width: 100%;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    transition: all 0.2s ease-in-out !important;
    cursor: pointer;
}

.match-card-btn:hover {
    border-color: #FFD700 !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.4) !important;
    transform: scale(1.02) !important;
}

/* Flowchart Connections: Sleek golden arrows pointing right between columns */
.round-column:not(:last-child) .match-card-btn {
    position: relative;
}

.round-column:not(:last-child) .match-card-btn::after {
    content: '➔';
    position: absolute;
    right: -20px;
    top: 50%;
    transform: translateY(-50%);
    color: #FFD700;
    font-size: 20px;
    text-shadow: 0 0 8px rgba(255, 215, 0, 0.8);
    z-index: 10;
}

/* Completely Isolated Champion Box at Right Center */
.champion-showcase {
    grid-column: 5;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #FFD700, #FF8C00);
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
    color: black !important;
}

/* Style Streamlit Metric labels and values for readability */
[data-testid="stMetricLabel"] p {
    color: #FFD700 !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    text-shadow: 0px 0px 5px rgba(0,0,0,0.8);
}
[data-testid="stMetricValue"] div {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* Tab container overrides for stadium dark overlay integration */
div[data-testid="stTabBar"] {
    background-color: rgba(10, 15, 20, 0.85);
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}
button[data-baseweb="tab"] {
    font-size: 1.1em;
    font-weight: 600;
    color: #A0AEC0 !important;
    background-color: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFD700 !important;
    border-bottom: 3px solid #FFD700 !important;
}
.h2h-container {
    background: rgba(10, 15, 20, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 30px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# Comprehensive emoji flag dictionary for all 48 countries in 2026 World Cup
FLAGS = {
    'Mexico': '🇲🇽', 'South Africa': '🇿🇦', 'Czechia': '🇨🇿', 'South Korea': '🇰🇷',
    'Canada': '🇨🇦', 'Austria': '🇦🇹', 'Nigeria': '🇳🇬', 'Paraguay': '🇵🇾',
    'USA': '🇺🇸', 'Switzerland': '🇨🇭', 'Morocco': '🇲🇦', 'Bolivia': '🇧🇴',
    'Brazil': '🇧🇷', 'China': '🇨🇳', 'Norway': '🇳🇴', 'El Salvador': '🇸🇻',
    'Spain': '🇪🇸', 'Uruguay': '🇺🇾', 'Costa Rica': '🇨🇷', 'Jordan': '🇯🇴',
    'Argentina': '🇦🇷', 'Sweden': '🇸🇪', 'Cameroon': '🇨🇲', 'Northern Ireland': '🇬🇧',
    'Germany': '🇩🇪', 'Colombia': '🇨🇴', 'United Arab Emirates': '🇦🇪', 'France': '🇫🇷',
    'Senegal': '🇸🇳', 'Panama': '🇵🇦', 'Iraq': '🇮🇶', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'Croatia': '🇭🇷', 'Tunisia': '🇹🇳', 'New Zealand': '🇳🇿', 'Portugal': '🇵🇹',
    'Denmark': '🇩🇰', 'Egypt': '🇪🇬', 'Honduras': '🇭🇳', 'Belgium': '🇧🇪',
    'Japan': '🇯🇵', 'Algeria': '🇩🇿', 'Venezuela': '🇻🇪', 'Netherlands': '🇳🇱',
    'Iran': '🇮🇷', 'Serbia': '🇷🇸', 'Haiti': '🇭🇹',
    # New qualified nations for 2026 groups:
    'Bosnia and Herzegovina': '🇧🇦', 'Qatar': '🇶🇦', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'Australia': '🇦🇺', 'Turkey': '🇹🇷', 'Ivory Coast': '🇨🇮', 'Ecuador': '🇪🇨',
    'Curacao': '🇨🇼', 'Saudi Arabia': '🇸🇦', 'Cape Verde': '🇨🇻', 'Uzbekistan': '🇺🇿',
    'DR Congo': '🇨🇩', 'Ghana': '🇬🇭'
}

# Model Wrapper for JSON-based manual evaluation
class JSONLogisticRegressionModel:
    def __init__(self, model_data):
        self.model_data = model_data
        
    def predict_proba(self, X_list):
        mean = self.model_data["scaler"]["mean"]
        scale = self.model_data["scaler"]["scale"]
        coef = self.model_data["classifier"]["coef"]
        intercept = self.model_data["classifier"]["intercept"]
        
        results = []
        for X in X_list:
            X_scaled = [(X[i] - mean[i]) / scale[i] for i in range(7)]
            z = sum(X_scaled[i] * coef[i] for i in range(7)) + intercept
            prob_home = 1.0 / (1.0 + math.exp(-z))
            prob_away = 1.0 - prob_home
            results.append([prob_away, prob_home])
        return np.array(results)

# Cache data loading and JSON model weights loading
@st.cache_resource
def load_model_and_data():
    X, y, rankings_df = load_data()
    
    # Load JSON model coefficients
    model_json_path = os.path.join(data_dir_path, "logistic_regression_model.json")
    with open(model_json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
        
    pipeline = JSONLogisticRegressionModel(model_data)
    return pipeline, rankings_df

with st.spinner("Loading tournament stats and compiling Logistic Regression model..."):
    pipeline, rankings_df = load_model_and_data()

# Strictly filter out all legacy teams from history and rankings
GROUPS = WCGroups
qualified_teams = set()
for grp in GROUPS:
    for team in grp:
        qualified_teams.add(team)

# Cleanse preprocess stats
preprocess.final_team_stats = {
    team: stats for team, stats in preprocess.final_team_stats.items() if team in qualified_teams
}

# Filter rankings_df
rankings_df = rankings_df[rankings_df['country_full'].isin(qualified_teams)].copy()

# Pre-build dictionary of ranks to avoid slow pandas filter in loop
team_ranks = {}
for team in qualified_teams:
    rows = rankings_df[rankings_df['country_full'] == team]
    team_ranks[team] = rows['rank'].values[-1] if len(rows) > 0 else 50

def feature_generator(team1, team2):
    home_rank = team_ranks.get(team1, 50)
    away_rank = team_ranks.get(team2, 50)
    rank_diff = home_rank - away_rank
    h_stats = preprocess.final_team_stats.get(team1, {'weighted_wins': 0.0, 'weighted_goals': 0.0})
    a_stats = preprocess.final_team_stats.get(team2, {'weighted_wins': 0.0, 'weighted_goals': 0.0})
    
    return [
        home_rank, away_rank, rank_diff,
        np.log1p(h_stats['weighted_wins']), np.log1p(a_stats['weighted_wins']),
        np.log1p(h_stats['weighted_goals']), np.log1p(a_stats['weighted_goals'])
    ]

# Load Live Match Data globally to be used in all tabs
LIVE_RESULTS_URL = "https://raw.githubusercontent.com/muhammedsavad10/world-cup-2026-prediction-engine/main/data/world_cup_2026_live_results.json"
live_results_file = os.path.join(data_dir_path, "world_cup_2026_live_results.json")

local_data = live_results_manager.load_live_results(live_results_file)
remote_data = None
try:
    import requests
    response = requests.get(LIVE_RESULTS_URL, timeout=3)
    if response.status_code == 200:
        remote_data = response.json()
except Exception:
    pass
    
local_matches = local_data.get("matches", [])
remote_matches = remote_data.get("matches", []) if remote_data else []

if len(local_matches) >= len(remote_matches):
    results_data = local_data
else:
    results_data = remote_data
    
completed_matches = results_data.get("matches", [])
for m in completed_matches:
    if "status" not in m:
        m["status"] = "FT"

# Initialize simulator
simulator = TournamentSimulator(datetime.date(2026, 6, 11), pipeline, GROUPS, feature_generator)
simulator.gamma = 0.15
simulator.prob_cap = 0.12

# Load precomputed live probabilities and baseline probabilities globally
last_updated = results_data.get("last_updated", "2026-06-11T00:00:00Z")

@st.cache_data
def load_precomputed_probabilities_static(last_updated_str):
    probs_file = os.path.join(data_dir_path, "world_cup_2026_live_probabilities.json")
    if not os.path.exists(probs_file):
        return None, "File not found"
    try:
        with open(probs_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "probabilities" in data and isinstance(data["probabilities"], dict):
                return data["probabilities"], None
            return None, "Invalid schema structure"
    except Exception as e:
        return None, str(e)

probs_dict, load_error = load_precomputed_probabilities_static(last_updated)

if probs_dict is not None:
    live_probs = probs_dict
else:
    if not IS_LOCAL_DEV:
        st.error("### 🛠️ Live Prediction System Maintenance\n\nThe prediction engine is currently updating results or undergoing scheduled maintenance.\n\nAll tournament dashboard visuals are temporarily paused. Please check back in a few minutes.")
        st.stop()
    else:
        # Fallback simulation
        try:
            np.random.seed(42)
            live_probs = simulator.run_monte_carlo_simulation(completed_matches, num_runs=1000)
        except Exception as sim_err:
            st.error(f"Failed to execute local fallback simulation: {sim_err}")
            st.stop()

# Load baseline probabilities
baseline_file = os.path.join(data_dir_path, "world_cup_2026_baseline_probabilities.json")
baseline_probs = {}
if os.path.exists(baseline_file):
    try:
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_probs = json.load(f)
    except Exception as e:
        st.error(f"Error loading baseline probabilities: {e}")

# get_calibration_metrics is imported from calibration.py
pass

# Define the tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Tournament Simulation Pipeline", 
    "Head-to-Head Bracket Inspector", 
    "Live Tournament Dashboard",
    "⚙️ Tournament Structure Audit"
])

def parse_match_label(label):
    # Parses format: "Argentina(0.68) vs. Japan(0.32)"
    parts = label.split(" vs. ")
    team1_str = parts[0]
    team2_str = parts[1]
    
    t1_name = team1_str.split("(")[0].strip()
    t1_prob = float(team1_str.split("(")[1].replace(")", "").strip())
    
    t2_name = team2_str.split("(")[0].strip()
    t2_prob = float(team2_str.split("(")[1].replace(")", "").strip())
    
    return t1_name, t1_prob, t2_name, t2_prob

def get_winner_and_loser(label):
    t1_name, t1_prob, t2_name, t2_prob = parse_match_label(label)
    t1_flag = FLAGS.get(t1_name, '🏳️')
    t2_flag = FLAGS.get(t2_name, '🏳️')
    
    if t1_prob > t2_prob:
        winner_name, winner_flag = t1_name, t1_flag
        loser_name, loser_flag = t2_name, t2_flag
    else:
        winner_name, winner_flag = t2_name, t2_flag
        loser_name, loser_flag = t1_name, t1_flag
        
    return loser_flag, loser_name, winner_name, winner_flag

@st.dialog("Match Tactical Analytics Breakdown")
def render_completed_match_evidence(context, state):
    if state == MatchState.COMPLETED:
        from reasoning_agent import build_evidence_summary
        quality, completeness_str, _, checklist = build_evidence_summary(context)
        checklist_html = checklist.replace('\n', '<br>')
        
        # 1. Render badges
        col1, col2 = st.columns(2)
        with col1:
            quality_colors = {"High": "#28a745", "Medium": "#ffc107", "Basic": "#17a2b8"}
            color = quality_colors.get(quality, "#6c757d")
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; text-align: center; border-left: 5px solid {color};">
                <span style="font-size: 0.85em; color: #CBD5E0; text-transform: uppercase;">Evidence Quality</span><br>
                <strong style="font-size: 1.25em; color: {color};">{quality}</strong>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; text-align: center; border-left: 5px solid #FFD700;">
                <span style="font-size: 0.85em; color: #CBD5E0; text-transform: uppercase;">Evidence Completeness</span><br>
                <strong style="font-size: 1.25em; color: #FFD700;">{completeness_str}</strong>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Render Verified Facts Card or Checklist
        if quality == "Basic":
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1); text-align: left;">
                <strong style="color: #FFD700; display: block; margin-bottom: 8px; font-size: 1.1em;">📋 Available Evidence Checklist</strong>
                {checklist_html}
                <p style="color: #FF8C00; font-size: 0.9em; margin-top: 10px; font-style: italic;">
                    ⚠️ Detailed player-level statistics and match events are currently unavailable for this fixture.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            events_html = ""
            if context.match_events:
                events_html += "<strong style='color: #FFD700; display: block; margin-top: 10px; font-size: 1.05em;'>⚽ Match Events</strong>"
                for goal in context.match_events.get("goals", []):
                    events_html += f"• Goal: <b>{goal.get('player')}</b> at {goal.get('minute')}'<br>"
                for card in context.match_events.get("cards", []):
                    events_html += f"• Card: <b>{card.get('player')}</b> ({card.get('card')} card) at {card.get('minute')}'<br>"
                for sub in context.match_events.get("substitutions", []):
                    events_html += f"• Sub: <b>{sub.get('player_in')}</b> replaced <b>{sub.get('player_out')}</b> at {sub.get('minute')}'<br>"
            
            stats_html = ""
            if context.verified_statistics:
                stats_html += "<strong style='color: #FFD700; display: block; margin-top: 10px; font-size: 1.05em;'>📊 Verified Statistics</strong>"
                for k, v in context.verified_statistics.items():
                    stats_html += f"• {k.capitalize()}: <b>{v}</b><br>"
                    
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid rgba(255,215,0,0.2); text-align: left;">
                <h5 style="color: #FFD700; margin: 0 0 10px 0; text-align: left; font-size: 1.1em;">✅ Verified Match Facts</h5>
                {events_html}
                {stats_html}
            </div>
            """, unsafe_allow_html=True)

def show_match_dialog(t1_name, t1_prob, t2_name, t2_prob):
    t1_flag = FLAGS.get(t1_name, '🏳️')
    t2_flag = FLAGS.get(t2_name, '🏳️')
    
    # Extract mathematical deltas for RAG reasoning engine
    features = get_match_features(t1_name, t2_name, rankings_df)
    team_a_rank = features[0]
    team_b_rank = features[1]
    team_a_form = features[3]
    team_b_form = features[4]
    team_a_goals = features[5]
    team_b_goals = features[6]
    rank_diff = team_a_rank - team_b_rank
    form_diff = team_a_form - team_b_form
    goals_diff = team_a_goals - team_b_goals
    
    st.markdown(clean_html(f"""
    <div style="display: flex; justify-content: space-around; align-items: center; text-align: center; margin-bottom: 25px;">
        <div style="flex: 1;">
            <div style="font-size: 4.5em; line-height: 1;">{t1_flag}</div>
            <h3 style="margin: 10px 0 0 0; color: #FFFFFF; font-size: 1.4em;">{t1_name}</h3>
        </div>
        <div style="font-size: 1.8em; font-weight: 800; color: #FFD700; font-style: italic; padding: 0 15px;">VS</div>
        <div style="flex: 1;">
            <div style="font-size: 4.5em; line-height: 1;">{t2_flag}</div>
            <h3 style="margin: 10px 0 0 0; color: #FFFFFF; font-size: 1.4em;">{t2_name}</h3>
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    # Check if this match is completed
    match_record = None
    for m in completed_matches:
        h = m.get("home_team")
        a = m.get("away_team")
        if (h == t1_name and a == t2_name) or (h == t2_name and a == t1_name):
            match_record = m
            break
            
    state = get_match_state(match_record)
    
    if state == MatchState.COMPLETED:
        # Override t1_prob, t2_prob with the cached pre-kickoff predictions from the model
        t1_prob, t2_prob = get_cached_prediction(t1_name, t2_name, pipeline)
        h_score = match_record.get("home_score")
        a_score = match_record.get("away_score")
        winner = match_record.get("winner")
        
        metrics = get_calibration_metrics(t1_name, t2_name, t1_prob, t2_prob, match_record, baseline_probs, live_probs)
        
        h2h_html = clean_html(f'''
        <div class="h2h-container" style="background: rgba(10, 15, 20, 0.85); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 12px; padding: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
            <div style="display: flex; justify-content: space-around; align-items: center; text-align: center; flex-wrap: wrap; margin-bottom: 20px;">
                <div style="flex: 1; min-width: 120px; margin: 10px;">
                    <div style="font-size: 5em; line-height: 1;">{t1_flag}</div>
                    <h3 style="margin: 10px 0 5px 0; color: #FFFFFF; font-weight: 800; font-size: 1.6em;">{t1_name}</h3>
                    <div style="font-size: 2.8em; font-weight: 800; color: #FFFFFF; margin-top: 5px;">{h_score if t1_name == match_record.get("home_team") else a_score}</div>
                </div>
                <div style="flex: 0 0 80px; font-size: 2.2em; font-weight: 800; color: #FF8C00; font-style: italic; margin: 10px;">FT</div>
                <div style="flex: 1; min-width: 120px; margin: 10px;">
                    <div style="font-size: 5em; line-height: 1;">{t2_flag}</div>
                    <h3 style="margin: 10px 0 5px 0; color: #FFFFFF; font-weight: 800; font-size: 1.6em;">{t2_name}</h3>
                    <div style="font-size: 2.8em; font-weight: 800; color: #FFFFFF; margin-top: 5px;">{a_score if t1_name == match_record.get("home_team") else h_score}</div>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 20px 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; text-align: left; color: #CBD5E0; font-size: 0.95em;">
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">🔮 Forecast</strong>
                    • {t1_name}: {t1_prob*100:.1f}%<br>
                    • {t2_name}: {t2_prob*100:.1f}%
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">⚽ Outcome</strong>
                    • Score: <b>{t1_name} {h_score} – {a_score} {t2_name}</b>
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">🎯 Prediction</strong>
                    • Accuracy: <b>{metrics.is_correct}</b><br>
                    • Confidence: <b>{metrics.confidence}</b><br>
                    • Upset: <b>{metrics.upset_level}</b>
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">📈 Tournament Impact</strong>
                    • Champ Odds: {t1_name} ({metrics.t1_champ_change}) | {t2_name} ({metrics.t2_champ_change})<br>
                    • Qual Odds: {t1_name} ({metrics.t1_qual_change}) | {t2_name} ({metrics.t2_qual_change})
                </div>
            </div>
        </div>
        ''')
        st.markdown(h2h_html, unsafe_allow_html=True)
    elif state == MatchState.LIVE:
        h_score = match_record.get("home_score", 0)
        away_score = match_record.get("away_score", 0)
        curr_min = match_record.get("current_minute", "Unknown")
        
        st.markdown(clean_html(f"""
        <div style="text-align: center; background: rgba(255, 140, 0, 0.1); border: 1px solid rgba(255, 140, 0, 0.3); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <div style="font-size: 0.9em; color: #FF8C00; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">LIVE MATCH IN PROGRESS ({curr_min}')</div>
            <div style="font-size: 2.2em; font-weight: 800; color: #FFFFFF; margin: 5px 0;">{t1_name} {h_score} – {away_score} {t2_name}</div>
        </div>
        """), unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label=f"{t1_name}", value=f"{t1_prob*100:.1f}%")
            st.progress(t1_prob)
        with col_b:
            st.metric(label=f"{t2_name}", value=f"{t2_prob*100:.1f}%")
            st.progress(t2_prob)
    elif state == MatchState.UNKNOWN:
        st.warning("⚠️ **Match Status**: `Unknown`  \n**Reason**: Awaiting official confirmation.  \nPredictions and analysis will resume automatically once verified match data becomes available.")
    else:
        st.markdown("<h4 style='color: #FFD700; text-align: left; font-size: 1.1em; margin-bottom: 15px;'>Win Probabilities</h4>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label=f"{t1_name}", value=f"{t1_prob*100:.1f}%")
            st.progress(t1_prob)
        with col_b:
            st.metric(label=f"{t2_name}", value=f"{t2_prob*100:.1f}%")
            st.progress(t2_prob)
            
    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #FFD700; text-align: left; font-size: 1.1em; margin-bottom: 10px;'>🤖 AI Tactical Reasoning</h4>", unsafe_allow_html=True)
    
    if state == MatchState.UNKNOWN:
        st.info("AI reasoning is paused for this fixture.")
    else:
        # Calculate probability impact if completed
        probabilities_impact = None
        if state == MatchState.COMPLETED:
            probabilities_impact = {
                "team_a_baseline_champ": baseline_probs.get(t1_name, {}).get("champion", 0.0) * 100,
                "team_a_current_champ": live_probs.get(t1_name, {}).get("champion", 0.0) * 100,
                "team_b_baseline_champ": baseline_probs.get(t2_name, {}).get("champion", 0.0) * 100,
                "team_b_current_champ": live_probs.get(t2_name, {}).get("champion", 0.0) * 100,
                "team_a_baseline_qual": baseline_probs.get(t1_name, {}).get("group_qual", 0.0) * 100,
                "team_a_current_qual": live_probs.get(t1_name, {}).get("group_qual", 0.0) * 100,
                "team_b_baseline_qual": baseline_probs.get(t2_name, {}).get("group_qual", 0.0) * 100,
                "team_b_current_qual": live_probs.get(t2_name, {}).get("group_qual", 0.0) * 100
            }
            
        with st.spinner("Reasoning Agent analyzing team dynamics..."):
            team_a_news = fetch_live_team_news(t1_name)
            team_b_news = fetch_live_team_news(t2_name)
            context = MatchAnalysisContext(
                team_a=t1_name,
                team_b=t2_name,
                prob_a=t1_prob * 100,
                prob_b=t2_prob * 100,
                rank_diff=rank_diff,
                form_diff=form_diff,
                goals_diff=goals_diff,
                current_phase="pre_tournament",
                match_record=match_record,
                probabilities_impact=probabilities_impact,
                team_a_news=team_a_news,
                team_b_news=team_b_news,
                forecast_date=last_updated[:10],
                live_results_version=f"Matchday {len(completed_matches)}",
                match_events=match_record.get("match_events") if match_record else None,
                verified_statistics=match_record.get("verified_statistics") if match_record else None
            )
            analysis = generate_match_analysis(context)
    
    if analysis == "AI_TACTICAL_ANALYSIS_UNAVAILABLE":
        st.warning(
            "### ⚠️ AI Tactical Analysis Temporarily Unavailable\n\n"
            "The prediction engine and tournament simulator are fully operational.\n\n"
            "The optional AI tactical reasoning service is currently unavailable because the language model API is not configured.\n\n"
            "All tournament predictions remain valid."
        )
    else:
        render_completed_match_evidence(context, state)
        st.info(analysis)

# Process active URL dialogue callbacks early
if "selected_match" in st.query_params:
    match_label = st.query_params["selected_match"]
    st.query_params.clear()
    t1_n, t1_p, t2_n, t2_p = parse_match_label(match_label)
    show_match_dialog(t1_n, t1_p, t2_n, t2_p)

# Cache H2H model predictions for round-robin loop to optimize load time
@st.cache_data
def get_cached_prediction(team_a, team_b, _pipeline):
    feat = get_match_features(team_a, team_b, rankings_df)
    probs = _pipeline.predict_proba([feat])[0]
    return float(probs[1]), float(probs[0])

# Generate Group Stage round-robin pairings mathematically (72 games total)
group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
group_fixtures = []
for g_idx, group in enumerate(GROUPS):
    g_letter = group_letters[g_idx]
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            team_a = group[i]
            team_b = group[j]
            # Compute probabilities using cached prediction function
            prob_a, prob_b = get_cached_prediction(team_a, team_b, pipeline)
            group_fixtures.append({
                'stage': f"Group Stage: [Group {g_letter}]",
                'team_a': team_a,
                'team_b': team_b,
                'prob_a': prob_a,
                'prob_b': prob_b,
                'label': f"Group Stage: [Group {g_letter}] {team_a} vs {team_b}"
            })

# Combine all generated fixtures chronologically for selection
dropdown_matches = list(group_fixtures)

if 'simulated_labels' in st.session_state:
    labels = st.session_state['simulated_labels']
    for idx, label in enumerate(labels):
        t1, p1, t2, p2 = parse_match_label(label)
        if idx < 16: stage_name = "Round of 32"
        elif idx < 24: stage_name = "Round of 16"
        elif idx < 28: stage_name = "Quarter-Final"
        elif idx < 30: stage_name = "Semi-Final"
        elif idx == 30: stage_name = "Final"
        else: stage_name = "Third-Place Playoff"
        
        dropdown_matches.append({
            'stage': stage_name,
            'team_a': t1,
            'team_b': t2,
            'prob_a': p1,
            'prob_b': p2,
            'label': f"{stage_name}: {t1} vs {t2}"
        })

def make_html_match_card(label):
    loser_flag, loser_name, winner_name, winner_flag = get_winner_and_loser(label)
    
    card_html = f'''
    <a href="?selected_match={label}" target="_self" style="text-decoration: none; color: inherit; display: block; margin: 8px 0;">
        <div class="match-card-btn" style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 1.15em; display: flex; align-items: center; gap: 5px;">{loser_flag} <span style="font-weight: 300; font-size: 0.85em; color: #cbd5e0;">{loser_name}</span></span>
            <span style="color: #FFD700; font-size: 1.1em; font-weight: bold; margin: 0 5px;">➔</span>
            <span style="font-size: 1.15em; display: flex; align-items: center; gap: 5px;"><span>{winner_name}</span> {winner_flag}</span>
        </div>
    </a>
    '''
    return card_html

# Tab 1: Tournament Simulation Pipeline
with tab1:
    st.write("")
    
    # Professional Hero Header Section
    st.markdown(clean_html("""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 215, 0, 0.35); border-radius: 15px; padding: 30px; margin-bottom: 25px; text-align: center; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);">
        <h2 style="color: #FFD700 !important; font-size: 2.2em; font-weight: 800; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0 0 15px rgba(255,215,0,0.4);">🏆 FIFA WORLD CUP 2026 PREDICTION ENGINE</h2>
        <p style="color: #A0AEC0; font-size: 1.2em; font-weight: 600; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">AI-Powered Tournament Forecasting System</p>
        
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin-bottom: 25px;">
            <span style="background: rgba(255, 215, 0, 0.1); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 20px; padding: 6px 16px; font-size: 0.9em; font-weight: 600; color: #FFD700; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">⚽ 48 Teams</span>
            <span style="background: rgba(255, 215, 0, 0.1); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 20px; padding: 6px 16px; font-size: 0.9em; font-weight: 600; color: #FFD700; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">📊 Machine Learning Model</span>
            <span style="background: rgba(255, 215, 0, 0.1); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 20px; padding: 6px 16px; font-size: 0.9em; font-weight: 600; color: #FFD700; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">🧠 AI Reasoning Engine</span>
            <span style="background: rgba(255, 215, 0, 0.1); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 20px; padding: 6px 16px; font-size: 0.9em; font-weight: 600; color: #FFD700; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">🏆 FIFA 2026 Bracket</span>
            <span style="background: rgba(255, 215, 0, 0.1); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 20px; padding: 6px 16px; font-size: 0.9em; font-weight: 600; color: #FFD700; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">📈 Team Intelligence Dataset</span>
        </div>
        
        <div style="max-width: 800px; margin: 0 auto; display: flex; flex-direction: column; align-items: center; text-align: center; gap: 15px;">
            <div style="background: rgba(15, 23, 42, 0.5); border-radius: 10px; padding: 15px 25px; width: 100%; text-align: left; border-left: 4px solid #FFD700;">
                <p style="color: #E2E8F0; font-weight: 600; margin: 0 0 10px 0; font-size: 1.05em;">This platform combines:</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; color: #CBD5E0; font-size: 0.95em;">
                    <div>• Historical International Match Data</div>
                    <div>• FIFA Rankings</div>
                    <div>• Team Form Analysis</div>
                    <div>• Squad Intelligence</div>
                    <div>• Coach Intelligence</div>
                    <div>• Key Player Performance Metrics</div>
                    <div style="grid-column: span 2;">• FIFA-Compliant Knockout Bracket Logic</div>
                </div>
            </div>
            
            <div style="background: rgba(15, 23, 42, 0.5); border-radius: 10px; padding: 15px 25px; width: 100%; text-align: left; border-left: 4px solid #FFD700;">
                <p style="color: #E2E8F0; font-weight: 600; margin: 0 0 10px 0; font-size: 1.05em;">To simulate the FIFA World Cup 2026 and predict:</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; color: #CBD5E0; font-size: 0.95em;">
                    <div>✓ Group Stage Outcomes</div>
                    <div>✓ Round of 32 Qualifiers</div>
                    <div>✓ Knockout Progression</div>
                    <div>✓ Semi-Finalists</div>
                    <div>✓ Finalists</div>
                    <div>✓ Champion</div>
                    <div style="grid-column: span 2;">✓ Third Place Winner</div>
                </div>
            </div>
        </div>
        <p style="color: #A0AEC0; font-size: 0.95em; margin-top: 20px; font-style: italic; max-width: 700px; margin-left: auto; margin-right: auto;">The simulation uses a calibrated machine learning model and official FIFA 2026 tournament regulations to generate realistic tournament forecasts.</p>
    </div>
    """), unsafe_allow_html=True)

    if st.button("Run 2026 Simulation", use_container_width=True, type="primary"):
        with st.spinner("Executing simulation matches..."):
            completed_lookup = {}
            for m in completed_matches:
                if m.get("stage") == "group_stage":
                    completed_lookup[(m["home_team"], m["away_team"])] = m
                    completed_lookup[(m["away_team"], m["home_team"])] = m
                match_num = m.get("match_number")
                if match_num:
                    completed_lookup[int(match_num)] = m
            round_results, labels, odds = simulator.playKnockOuts(completed_lookup=completed_lookup)
            
            # Save labels to session state for Tab 2
            st.session_state['simulated_labels'] = labels
            st.session_state['round_results'] = round_results
            st.session_state['show_balloons'] = True

    if 'simulated_labels' in st.session_state:
        labels = st.session_state['simulated_labels']
        round_results = st.session_state['round_results']
        
        # Map labels by match number (Index 30 is final/104, Index 31 is third_place/103)
        match_labels = {}
        for idx, label in enumerate(labels):
            if idx == 30:
                mid = 104
            elif idx == 31:
                mid = 103
            else:
                mid = 73 + idx
            match_labels[mid] = label
            
        # Re-order rounds topologically so they align visually in the layout columns
        r32_order = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]
        r16_order = [89, 90, 93, 94, 91, 92, 95, 96]
        qf_order = [97, 98, 99, 100]
        sf_order = [101, 102]
        
        r32 = [match_labels[mid] for mid in r32_order]
        r16 = [match_labels[mid] for mid in r16_order]
        qf = [match_labels[mid] for mid in qf_order]
        sf = [match_labels[mid] for mid in sf_order]
        
        # Render column headers above the bracket container for clean alignment
        headers_html = '<div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 5px; padding: 0 10px;"><div style="width: 18%; text-align: center; color: #FFD700; font-weight: 800; text-transform: uppercase; font-size: 1.05em; border-bottom: 2px solid #FFD700; padding-bottom: 8px; letter-spacing: 1px; text-shadow: 0 0 5px rgba(255,215,0,0.3);">Round of 32</div><div style="width: 18%; text-align: center; color: #FFD700; font-weight: 800; text-transform: uppercase; font-size: 1.05em; border-bottom: 2px solid #FFD700; padding-bottom: 8px; letter-spacing: 1px; text-shadow: 0 0 5px rgba(255,215,0,0.3);">Round of 16</div><div style="width: 18%; text-align: center; color: #FFD700; font-weight: 800; text-transform: uppercase; font-size: 1.05em; border-bottom: 2px solid #FFD700; padding-bottom: 8px; letter-spacing: 1px; text-shadow: 0 0 5px rgba(255,215,0,0.3);">Quarter-Finals</div><div style="width: 18%; text-align: center; color: #FFD700; font-weight: 800; text-transform: uppercase; font-size: 1.05em; border-bottom: 2px solid #FFD700; padding-bottom: 8px; letter-spacing: 1px; text-shadow: 0 0 5px rgba(255,215,0,0.3);">Semi-Finals</div><div style="width: 18%; text-align: center; color: #FFD700; font-weight: 800; text-transform: uppercase; font-size: 1.05em; border-bottom: 2px solid #FFD700; padding-bottom: 8px; letter-spacing: 1px; text-shadow: 0 0 5px rgba(255,215,0,0.3);">Final & Champion</div></div>'
        st.markdown(headers_html, unsafe_allow_html=True)
        
        # Build the HTML elements for each column
        r32_html_cards = "".join([make_html_match_card(l) for l in r32])
        r16_html_cards = "".join([make_html_match_card(l) for l in r16])
        qf_html_cards = "".join([make_html_match_card(l) for l in qf])
        sf_html_cards = "".join([make_html_match_card(l) for l in sf])
        
        # Column 5 contains final and the gold champion card + third place card
        final_winner = round_results['final'][0]
        third_place_winner = round_results['third_place'][0]
        
        # Wrap the layout columns inside a single clean macro-string container
        complete_flowchart_html = f"""
<div class="flowchart-grid">
    <div class="round-column">
        {r32_html_cards}
    </div>
    
    <div class="round-column">
        {r16_html_cards}
    </div>
    
    <div class="round-column">
        {qf_html_cards}
    </div>
    
    <div class="round-column">
        {sf_html_cards}
    </div>
    
    <div class="round-column" style="justify-content: center; gap: 20px;">
        <div class="champion-showcase">
            <span style="font-size: 3em; line-height: 1;">🏆</span>
            <span style="font-size: 1.6em; display: block; margin: 10px 0; font-weight: 900;">{FLAGS.get(final_winner, '🏳️')} {final_winner}</span>
            <span style="font-size: 0.85em; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 800; color: rgba(0,0,0,0.6);">2026 World Cup Champion</span>
        </div>
        <div class="champion-showcase" style="background: linear-gradient(135deg, #a0aec0, #718096); box-shadow: 0 0 20px rgba(113, 128, 150, 0.5); padding: 15px;">
            <span style="font-size: 1.8em; line-height: 1;">🥉</span>
            <span style="font-size: 1.2em; display: block; margin: 5px 0; font-weight: 800; color: white;">{FLAGS.get(third_place_winner, '🏳️')} {third_place_winner}</span>
            <span style="font-size: 0.75em; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 800; color: rgba(255,255,255,0.8);">Third Place Winner</span>
        </div>
    </div>
</div>
"""
        
        # CRITICAL ACTION: Use st.html to render the tournament tree HTML string directly to prevent text dumping bugs
        st.html(complete_flowchart_html)
        
        # Trigger celebration balloons
        if st.session_state.get('show_balloons', False):
            st.balloons()
            st.session_state['show_balloons'] = False

# Tab 2: Head-to-Head Bracket Inspector
with tab2:
    st.header("Head-to-Head Bracket Inspector")
    st.markdown("Select any official Group Stage match or live simulated Knockout fixture to inspect win probabilities and AI tactical analysis.")
    
    dropdown_options = [m['label'] for m in dropdown_matches]
    selected_match_idx = st.selectbox("Select Match to Inspect:", range(len(dropdown_options)), format_func=lambda x: dropdown_options[x])
    
    selected_match = dropdown_matches[selected_match_idx]
    t1_name = selected_match['team_a']
    t2_name = selected_match['team_b']
    exact_t1_prob = selected_match['prob_a']
    exact_t2_prob = selected_match['prob_b']
    stage_name = selected_match['stage']
    
    f1, f2 = FLAGS.get(t1_name, '🏳️'), FLAGS.get(t2_name, '🏳️')
    
    # Extract mathematical deltas for RAG reasoning engine
    features = get_match_features(t1_name, t2_name, rankings_df)
    team_a_rank = features[0]
    team_b_rank = features[1]
    team_a_form = features[3]
    team_b_form = features[4]
    team_a_goals = features[5]
    team_b_goals = features[6]
    rank_diff = team_a_rank - team_b_rank
    form_diff = team_a_form - team_b_form
    goals_diff = team_a_goals - team_b_goals
    
    # Check if this match is completed in Tab 2
    match_record = None
    for m in completed_matches:
        h = m.get("home_team")
        a = m.get("away_team")
        if (h == t1_name and a == t2_name) or (h == t2_name and a == t1_name):
            match_record = m
            break
            
    state = get_match_state(match_record)
    
    if state == MatchState.COMPLETED:
        t1_prob, t2_prob = get_cached_prediction(t1_name, t2_name, pipeline)
        h_score = match_record.get("home_score")
        a_score = match_record.get("away_score")
        winner = match_record.get("winner")
        
        metrics = get_calibration_metrics(t1_name, t2_name, t1_prob, t2_prob, match_record, baseline_probs, live_probs)
        
        h2h_html = clean_html(f'''
        <div class="h2h-container" style="background: rgba(10, 15, 20, 0.85); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 12px; padding: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
            <div style="display: flex; justify-content: space-around; align-items: center; text-align: center; flex-wrap: wrap; margin-bottom: 20px;">
                <div style="flex: 1; min-width: 120px; margin: 10px;">
                    <div style="font-size: 5em; line-height: 1;">{f1}</div>
                    <h2 style="margin: 15px 0 5px 0; color: #FFFFFF; font-weight: 800; font-size: 1.8em;">{t1_name}</h2>
                    <div style="font-size: 2.8em; font-weight: 800; color: #FFFFFF; margin-top: 5px;">{h_score if t1_name == match_record.get("home_team") else a_score}</div>
                </div>
                <div style="flex: 0 0 80px; font-size: 2.2em; font-weight: 800; color: #FF8C00; font-style: italic; margin: 10px;">FT</div>
                <div style="flex: 1; min-width: 120px; margin: 10px;">
                    <div style="font-size: 5em; line-height: 1;">{f2}</div>
                    <h2 style="margin: 15px 0 5px 0; color: #FFFFFF; font-weight: 800; font-size: 1.8em;">{t2_name}</h2>
                    <div style="font-size: 2.8em; font-weight: 800; color: #FFFFFF; margin-top: 5px;">{a_score if t1_name == match_record.get("home_team") else h_score}</div>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 20px 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; text-align: left; color: #CBD5E0; font-size: 0.95em;">
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">🔮 Forecast</strong>
                    • {t1_name}: {t1_prob*100:.1f}%<br>
                    • {t2_name}: {t2_prob*100:.1f}%
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">⚽ Outcome</strong>
                    • Score: <b>{t1_name} {h_score if t1_name == match_record.get("home_team") else a_score} – {a_score if t1_name == match_record.get("home_team") else h_score} {t2_name}</b>
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">🎯 Prediction</strong>
                    • Accuracy: <b>{metrics.is_correct}</b><br>
                    • Confidence: <b>{metrics.confidence}</b><br>
                    • Upset: <b>{metrics.upset_level}</b>
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px;">
                    <strong style="color: #FFD700; display: block; margin-bottom: 5px;">📈 Tournament Impact</strong>
                    • Champ Odds: {t1_name} ({metrics.t1_champ_change}) | {t2_name} ({metrics.t2_champ_change})<br>
                    • Qual Odds: {t1_name} ({metrics.t1_qual_change}) | {t2_name} ({metrics.t2_qual_change})
                </div>
            </div>
        </div>
        ''')
        st.markdown(h2h_html, unsafe_allow_html=True)
        st.markdown("### Match Review & Probabilities")
        st.markdown(f"🎯 **AI Prediction Correctness**: `{metrics.is_correct}` (Predicted Winner: **{metrics.pred_winner}**, Actual Outcome: **{metrics.actual_winner}**)")
    elif state == MatchState.LIVE:
        h_score = match_record.get("home_score", 0)
        a_score = match_record.get("away_score", 0)
        curr_min = match_record.get("current_minute", "Unknown")
        
        st.markdown(clean_html(f"""
        <div style="text-align: center; background: rgba(255, 140, 0, 0.1); border: 1px solid rgba(255, 140, 0, 0.3); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <div style="font-size: 0.9em; color: #FF8C00; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">LIVE MATCH IN PROGRESS ({curr_min}')</div>
            <div style="font-size: 2.2em; font-weight: 800; color: #FFFFFF; margin: 5px 0;">{t1_name} {h_score} – {a_score} {t2_name}</div>
        </div>
        """), unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label=f"{t1_name}", value=f"{exact_t1_prob*100:.2f}%")
            st.progress(exact_t1_prob)
        with col_b:
            st.metric(label=f"{t2_name}", value=f"{exact_t2_prob*100:.2f}%")
            st.progress(exact_t2_prob)
    elif state == MatchState.UNKNOWN:
        st.warning("⚠️ **Match Status**: `Unknown`  \n**Reason**: Awaiting official confirmation.  \nPredictions and analysis will resume automatically once verified match data becomes available.")
    else:
        # Display the comparison card
        h2h_html = clean_html(f'''
        <div class="h2h-container">
            <div style="display: flex; justify-content: space-around; align-items: center; text-align: center; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 150px; margin: 10px;">
                    <div style="font-size: 6em; line-height: 1;">{f1}</div>
                    <h2 style="margin: 15px 0 5px 0; color: #FFFFFF; font-weight: 800; font-size: 1.8em;">{t1_name}</h2>
                    <div style="color: #A0AEC0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Home / Team A</div>
                </div>
                <div style="flex: 0 0 100px; font-size: 2.5em; font-weight: 800; color: #FF8C00; font-style: italic; margin: 10px;">VS</div>
                <div style="flex: 1; min-width: 150px; margin: 10px;">
                    <div style="font-size: 6em; line-height: 1;">{f2}</div>
                    <h2 style="margin: 15px 0 5px 0; color: #FFFFFF; font-weight: 800; font-size: 1.8em;">{t2_name}</h2>
                    <div style="color: #A0AEC0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Away / Team B</div>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 25px 0;">
            <div style="text-align: center;">
                <div style="font-size: 1.1em; color: #A0AEC0; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Match Category</div>
                <div style="font-size: 2.2em; font-weight: 800; color: #FFD700;">{stage_name}</div>
            </div>
        </div>
        ''')
        st.markdown(h2h_html, unsafe_allow_html=True)
        
        st.markdown("### Win Probability Mappings")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label=f"{f1} {t1_name} Win Probability", value=f"{exact_t1_prob*100:.2f}%")
            st.progress(exact_t1_prob)
        with col_b:
            st.metric(label=f"{f2} {t2_name} Win Probability", value=f"{exact_t2_prob*100:.2f}%")
            st.progress(exact_t2_prob)
            
    st.markdown("---")
    st.markdown("### 🤖 AI Tactical Reasoning")
    
    # Decide phase
    if "Group Stage" in stage_name:
        phase = "pre_tournament"
    else:
        phase = "post_group_stage"
        
    # Key for session state caching
    h2h_key = f"h2h_{t1_name}_{t2_name}"
    
    # Construct base context if match is completed to enable facts rendering
    context = None
    if state == MatchState.COMPLETED:
        probabilities_impact = {
            "team_a_baseline_champ": baseline_probs.get(t1_name, {}).get("champion", 0.0) * 100,
            "team_a_current_champ": live_probs.get(t1_name, {}).get("champion", 0.0) * 100,
            "team_b_baseline_champ": baseline_probs.get(t2_name, {}).get("champion", 0.0) * 100,
            "team_b_current_champ": live_probs.get(t2_name, {}).get("champion", 0.0) * 100,
            "team_a_baseline_qual": baseline_probs.get(t1_name, {}).get("group_qual", 0.0) * 100,
            "team_a_current_qual": live_probs.get(t1_name, {}).get("group_qual", 0.0) * 100,
            "team_b_baseline_qual": baseline_probs.get(t2_name, {}).get("group_qual", 0.0) * 100,
            "team_b_current_qual": live_probs.get(t2_name, {}).get("group_qual", 0.0) * 100
        }
        context = MatchAnalysisContext(
            team_a=t1_name,
            team_b=t2_name,
            prob_a=t1_prob * 100,
            prob_b=t2_prob * 100,
            rank_diff=rank_diff,
            form_diff=form_diff,
            goals_diff=goals_diff,
            current_phase=phase,
            match_record=match_record,
            probabilities_impact=probabilities_impact,
            team_a_news="",
            team_b_news="",
            forecast_date=last_updated[:10],
            live_results_version=f"Matchday {len(completed_matches)}",
            match_events=match_record.get("match_events") if match_record else None,
            verified_statistics=match_record.get("verified_statistics") if match_record else None
        )
    
    if state == MatchState.UNKNOWN:
        st.info("AI reasoning is paused for this fixture.")
    elif st.session_state.get(h2h_key):
        if state == MatchState.COMPLETED and context:
            render_completed_match_evidence(context, state)
        st.info(st.session_state[h2h_key])
    else:
        st.write("AI tactical analysis is deferred to optimize performance. Click below to generate analysis on demand.")
        if st.button("Generate AI Analysis", key="btn_h2h_ai"):
            with st.spinner("Agentic AI is analyzing..."):
                probabilities_impact = None
                if state == MatchState.COMPLETED:
                    probabilities_impact = {
                        "team_a_baseline_champ": baseline_probs.get(t1_name, {}).get("champion", 0.0) * 100,
                        "team_a_current_champ": live_probs.get(t1_name, {}).get("champion", 0.0) * 100,
                        "team_b_baseline_champ": baseline_probs.get(t2_name, {}).get("champion", 0.0) * 100,
                        "team_b_current_champ": live_probs.get(t2_name, {}).get("champion", 0.0) * 100,
                        "team_a_baseline_qual": baseline_probs.get(t1_name, {}).get("group_qual", 0.0) * 100,
                        "team_a_current_qual": live_probs.get(t1_name, {}).get("group_qual", 0.0) * 100,
                        "team_b_baseline_qual": baseline_probs.get(t2_name, {}).get("group_qual", 0.0) * 100,
                        "team_b_current_qual": live_probs.get(t2_name, {}).get("group_qual", 0.0) * 100
                    }
                
                if state == MatchState.COMPLETED:
                    prob_a_input = t1_prob * 100
                    prob_b_input = t2_prob * 100
                else:
                    prob_a_input = exact_t1_prob * 100
                    prob_b_input = exact_t2_prob * 100
                
                team_a_news = fetch_live_team_news(t1_name)
                team_b_news = fetch_live_team_news(t2_name)
                
                if not context:
                    context = MatchAnalysisContext(
                        team_a=t1_name,
                        team_b=t2_name,
                        prob_a=prob_a_input,
                        prob_b=prob_b_input,
                        rank_diff=rank_diff,
                        form_diff=form_diff,
                        goals_diff=goals_diff,
                        current_phase=phase,
                        match_record=match_record,
                        probabilities_impact=probabilities_impact,
                        team_a_news=team_a_news,
                        team_b_news=team_b_news,
                        forecast_date=last_updated[:10],
                        live_results_version=f"Matchday {len(completed_matches)}",
                        match_events=match_record.get("match_events") if match_record else None,
                        verified_statistics=match_record.get("verified_statistics") if match_record else None
                    )
                else:
                    context.team_a_news = team_a_news
                    context.team_b_news = team_b_news
                
                analysis = generate_match_analysis(context)
                if analysis == "AI_TACTICAL_ANALYSIS_UNAVAILABLE":
                    st.warning("🤖 AI Tactical Analysis is currently unavailable because the Groq language model API is not configured.")
                else:
                    st.session_state[h2h_key] = analysis
                    st.rerun()

# Tab 3: Live Tournament Dashboard
with tab3:
    st.header("🏆 Live Tournament Dashboard")
    st.markdown("Monitor dynamic tournament standings, advancement probabilities, momentum swings, and AI tactical reasoning updated continuously in real-time.")
    
    # 0. Probability Guide Widget
    with st.expander("💡 Why Did Probabilities Change? (Explanation Guide)"):
        st.markdown("""
        **Championship probabilities fluctuate dynamically as real matches are completed. Here are the three main causes for these shifts:**
        
        *   **1. Bracket Path Collisions (The 'Success Tax')**
            *   *What it is*: A team wins a match, but their overall title probability decreases.
            *   *Why it happens*: Winning a group stage match can lock a team into finishing 1st in their group. However, depending on outcomes in other groups, finishing 1st might steer them into a significantly harder side of the knockout bracket (e.g., meeting a heavyweight like Spain or France in the Quarterfinals) compared to finishing 2nd. 
        *   **2. Probability Normalization (The 'Crowding Effect')**
            *   *What it is*: A team does not play (or wins their match), yet their probability falls.
            *   *Why it happens*: The sum of all 48 teams' championship probabilities must always equal exactly 100%. If a major contender (e.g., England or Portugal) gains probability due to favorable outcomes elsewhere, the odds of all other teams must shrink slightly to compensate.
        *   **3. Monte Carlo Noise (Simulation Variance)**
            *   *What it is*: Small fluctuations (usually less than ±0.3%) for low-probability teams.
            *   *Why it happens*: Our simulation engine runs parallel iterations of the remaining tournament. For rare outcomes, small variation from run to run is normal statistical variance. At 1,000 runs, the 95% confidence margin for a 4% event is ±1.20%.
        """)
    
    # 1. Live Match Data has been loaded globally at startup
    last_updated = results_data.get("last_updated", "2026-06-11T00:00:00Z")
    
    last_updated_display = last_updated[:10]
    try:
        clean_ts = last_updated.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean_ts)
        last_updated_display = dt.strftime("%B %d, %Y")
    except Exception:
        pass
        
    completed_count = len(completed_matches)
    remaining_count = max(0, 104 - completed_count)
    
    # Hero status metrics
    st.markdown("### 📊 Live Tournament Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Matches Completed", value=f"{completed_count} / 104")
    with col2:
        st.metric(label="Matches Remaining", value=f"{remaining_count}")
    with col3:
        st.metric(label="Last Updated", value=last_updated_display)
        
    # 2. Invariant Standings & Probabilities Load (Utilizing globally loaded live_probs and baseline_probs)
    pass
            
    # Cached group tables logic to avoid slow rendering
    @st.cache_data
    def get_cached_group_tables(completed_matches_list, _rankings_df):
        return live_results_manager.calculate_group_tables(completed_matches_list, _rankings_df)
        
    # 3. Dynamic Group Tables
    st.markdown("---")
    st.markdown("### 📋 Current Group Standings")
    live_tables = get_cached_group_tables(completed_matches, rankings_df)
    group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    cols = st.columns(3)
    for idx, letter in enumerate(group_letters):
        col_idx = idx % 3
        with cols[col_idx]:
            with st.expander(f"Group {letter}"):
                table = live_tables[letter]
                df_table = pd.DataFrame(table)
                if not df_table.empty:
                    # Map group qualification probabilities from Monte Carlo results
                    df_table["Qual %"] = df_table["team"].map(
                        lambda t: f"{live_probs.get(t, {}).get('group_qual', 0.0) * 100:.1f}%"
                    )
                    df_table = df_table.rename(columns={
                        "team": "Team", "matches_played": "MP",
                        "wins": "W", "draws": "D", "losses": "L",
                        "goals_scored": "GS", "goals_conceded": "GA",
                        "goal_difference": "GD", "points": "Pts"
                    })
                    st.dataframe(df_table[["Team", "MP", "W", "D", "L", "GD", "Pts", "Qual %"]], hide_index=True, use_container_width=True)
                else:
                    st.write("Group table empty")
                    
    # 4. Probabilities & Momentum (Risers and Fallers)
    st.markdown("---")
    col_probs, col_momentum = st.columns(2)
    
    with col_probs:
        st.markdown("### 🏆 Championship Winner Probabilities")
        champ_probs = {team: probs["champion"] for team, probs in live_probs.items()}
        top_champs = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        df_champs = pd.DataFrame(top_champs, columns=["Team", "Probability"])
        df_champs["Probability"] = df_champs["Probability"] * 100
        st.bar_chart(df_champs, x="Team", y="Probability", use_container_width=True)
        
    with col_momentum:
        st.markdown("### 📈 Tournament Momentum (Championship Probability Delta)")
        
        # Collect played teams and their last match outcomes
        played_teams = set()
        last_results = {}
        for m in completed_matches:
            t1 = m["home_team"]
            t2 = m["away_team"]
            played_teams.add(t1)
            played_teams.add(t2)
            
            winner = m.get("winner")
            h_score = m.get("home_score")
            a_score = m.get("away_score")
            
            if winner == "Draw" or winner is None or h_score == a_score:
                last_results[t1] = 'draw'
                last_results[t2] = 'draw'
            elif winner == t1:
                last_results[t1] = 'win'
                last_results[t2] = 'loss'
            elif winner == t2:
                last_results[t1] = 'loss'
                last_results[t2] = 'win'
            
        risers = []
        fallers = []
        unchanged = []
        
        for team in live_probs:
            curr = live_probs[team]["champion"]
            base = baseline_probs.get(team, {}).get("champion", curr)
            delta = curr - base
            
            # Standard Error / Confidence Interval calculations for statistical significance
            p_bound = max(0.001, min(0.999, base))
            se_base = np.sqrt(p_bound * (1 - p_bound) / 10000)
            se_curr = np.sqrt(curr * (1 - curr) / st.session_state.num_runs) if curr > 0 else 0
            se_diff = np.sqrt(se_base**2 + se_curr**2)
            ci = 1.96 * se_diff
            
            if abs(delta) > ci and abs(delta) >= 0.001:
                last_res = last_results.get(team)
                if delta > 0:
                    if last_res != 'loss':
                        risers.append((team, curr, delta))
                    else:
                        unchanged.append((team, curr, delta))
                else:
                    if last_res != 'win':
                        fallers.append((team, curr, delta))
                    else:
                        unchanged.append((team, curr, delta))
            else:
                unchanged.append((team, curr, delta))
                
        col_r, col_f = st.columns(2)
        with col_r:
            st.markdown("##### Biggest Risers")
            sorted_risers = sorted(risers, key=lambda x: x[2], reverse=True)[:4]
            count_r = 0
            for team, curr, delta in sorted_risers:
                st.metric(label=team, value=f"{curr*100:.1f}%", delta=f"+{delta*100:.2f}%")
                count_r += 1
            if count_r == 0:
                st.write("No statistically significant risers yet.")
                
        with col_f:
            st.markdown("##### Biggest Fallers")
            sorted_fallers = sorted(fallers, key=lambda x: x[2])[:4]
            count_f = 0
            for team, curr, delta in sorted_fallers:
                st.metric(label=team, value=f"{curr*100:.1f}%", delta=f"{delta*100:.2f}%")
                count_f += 1
            if count_f == 0:
                st.write("No statistically significant fallers yet.")
                
        # Display Stable Paths for teams that have played but have insignificant movement
        unchanged_played = [t for t in unchanged if t[0] in played_teams]
        if unchanged_played:
            st.markdown("##### Stable Paths (Essentially Unchanged)")
            for team, curr, delta in sorted(unchanged_played, key=lambda x: abs(x[2])):
                st.markdown(f"• **{team}**: `🟡 Essentially Unchanged`")
                
    # 5. Transparency explainer (Spectator-focused with Advanced Analytics expander fallback)
    st.markdown("---")
    st.markdown("### 🔍 Live Probability Path Analyzer")
    st.markdown("Select a team to inspect their current tournament path and get a detailed, simulator-backed breakdown of their advancement probabilities.")
    
    selected_explain_team = st.selectbox("Inspect Team:", sorted(list(live_probs.keys())))
    
    # Retrieve stats for the selected team
    team_data = live_probs[selected_explain_team]
    curr_diff = team_data.get("avg_opponent_rank", 50.0)
    base_diff = baseline_probs.get(selected_explain_team, {}).get("avg_opponent_rank", 50.0)
    diff_change = curr_diff - base_diff
    
    # Calculate likely opponents from simulation frequencies
    curr_matchups = team_data.get("matchup_frequencies", {})
    all_opp_counts = {}
    for stage in ["round_of_32", "round_of_16"]:
        for opp, freq in curr_matchups.get(stage, {}).items():
            all_opp_counts[opp] = all_opp_counts.get(opp, 0) + freq
            
    sorted_opps = sorted(all_opp_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    col_path_info, col_insights = st.columns(2)
    
    with col_path_info:
        st.markdown("#### 🗺️ Most Likely Knockout Opponents")
        if sorted_opps:
            for opp, prob in sorted_opps:
                st.write(f"• **{opp}** ({prob * 100:.0f}%)")
        else:
            st.write("*No knockout matchups projected yet.*")
            
        st.markdown("#### Average Opponent FIFA Rank")
        st.write(f"**Baseline**: `{base_diff:.1f}`")
        st.write(f"**Current**: `{curr_diff:.1f}`")
        
        # Route status using simple thresholds
        if diff_change > 1.5:
            route_status = "🟢 Easier"
        elif diff_change < -1.5:
            route_status = "🔴 Harder"
        else:
            route_status = "🟡 Similar"
        st.markdown(f"**Route Status**: `{route_status}`")
        
        # Signal-To-Noise Outlook Status
        base_champ = baseline_probs.get(selected_explain_team, {}).get("champion", 0.0)
        curr_champ = team_data.get("champion", 0.0)
        delta_champ = curr_champ - base_champ
        
        p_bound = max(0.001, min(0.999, base_champ))
        se_base = np.sqrt(p_bound * (1 - p_bound) / 10000)
        se_curr = np.sqrt(curr_champ * (1 - curr_champ) / st.session_state.num_runs) if curr_champ > 0 else 0
        se_diff = np.sqrt(se_base**2 + se_curr**2)
        ci_bound = 1.96 * se_diff
        
        if abs(delta_champ) > ci_bound and abs(delta_champ) >= 0.001:
            if delta_champ > 0:
                status_str = f"🟢 Improved (+{delta_champ*100:.1f}%)"
            else:
                status_str = f"🔴 Decreased ({delta_champ*100:.1f}%)"
        else:
            status_str = "🟡 Essentially Unchanged"
            
        st.markdown(f"**Championship Outlook**: `{status_str}`")
        
    with col_insights:
        form_dict = live_results_manager.calculate_rolling_form(completed_matches)
        team_form = form_dict.get(selected_explain_team, "")
        
        explanation = generate_recalibration_explanation(
            selected_explain_team,
            baseline_probs.get(selected_explain_team, {}),
            team_data,
            completed_matches,
            team_form,
            num_runs=st.session_state.num_runs
        )
        
        st.markdown("#### Commentary Insight:")
        st.info(explanation["spectator_insight"])
        
    # Hide all statistical detail inside the Advanced Analytics expander
    with st.expander("⚙️ Advanced Analytics & Confidence Margins"):
        st.markdown("### 🎲 Monte Carlo Re-Simulation Settings")
        if not IS_LOCAL_DEV:
            st.selectbox(
                "Select Monte Carlo Simulation Count (Locked to precomputed in production):",
                [10000],
                index=0,
                disabled=True
            )
            # Ensure num_runs is set to 10000
            st.session_state.num_runs = 10000
        else:
            if "num_runs" not in st.session_state:
                st.session_state.num_runs = 1000
            new_runs = st.selectbox(
                "Select Monte Carlo Simulation Count (higher runs increase accuracy but take longer):",
                [100, 500, 1000, 5000, 10000],
                index=[100, 500, 1000, 5000, 10000].index(st.session_state.num_runs)
            )
            if new_runs != st.session_state.num_runs:
                st.session_state.num_runs = new_runs
                st.rerun()
            
        st.markdown("---")
        st.markdown("### 🔍 Audited Probability Movements")
        st.write("Audited significant shifts in championship probabilities compared to pre-tournament baseline, including statistical confidence intervals and root cause diagnoses:")
        
        audited_results = probability_audit.audit_probabilities(baseline_probs, live_probs, completed_matches, num_runs=st.session_state.num_runs)
        flagged_teams = [t for t, data in audited_results.items() if data["flagged"] or abs(data["delta"]) >= 0.01]
        
        if "Germany" in audited_results and "Germany" not in flagged_teams:
            flagged_teams.append("Germany")
            
        if flagged_teams:
            for team in sorted(flagged_teams):
                data = audited_results[team]
                delta_p = data["delta"]
                ci_p = data["ci"]
                
                with st.container(border=True):
                    col_t, col_vals, col_cause = st.columns([1, 1, 2])
                    with col_t:
                        st.markdown(f"**Team: {team}**")
                        if data["case"]:
                            st.caption(f"⚠️ {data['case']}")
                    with col_vals:
                        st.write(f"Previous: {data['previous']*100:.2f}%")
                        st.write(f"Current: {data['current']*100:.2f}%")
                        color = "red" if delta_p < 0 else "green"
                        st.markdown(f"Change: <span style='color:{color}; font-weight:bold;'>{delta_p*100:+.2f}%</span>", unsafe_allow_html=True)
                        st.write(f"Confidence Interval: ±{ci_p*100:.2f}%")
                    with col_cause:
                        st.markdown(f"**Likely Cause**: `{data['cause']}`")
                        st.write(data["explanation"])
        else:
            st.write("No major shifts or anomalies flagged.")
            
        st.markdown("---")
        st.markdown(explanation["advanced_analytics"])

# Tab 4: Tournament Structure Audit
with tab4:
    st.header("⚙️ Tournament Structure Audit")
    st.markdown("Verify the structural integrity, group distributions, and dataset lookup mappings for the World Cup 2026 Prediction Engine.")
    
    # Display group distributions
    st.markdown("### 📋 Group Stage Team Distributions")
    group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    # Show in a 4-column layout for visual excellence
    cols = st.columns(4)
    for idx, letter in enumerate(group_letters):
        col_idx = idx % 4
        with cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"**Group {letter}**")
                for team in GROUPS[idx]:
                    flag = FLAGS.get(team, '🏳️')
                    st.write(f"{flag} {team}")
                    
    # Calculate totals
    flat_teams = [team for grp in GROUPS for team in grp]
    total_teams = len(flat_teams)
    unique_teams = set(flat_teams)
    
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Total Teams Listed", value=str(total_teams))
    with col_stat2:
        st.metric(label="Unique Teams", value=str(len(unique_teams)))
        
    st.markdown("---")
    st.markdown("### 🔍 Validation Checks")
    
    validation_passed = True
    
    # 1. Exactly 12 groups
    if len(GROUPS) == 12:
        st.success("✅ **1. Exactly 12 groups** (Passed)")
    else:
        st.error(f"❌ **1. Exactly 12 groups** (Failed: Found {len(GROUPS)} groups)")
        validation_passed = False
        
    # 2. Exactly 4 teams per group
    wrong_groups = [group_letters[i] for i, grp in enumerate(GROUPS) if len(grp) != 4]
    if not wrong_groups:
        st.success("✅ **2. Exactly 4 teams per group** (Passed)")
    else:
        st.error(f"❌ **2. Exactly 4 teams per group** (Failed in groups: {', '.join(wrong_groups)})")
        validation_passed = False
        
    # 3. Exactly 48 unique teams
    if len(unique_teams) == 48:
        st.success("✅ **3. Exactly 48 unique teams** (Passed)")
    else:
        st.error(f"❌ **3. Exactly 48 unique teams** (Failed: Found {len(unique_teams)} unique teams)")
        validation_passed = False
        
    # 4. No duplicate teams
    duplicates = set([t for t in flat_teams if flat_teams.count(t) > 1])
    if not duplicates:
        st.success("✅ **4. No duplicate teams** (Passed)")
    else:
        st.error(f"❌ **4. No duplicate teams** (Failed: Duplicates found: {', '.join(duplicates)})")
        validation_passed = False
        
    # 5. Every team exists in the FIFA rankings dataset
    missing_rankings = []
    for team in flat_teams:
        if team not in team_ranks:
            missing_rankings.append(team)
    if not missing_rankings:
        st.success("✅ **5. Every team exists in the FIFA rankings dataset** (Passed)")
    else:
        st.error(f"❌ **5. Every team exists in the FIFA rankings dataset** (Failed: Missing teams: {', '.join(missing_rankings)})")
        validation_passed = False
        
    # 6. Every team exists in the simulator lookup tables
    missing_stats = []
    for team in flat_teams:
        if team not in preprocess.final_team_stats:
            missing_stats.append(team)
    if not missing_stats:
        st.success("✅ **6. Every team exists in the simulator lookup tables** (Passed)")
    else:
        st.error(f"❌ **6. Every team exists in the simulator lookup tables** (Failed: Missing teams: {', '.join(missing_stats)})")
        validation_passed = False
        
    if validation_passed:
        st.success("🎉 **All structural checks passed successfully! Tournament setup is fully valid.**")
