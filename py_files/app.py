import streamlit as st
import datetime
import textwrap
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from const import WCGroups
from preprocess import load_data, get_match_features
import preprocess
from logistic_regression_class import LogisticRegressionClass
from tournament_simulator import TournamentSimulator
from reasoning_agent import generate_match_analysis

def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.split("\n"))

# Set wide mode config first
st.set_page_config(layout="wide", page_title="2026 FIFA World Cup AI Predictor")

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

# Cache data loading and pipeline training
@st.cache_resource
def load_model_and_data():
    X, y, rankings_df = load_data()
    model_class = LogisticRegressionClass()
    base_model = model_class.get_model()
    
    # Scale features using StandardScaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_model)
    ])
    pipeline.fit(X, y)
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

def feature_generator(team1, team2):
    return get_match_features(team1, team2, rankings_df)

# Initialize simulator
simulator = TournamentSimulator(datetime.date(2026, 6, 11), pipeline, GROUPS, feature_generator)

# Define the tabs
tab1, tab2 = st.tabs(["Tournament Simulation Pipeline", "Head-to-Head Bracket Inspector"])

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
    
    with st.spinner("Reasoning Agent analyzing team dynamics..."):
        analysis = generate_match_analysis(
            t1_name, t2_name, 
            t1_prob * 100, t2_prob * 100, 
            rank_diff, form_diff, goals_diff, 
            current_phase="pre_tournament"
        )
    
    st.info(analysis)

# Process active URL dialogue callbacks early
if "selected_match" in st.query_params:
    match_label = st.query_params["selected_match"]
    st.query_params.clear()
    t1_n, t1_p, t2_n, t2_p = parse_match_label(match_label)
    show_match_dialog(t1_n, t1_p, t2_n, t2_p)

# Generate Group Stage round-robin pairings mathematically (72 games total)
group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
group_fixtures = []
for g_idx, group in enumerate(GROUPS):
    g_letter = group_letters[g_idx]
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            team_a = group[i]
            team_b = group[j]
            # Compute probabilities using the trained model
            feat = get_match_features(team_a, team_b, rankings_df)
            probs = pipeline.predict_proba([feat])[0]
            group_fixtures.append({
                'stage': f"Group Stage: [Group {g_letter}]",
                'team_a': team_a,
                'team_b': team_b,
                'prob_a': probs[1],
                'prob_b': probs[0],
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
            round_results, labels, odds = simulator.playKnockOuts()
            
            # Save labels to session state for Tab 2
            st.session_state['simulated_labels'] = labels
            st.session_state['round_results'] = round_results
            st.session_state['show_balloons'] = True

    if 'simulated_labels' in st.session_state:
        labels = st.session_state['simulated_labels']
        round_results = st.session_state['round_results']
        
        # Divide matches by round
        r32 = labels[0:16]
        r16 = labels[16:24]
        qf = labels[24:28]
        sf = labels[28:30]
        final = labels[30:31]
        third_place = labels[31:32]
        
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
        
    with st.spinner("Agentic AI is analyzing..."):
        analysis = generate_match_analysis(
            t1_name, t2_name, 
            exact_t1_prob * 100, exact_t2_prob * 100, 
            rank_diff, form_diff, goals_diff, 
            current_phase=phase
        )
    
    st.info(analysis)
