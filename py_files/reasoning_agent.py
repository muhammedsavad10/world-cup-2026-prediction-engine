import os
import json
from enum import Enum, auto
from functools import lru_cache
from groq import Groq
from config import PROMPT_VERSION, FORECAST_VERSION, MODEL_VERSION, SIMULATION_RUNS

# Match State Enum definition
class MatchState(Enum):
    FUTURE = auto()
    LIVE = auto()
    COMPLETED = auto()
    UNKNOWN = auto()

# Initialize the Groq client safely
client = None
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    client = Groq(api_key=api_key)

# Load team intelligence dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "data", "world_cup_2026_team_intelligence.json")

team_intelligence = {}
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for item in raw_data:
                team_intelligence[item["team"].lower().strip()] = item
    except Exception as e:
        print(f"Error loading team intelligence dataset: {e}")

def get_team_info(team_name):
    name_lookup = team_name.lower().strip()
    if name_lookup == "usa":
        name_lookup = "united states"
    return team_intelligence.get(name_lookup)

def format_team_context(info, team_name):
    if not info:
        return f"No local team intelligence data available for {team_name}."
    
    coach = info.get("coach", {})
    captain = info.get("captain", {})
    top_scorer = info.get("top_scorer", {})
    
    context = f"TEAM: {info.get('team')}\n"
    context += f"- Head Coach/Manager: {coach.get('name')} (Years in charge: {coach.get('years_in_charge')}, Preferred formation: {coach.get('preferred_formation')}, Tactical style: {coach.get('tactical_style')})\n"
    context += f"- Captain: {captain.get('name')} (Position: {captain.get('position')}, Club: {captain.get('club')})\n"
    context += f"- Top Scorer: {top_scorer.get('name')} (Goals: {top_scorer.get('goals')})\n"
    context += "- Key Players Details:\n"
    for player in info.get("key_players", []):
        context += f"  * {player.get('name')} ({player.get('position')}, Club: {player.get('club')}, Age: {player.get('age')}, Caps/Goals: {player.get('caps')}/{player.get('goals')}, Value: {player.get('market_value')}, Injury: {player.get('injury_status')}, Season Stats: {player.get('current_season_stats')}, Form: {player.get('recent_form_summary')})\n"
    return context

@lru_cache(maxsize=None)
def load_prompt(mode, version):
    """
    Cached prompt loader. Reads text file templates from the prompts/ directory.
    """
    prompt_path = os.path.join(base_dir, "prompts", f"{mode}_{version}.txt")
    if not os.path.exists(prompt_path):
        return ""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def get_match_state(match_record):
    """
    Centralized match-state resolver mapping short status codes from API-Football.
    """
    if match_record is None:
        return MatchState.FUTURE
        
    status = match_record.get("status", "UNKNOWN")
    if status in ["FT", "AET", "PEN", "AWD", "WO"]:
        return MatchState.COMPLETED
    if status in ["1H", "HT", "2H", "ET"]:
        return MatchState.LIVE
    if status in ["NS", "TBD"]:
        return MatchState.FUTURE
    if status in ["PST", "CANC", "ABD", "INT", "SUSP"]:
        return MatchState.UNKNOWN
        
    return MatchState.UNKNOWN

def get_teams_context(team_a, team_b, team_a_news=None, team_b_news=None):
    """
    Builds context string for both teams, incorporating structured dataset details and optional external news.
    """
    info_a = get_team_info(team_a)
    info_b = get_team_info(team_b)
    
    team_a_context = format_team_context(info_a, team_a)
    team_b_context = format_team_context(info_b, team_b)
    
    dataset_context = f"""
    AUTHORITATIVE DATASET INFO:
    Team A ({team_a}) Context:
    {team_a_context}
    
    Team B ({team_b}) Context:
    {team_b_context}
    """
    
    news_context = ""
    if team_a_news or team_b_news:
        news_context = "\nEXTERNAL LIVE NEWS REPORT:"
        if team_a_news:
            news_context += f"\n- Team A ({team_a}) News: {team_a_news}"
        if team_b_news:
            news_context += f"\n- Team B ({team_b}) News: {team_b_news}"
            
    return dataset_context + news_context

def build_explanation_footer(state, forecast_date=None, live_results_version=None):
    """
    Constructs a structured, transparent metadata explanation footer.
    """
    if state == MatchState.COMPLETED:
        mode_text = "Completed Match Review"
        evidence_text = "Official Match Result"
        sim_text = "Completed matches fixed. Remaining tournament simulated using 10,000 Monte Carlo runs."
    elif state == MatchState.LIVE:
        mode_text = "Live Match Commentary"
        evidence_text = "Live Match State"
        sim_text = "Current result fixed. Remaining tournament simulated using 10,000 Monte Carlo runs."
    elif state == MatchState.UNKNOWN:
        mode_text = "Unknown Status Fallback"
        evidence_text = "Incomplete/Suspended Match Info"
        sim_text = "Fallback to pre-match forecast."
    else:
        mode_text = "Forecast"
        evidence_text = "Pre-match statistical estimates"
        sim_text = "Entire tournament simulated using 10,000 Monte Carlo runs."
        
    date_str = forecast_date if forecast_date else "2026-06-25"
    version_str = live_results_version if live_results_version else "Matchday 1"
    
    footer = f"""

---
### 📊 Analysis Metadata
*   **Mode**: {mode_text}
*   **Evidence**: {evidence_text}
*   **Simulation**: {sim_text}
*   **Forecast Version**: {FORECAST_VERSION}
*   **Prompt Version**: {PROMPT_VERSION}
*   **Model Version**: {MODEL_VERSION}
*   **Simulation Runs**: {SIMULATION_RUNS}
*   **Forecast Date**: {date_str}
*   **Live Results Version**: {version_str}
"""
    return footer

def call_groq_completions(system_prompt, user_prompt):
    global client
    if not client:
        # Try initializing client again in case the key was configured late
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            client = Groq(api_key=api_key)
        else:
            return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=800,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        err_msg = str(e).lower()
        if "api_key" in err_msg or "api key" in err_msg or "authentication" in err_msg or "unauthorized" in err_msg or "401" in err_msg:
            return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"
        return f"Tactical analysis synthesis interrupted. Error: {str(e)}"

def generate_prediction_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, current_phase, team_a_news=None, team_b_news=None):
    system_prompt = load_prompt("prediction", PROMPT_VERSION)
    if not system_prompt:
        return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"

    rounded_prob_a = round(float(prob_a), 2)
    rounded_prob_b = round(float(prob_b), 2)
    rounded_rank = round(float(rank_diff), 2)
    rounded_form = round(float(form_diff), 2)
    rounded_goals = round(float(goals_diff), 2)
    
    if rounded_prob_a > rounded_prob_b:
        prediction_headline = f"{team_a} defeats {team_b}"
    elif rounded_prob_b > rounded_prob_a:
        prediction_headline = f"{team_b} defeats {team_a}"
    else:
        prediction_headline = f"{team_a} and {team_b} draw"

    math_context = f"""
    PREDICTION AND MODEL PERFORMANCE METRICS:
    - Prediction: {prediction_headline}
    - {team_a} Win Probability: {rounded_prob_a}%
    - {team_b} Win Probability: {rounded_prob_b}%
    - FIFA Rank Gap: {rounded_rank}
    - Form Advantage: {rounded_form}
    - Goal Efficiency Lead: {rounded_goals}
    """
    
    team_context = get_teams_context(team_a, team_b, team_a_news, team_b_news)
    user_prompt = f"Matchup: {team_a} vs {team_b}\nPhase: {current_phase}\n\n{math_context}\n\n{team_context}"
    
    return call_groq_completions(system_prompt, user_prompt)

def generate_live_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, match_record, probabilities_impact, team_a_news=None, team_b_news=None):
    system_prompt = load_prompt("live", PROMPT_VERSION)
    if not system_prompt:
        return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"

    home_score = match_record.get("home_score", 0)
    away_score = match_record.get("away_score", 0)
    current_minute = match_record.get("current_minute", "Unknown")
    
    rounded_prob_a = round(float(prob_a), 2)
    rounded_prob_b = round(float(prob_b), 2)
    rounded_rank = round(float(rank_diff), 2)
    rounded_form = round(float(form_diff), 2)
    rounded_goals = round(float(goals_diff), 2)
    
    live_context = f"""
    LIVE MATCH STATE:
    - Current Score: {team_a} {home_score} – {away_score} {team_b}
    - Current Minute: {current_minute}
    - Home Probability: {rounded_prob_a}%
    - Away Probability: {rounded_prob_b}%
    - FIFA Rank Gap: {rounded_rank}
    - Form Advantage: {rounded_form}
    - Goal Efficiency Lead: {rounded_goals}
    """
    
    team_context = get_teams_context(team_a, team_b, team_a_news, team_b_news)
    user_prompt = f"Matchup: {team_a} vs {team_b}\n\n{live_context}\n\n{team_context}"
    
    return call_groq_completions(system_prompt, user_prompt)

def generate_result_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, match_record, probabilities_impact, team_a_news=None, team_b_news=None):
    system_prompt = load_prompt("completed", PROMPT_VERSION)
    if not system_prompt:
        return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"

    home_team = match_record.get("home_team")
    away_team = match_record.get("away_team")
    home_score = match_record.get("home_score", 0)
    away_score = match_record.get("away_score", 0)
    winner = match_record.get("winner")
    
    actual_winner = winner if winner != "Draw" and winner is not None else "Draw"
    
    rounded_prob_a = round(float(prob_a), 2)
    rounded_prob_b = round(float(prob_b), 2)
    rounded_rank = round(float(rank_diff), 2)
    rounded_form = round(float(form_diff), 2)
    rounded_goals = round(float(goals_diff), 2)
    
    # Determine pre-match predicted winner
    if rounded_prob_a > rounded_prob_b:
        pred_winner = team_a
    elif rounded_prob_b > rounded_prob_a:
        pred_winner = team_b
    else:
        pred_winner = "Draw"
        
    is_correct = "Correct" if pred_winner == actual_winner else "Incorrect"
    
    prob_impact_text = ""
    if probabilities_impact:
        prob_impact_text = f"""
        TOURNAMENT PROBABILITY IMPACT (from pre-tournament/baseline estimate to current live status):
        - {team_a} Championship Probability: {probabilities_impact.get('team_a_baseline_champ', 0.0):.2f}% -> {probabilities_impact.get('team_a_current_champ', 0.0):.2f}%
        - {team_b} Championship Probability: {probabilities_impact.get('team_b_baseline_champ', 0.0):.2f}% -> {probabilities_impact.get('team_b_current_champ', 0.0):.2f}%
        - {team_a} Group Qualification Probability: {probabilities_impact.get('team_a_baseline_qual', 0.0):.2f}% -> {probabilities_impact.get('team_a_current_qual', 0.0):.2f}%
        - {team_b} Group Qualification Probability: {probabilities_impact.get('team_b_baseline_qual', 0.0):.2f}% -> {probabilities_impact.get('team_b_current_qual', 0.0):.2f}%
        """
        
    result_context = f"""
    COMPLETED MATCH RESULT REVIEW:
    - Final Score: {home_team} {home_score} – {away_score} {away_team}
    - Winner/Outcome: {actual_winner}
    - Pre-kickoff Win Probability {team_a}: {rounded_prob_a}%
    - Pre-kickoff Win Probability {team_b}: {rounded_prob_b}%
    - Model Forecast Accuracy: {is_correct}
    
    {prob_impact_text}
    
    FIFA Rank Gap: {rounded_rank}
    Form Advantage: {rounded_form}
    Goal Efficiency Lead: {rounded_goals}
    """
    
    team_context = get_teams_context(team_a, team_b, team_a_news, team_b_news)
    user_prompt = f"Matchup: {team_a} vs {team_b}\n\n{result_context}\n\n{team_context}"
    
    return call_groq_completions(system_prompt, user_prompt)

def generate_match_analysis(
    team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff,
    current_phase="pre_tournament", match_record=None, probabilities_impact=None,
    team_a_news=None, team_b_news=None, forecast_date=None, live_results_version=None
):
    """
    The main Explainable AI dispatcher. Resolves the fixture state and routes execution to
    the correct prompt constructor, appending the transparent metadata footer.
    """
    state = get_match_state(match_record)
    
    if state == MatchState.COMPLETED:
        analysis = generate_result_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, match_record, probabilities_impact, team_a_news, team_b_news)
    elif state == MatchState.LIVE:
        analysis = generate_live_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, match_record, probabilities_impact, team_a_news, team_b_news)
    elif state == MatchState.UNKNOWN:
        # Gracefully handle UNKNOWN state: return a notice or fall back to predictions
        notice = "### ⚠️ Match Status Unavailable\nThis match status is currently marked as Postponed, Suspended, or Cancelled. Real-time statistical analysis is currently deferred. Showing pre-match prediction forecast below.\n\n"
        fallback = generate_prediction_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, current_phase, team_a_news, team_b_news)
        analysis = notice + fallback
    else:
        analysis = generate_prediction_explanation(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, current_phase, team_a_news, team_b_news)
        
    if analysis == "AI_TACTICAL_ANALYSIS_UNAVAILABLE":
        return analysis
        
    # Append transparent explanation footer
    footer = build_explanation_footer(state, forecast_date, live_results_version)
    return analysis + footer
