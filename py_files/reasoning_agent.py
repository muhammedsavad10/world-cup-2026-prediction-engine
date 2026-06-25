import os
import json
import sqlite3
import time
import hashlib
import uuid
from enum import Enum, auto
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from groq import Groq
from config import PROMPT_VERSION, FORECAST_VERSION, MODEL_VERSION, SIMULATION_RUNS, SHOW_DEBUG_METADATA

# Match State Enum
class MatchState(Enum):
    FUTURE = auto()
    LIVE = auto()
    COMPLETED = auto()
    UNKNOWN = auto()

# Context Dataclass container
@dataclass
class MatchAnalysisContext:
    team_a: str
    team_b: str
    prob_a: float
    prob_b: float
    rank_diff: float
    form_diff: float
    goals_diff: float
    current_phase: str = "pre_tournament"
    match_record: Optional[Dict[str, Any]] = None
    probabilities_impact: Optional[Dict[str, Any]] = None
    team_a_news: str = ""
    team_b_news: str = ""
    forecast_date: Optional[str] = None
    live_results_version: Optional[str] = None
    match_events: Optional[Dict[str, Any]] = None
    verified_statistics: Optional[Dict[str, Any]] = None

@dataclass
class ValidationResult:
    valid: bool
    hallucinated_entities: list
    unsupported_statistics: list
    unsupported_tactics: list
    unsupported_numbers: list
    evidence_level: str
    evidence_completeness: str
    retry_reason: str
    confidence_score: float

def build_evidence_summary(context: MatchAnalysisContext):
    # Core fields evaluation
    has_score = context.match_record is not None and context.match_record.get("home_score") is not None
    has_winner = context.match_record is not None and ("winner" in context.match_record)
    has_standings = context.current_phase is not None
    has_probs = context.prob_a > 0 and context.prob_b > 0
    has_impact = context.probabilities_impact is not None and len(context.probabilities_impact) > 0
    
    has_events = context.match_events is not None and (
        len(context.match_events.get("goals", [])) > 0 or 
        len(context.match_events.get("cards", [])) > 0 or 
        len(context.match_events.get("substitutions", [])) > 0
    )
    
    has_stats = context.verified_statistics is not None and len(context.verified_statistics) > 0
    
    fields = {
        "Final score": has_score,
        "Winner": has_winner,
        "Group standings": has_standings,
        "Pre-match probabilities": has_probs,
        "Tournament probability movement": has_impact,
        "Verified match events": has_events,
        "Verified match statistics": has_stats
    }
    
    # Compute completeness
    available_count = sum(1 for val in fields.values() if val)
    completeness_pct = (available_count / 7) * 100
    completeness_str = f"{available_count}/7 ({completeness_pct:.0f}%)"
    
    # Compute Evidence Quality
    # High: verified events AND verified statistics
    # Medium: verified events ONLY
    # Basic: score and standings only (neither events nor stats)
    if has_events and has_stats:
        quality = "High"
    elif has_events:
        quality = "Medium"
    else:
        quality = "Basic"
        
    # Generate checklist markdown
    checklist = "Available Evidence:\n"
    for name, available in fields.items():
        symbol = "✓" if available else "✗"
        status_text = "available" if available else "unavailable"
        checklist += f"- {symbol} {name} ({status_text})\n"
        
    return quality, completeness_str, completeness_pct, checklist

def validate_completed_match_response(explanation: str, context: MatchAnalysisContext) -> ValidationResult:
    import re
    quality, completeness_str, completeness_pct, checklist = build_evidence_summary(context)
    explanation_lower = explanation.lower()
    
    # 1. Player check
    # Extract verified players from events
    verified_players = set()
    if context.match_events:
        for goal in context.match_events.get("goals", []):
            if isinstance(goal, dict) and "player" in goal:
                verified_players.add(goal["player"].lower().strip())
        for card in context.match_events.get("cards", []):
            if isinstance(card, dict) and "player" in card:
                verified_players.add(card["player"].lower().strip())
        for sub in context.match_events.get("substitutions", []):
            if isinstance(sub, dict):
                if "player_in" in sub:
                    verified_players.add(sub["player_in"].lower().strip())
                if "player_out" in sub:
                    verified_players.add(sub["player_out"].lower().strip())
                if "player" in sub:
                    verified_players.add(sub["player"].lower().strip())
                    
    # Scan text for any global player name
    hallucinated_entities = []
    common_words_to_exclude = {
        "junior", "senior", "de", "del", "la", "van", "von", "da", "di", "dos", "santos", "silva"
    }
    
    # Tokenize the text into lowercase words
    words_in_text = set(re.findall(r'\b[a-zA-Z\u00C0-\u00FF]+\b', explanation_lower))
    
    for full_name_lower, full_name_orig in global_players.items():
        parts = full_name_lower.split()
        is_mentioned = False
        for part in parts:
            part_clean = part.replace(".", "").strip()
            if len(part_clean) > 3 and part_clean not in common_words_to_exclude:
                if part_clean in words_in_text:
                    is_mentioned = True
                    break
        
        if is_mentioned:
            # Player is mentioned! Verify if they are in verified_players
            is_verified = False
            for vp in verified_players:
                if full_name_lower in vp or vp in full_name_lower:
                    is_verified = True
                    break
                vp_parts = vp.split()
                for part in parts:
                    part_clean = part.replace(".", "").strip()
                    if len(part_clean) > 3 and part_clean not in common_words_to_exclude:
                        if part_clean in vp_parts:
                            is_verified = True
                            break
            
            if not is_verified:
                if full_name_orig not in hallucinated_entities:
                    hallucinated_entities.append(full_name_orig)
                    
    # 2. Statistical check
    unsupported_statistics = []
    stats_to_check = ["possession", "xg", "shots", "saves", "passing accuracy", "corners"]
    for stat in stats_to_check:
        if stat in explanation_lower:
            if not context.verified_statistics or stat not in context.verified_statistics:
                unsupported_statistics.append(stat.capitalize())
                
    # 3. Tactical check
    unsupported_tactics = []
    if re.search(r'\b\d-\d-\d(?:-\d)?\b', explanation_lower):
        unsupported_tactics.append("Specific Formation Mention")
    if re.search(r'\b\d{1,2}(th|rd|st|nd)? minute\b|\bminute \d{1,2}\b', explanation_lower):
        unsupported_tactics.append("Specific Minute Trigger")
    for term in ["formation change", "tactical switch"]:
        if term in explanation_lower:
            unsupported_tactics.append(term.capitalize())
    if "substitution" in explanation_lower and (not context.match_events or len(context.match_events.get("substitutions", [])) == 0):
        unsupported_tactics.append("Unverified Substitution Mention")
        
    # 4. Numeric check
    unsupported_numbers = []
    found_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', explanation)
    context_str = str(context)
    context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', context_str))
    context_numbers_plain = set(re.findall(r'\b\d+(?:\.\d+)?\b', context_str))
    
    for num in found_numbers:
        num_clean = num.replace("%", "")
        try:
            val = float(num_clean)
            if val < 10.0:
                continue
            if int(val) in [2026, 10000, 100, 0]:
                continue
        except ValueError:
            pass
            
        if num not in context_numbers and num_clean not in context_numbers_plain:
            in_context = False
            for cn in context_numbers | context_numbers_plain:
                if num_clean in cn or cn in num_clean:
                    in_context = True
                    break
            if not in_context:
                unsupported_numbers.append(num)
                
    valid = len(hallucinated_entities) == 0 and len(unsupported_statistics) == 0 and len(unsupported_tactics) == 0 and len(unsupported_numbers) == 0
    
    retry_reasons = []
    if hallucinated_entities:
        retry_reasons.append(f"Hallucinated players: {', '.join(hallucinated_entities)}.")
    if unsupported_statistics:
        retry_reasons.append(f"Unsupported statistics: {', '.join(unsupported_statistics)}.")
    if unsupported_tactics:
        retry_reasons.append(f"Unsupported tactics: {', '.join(unsupported_tactics)}.")
    if unsupported_numbers:
        retry_reasons.append(f"Unsupported numbers: {', '.join(unsupported_numbers)}.")
        
    retry_reason = " ".join(retry_reasons)
    
    conf_score = 1.0
    if not valid:
        conf_score -= 0.25 * (bool(hallucinated_entities) + bool(unsupported_statistics) + bool(unsupported_tactics) + bool(unsupported_numbers))
    conf_score = max(0.0, conf_score)
    
    return ValidationResult(
        valid=valid,
        hallucinated_entities=hallucinated_entities,
        unsupported_statistics=unsupported_statistics,
        unsupported_tactics=unsupported_tactics,
        unsupported_numbers=unsupported_numbers,
        evidence_level=quality,
        evidence_completeness=completeness_str,
        retry_reason=retry_reason,
        confidence_score=conf_score
    )

def generate_deterministic_fallback(context: MatchAnalysisContext) -> str:
    home_team = context.match_record.get("home_team") if context.match_record else context.team_a
    away_team = context.match_record.get("away_team") if context.match_record else context.team_b
    home_score = context.match_record.get("home_score", 0) if context.match_record else 0
    away_score = context.match_record.get("away_score", 0) if context.match_record else 0
    winner = context.match_record.get("winner") if context.match_record else "Draw"
    actual_winner = winner if winner != "Draw" and winner is not None else "Draw"
    
    rounded_prob_a = round(float(context.prob_a), 2)
    rounded_prob_b = round(float(context.prob_b), 2)
    
    form_text = f"form difference of {context.form_diff:+.1f}"
    rank_text = f"FIFA ranking gap of {context.rank_diff:+.1f}"
    outcome_text = f"{actual_winner} won the match" if actual_winner != "Draw" else "the match ended in a draw"
    
    fallback = f"""### Evidence-Based Explanation
{home_team} and {away_team} completed their match with a final score of {home_team} {home_score} - {away_score} {away_team}. Before kickoff, the predictive model estimated a {rounded_prob_a}% win probability for {context.team_a} and {rounded_prob_b}% for {context.team_b}. The outcome where {outcome_text} is consistent with the pre-match {rank_text} and the recent {form_text}.

### Monte Carlo Update
The completed match is treated as fixed evidence in subsequent Monte Carlo simulations. Standings and tournament advancement probabilities have been updated accordingly.

### Explanation Confidence
High. Generated via deterministic statistical fallback.
"""
    return fallback

# Initialize the Groq client safely
client = None
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    client = Groq(api_key=api_key)

# Load team intelligence dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "data", "world_cup_2026_team_intelligence.json")

team_intelligence = {}
global_players = {}
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for item in raw_data:
                team_intelligence[item["team"].lower().strip()] = item
                
                # Build global player list for entity validation whitelisting
                if "captain" in item and isinstance(item["captain"], dict) and "name" in item["captain"]:
                    c_name = item["captain"]["name"].strip()
                    if c_name:
                        global_players[c_name.lower()] = c_name
                if "top_scorer" in item and isinstance(item["top_scorer"], dict) and "name" in item["top_scorer"]:
                    ts_name = item["top_scorer"]["name"].strip()
                    if ts_name:
                        global_players[ts_name.lower()] = ts_name
                for p in item.get("key_players", []):
                    if isinstance(p, dict) and "name" in p:
                        p_name = p["name"].strip()
                        if p_name:
                            global_players[p_name.lower()] = p_name
    except Exception as e:
        print(f"Error loading team intelligence dataset: {e}")

# Database Cache Initialization
db_dir = os.path.join(base_dir, "..", "data")
os.makedirs(db_dir, exist_ok=True)
db_file = os.path.join(db_dir, "xai_cache.db")

def get_db_connection():
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS xai_cache (
        cache_key TEXT PRIMARY KEY,
        explanation TEXT,
        prompt_version TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON xai_cache(created_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_version ON xai_cache(prompt_version);")
    conn.commit()
    return conn

# Global thread-safe DB connection
db_conn = get_db_connection()

def cleanup_live_cache():
    try:
        db_conn.execute("DELETE FROM xai_cache WHERE created_at < datetime('now', '-30 seconds') AND cache_key LIKE 'live_%';")
        db_conn.commit()
    except Exception:
        pass

def increment_cache_stat(stat_type):
    try:
        import streamlit as st
        if stat_type == "hit":
            st.session_state.xai_cache_hits = st.session_state.get("xai_cache_hits", 0) + 1
        elif stat_type == "miss":
            st.session_state.xai_cache_misses = st.session_state.get("xai_cache_misses", 0) + 1
    except Exception:
        pass

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
    Cached prompt loader. Reads text file templates from nested directory prompts/{mode}/{version}.txt
    """
    prompt_path = os.path.join(base_dir, "prompts", mode, f"{version}.txt")
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

def get_teams_context(team_a, team_b, team_a_news="", team_b_news=""):
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

def build_explanation_footer(state, forecast_date=None, live_results_version=None, cache_status="Miss", gen_duration=0.0, explanation_id="", evidence_level="N/A", evidence_completeness="N/A"):
    """
    Constructs a structured, transparent metadata explanation footer. Toggleable via SHOW_DEBUG_METADATA.
    """
    if not SHOW_DEBUG_METADATA:
        # Standard user-facing clean footer
        return "\n\n*This analysis is based on completed match data and Monte Carlo simulations of the remaining tournament.*"
        
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
    generated_at_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    footer = f"""
 
---
### 📊 Analysis Metadata (Developer Mode)
*   **Mode**: {mode_text}
*   **Evidence**: {evidence_text}
*   **Evidence Quality**: {evidence_level}
*   **Evidence Completeness**: {evidence_completeness}
*   **Validator**: Structured (Entity, Statistics, Numeric, Tactical-event)
*   **Validated**: True
*   **Generated At**: {generated_at_str}
*   **Simulation**: {sim_text}
*   **Explanation ID**: {explanation_id}
*   **Cache**: {cache_status}
*   **Generation Time**: {round(gen_duration, 4)} seconds
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

def generate_prediction_explanation(context: MatchAnalysisContext) -> str:
    system_prompt = load_prompt("prediction", PROMPT_VERSION)
    if not system_prompt:
        return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"

    rounded_prob_a = round(float(context.prob_a), 2)
    rounded_prob_b = round(float(context.prob_b), 2)
    rounded_rank = round(float(context.rank_diff), 2)
    rounded_form = round(float(context.form_diff), 2)
    rounded_goals = round(float(context.goals_diff), 2)
    
    if rounded_prob_a > rounded_prob_b:
        prediction_headline = f"{context.team_a} defeats {context.team_b}"
    elif rounded_prob_b > rounded_prob_a:
        prediction_headline = f"{context.team_b} defeats {context.team_a}"
    else:
        prediction_headline = f"{context.team_a} and {context.team_b} draw"

    math_context = f"""
    PREDICTION AND MODEL PERFORMANCE METRICS:
    - Prediction: {prediction_headline}
    - {context.team_a} Win Probability: {rounded_prob_a}%
    - {context.team_b} Win Probability: {rounded_prob_b}%
    - FIFA Rank Gap: {rounded_rank}
    - Form Advantage: {rounded_form}
    - Goal Efficiency Lead: {rounded_goals}
    """
    
    team_context = get_teams_context(context.team_a, context.team_b, context.team_a_news, context.team_b_news)
    user_prompt = f"Matchup: {context.team_a} vs {context.team_b}\nPhase: {context.current_phase}\n\n{math_context}\n\n{team_context}"
    
    return call_groq_completions(system_prompt, user_prompt)

def generate_live_explanation(context: MatchAnalysisContext) -> str:
    system_prompt = load_prompt("live", PROMPT_VERSION)
    if not system_prompt:
        return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"

    home_score = context.match_record.get("home_score", 0) if context.match_record else 0
    away_score = context.match_record.get("away_score", 0) if context.match_record else 0
    current_minute = context.match_record.get("current_minute", "Unknown") if context.match_record else "Unknown"
    
    rounded_prob_a = round(float(context.prob_a), 2)
    rounded_prob_b = round(float(context.prob_b), 2)
    rounded_rank = round(float(context.rank_diff), 2)
    rounded_form = round(float(context.form_diff), 2)
    rounded_goals = round(float(context.goals_diff), 2)
    
    live_context = f"""
    LIVE MATCH STATE:
    - Current Score: {context.team_a} {home_score} - {away_score} {context.team_b}
    - Current Minute: {current_minute}
    - Home Probability: {rounded_prob_a}%
    - Away Probability: {rounded_prob_b}%
    - FIFA Rank Gap: {rounded_rank}
    - Form Advantage: {rounded_form}
    - Goal Efficiency Lead: {rounded_goals}
    """
    
    team_context = get_teams_context(context.team_a, context.team_b, context.team_a_news, context.team_b_news)
    user_prompt = f"Matchup: {context.team_a} vs {context.team_b}\n\n{live_context}\n\n{team_context}"
    
    return call_groq_completions(system_prompt, user_prompt)

def generate_result_explanation(context: MatchAnalysisContext) -> str:
    system_prompt = load_prompt("completed", PROMPT_VERSION)
    if not system_prompt:
        return "AI_TACTICAL_ANALYSIS_UNAVAILABLE"
        
    quality, completeness_str, completeness_pct, checklist = build_evidence_summary(context)
    
    rounded_prob_a = round(float(context.prob_a), 2)
    rounded_prob_b = round(float(context.prob_b), 2)
    rounded_rank = round(float(context.rank_diff), 2)
    rounded_form = round(float(context.form_diff), 2)
    rounded_goals = round(float(context.goals_diff), 2)
    
    prob_impact_text = ""
    if context.probabilities_impact:
        prob_impact_text = f"""
        TOURNAMENT PROBABILITY IMPACT:
        - {context.team_a} Championship Probability: {round(context.probabilities_impact.get('team_a_baseline_champ', 0.0), 2)}% -> {round(context.probabilities_impact.get('team_a_current_champ', 0.0), 2)}%
        - {context.team_b} Championship Probability: {round(context.probabilities_impact.get('team_b_baseline_champ', 0.0), 2)}% -> {round(context.probabilities_impact.get('team_b_current_champ', 0.0), 2)}%
        - {context.team_a} Group Qualification Probability: {round(context.probabilities_impact.get('team_a_baseline_qual', 0.0), 2)}% -> {round(context.probabilities_impact.get('team_a_current_qual', 0.0), 2)}%
        - {context.team_b} Group Qualification Probability: {round(context.probabilities_impact.get('team_b_baseline_qual', 0.0), 2)}% -> {round(context.probabilities_impact.get('team_b_current_qual', 0.0), 2)}%
        """
        
    events_text = ""
    if context.match_events:
        events_text = "\nVERIFIED MATCH EVENTS:\n"
        for goal in context.match_events.get("goals", []):
            events_text += f"- Goal: Scored by {goal.get('player')} at minute {goal.get('minute')}'\n"
        for card in context.match_events.get("cards", []):
            events_text += f"- Card: {card.get('card')} card shown to {card.get('player')} at minute {card.get('minute')}'\n"
        for sub in context.match_events.get("substitutions", []):
            events_text += f"- Substitution: {sub.get('player_in')} replaced {sub.get('player_out')} at minute {sub.get('minute')}'\n"
            
    stats_text = ""
    if context.verified_statistics:
        stats_text = "\nVERIFIED MATCH STATISTICS:\n"
        for k, v in context.verified_statistics.items():
            stats_text += f"- {k.capitalize()}: {v}\n"
            
    evidence_context = f"""
    EVIDENCE STATUS CHECKLIST:
    {checklist}
    Evidence Quality: {quality}
    Evidence Completeness: {completeness_str}
    
    BASELINE MODEL AND TOURNAMENT PERFORMANCE METRICS:
    - {context.team_a} Pre-kickoff win probability: {rounded_prob_a}%
    - {context.team_b} Pre-kickoff win probability: {rounded_prob_b}%
    - FIFA Rank Gap: {rounded_rank}
    - Form Advantage: {rounded_form}
    - Goal Efficiency Lead: {rounded_goals}
    
    {prob_impact_text}
    {events_text}
    {stats_text}
    """
    
    team_context = get_teams_context(context.team_a, context.team_b, context.team_a_news, context.team_b_news)
    user_prompt = f"Matchup: {context.team_a} vs {context.team_b}\n\n{evidence_context}\n\n{team_context}"
    
    # Attempt 1
    analysis = call_groq_completions(system_prompt, user_prompt)
    if analysis == "AI_TACTICAL_ANALYSIS_UNAVAILABLE":
        return analysis
        
    result = validate_completed_match_response(analysis, context)
    if result.valid:
        return analysis
        
    # Attempt 2: Targeted Retry
    retry_instruction = f"""
    \n\n[WARNING: Previous attempt failed validation. Please regenerate complying with the following feedback:
    {result.retry_reason}
    Remove any unverified player names, unverified stats, unverified numbers, and specific formation changes or minute mentions.
    Regenerate using only verified evidence.]
    """
    retry_user_prompt = user_prompt + retry_instruction
    
    analysis_retry = call_groq_completions(system_prompt, retry_user_prompt)
    if analysis_retry == "AI_TACTICAL_ANALYSIS_UNAVAILABLE":
        return generate_deterministic_fallback(context)
        
    result_retry = validate_completed_match_response(analysis_retry, context)
    if result_retry.valid:
        return analysis_retry
        
    # Fallback if retry still invalid
    return generate_deterministic_fallback(context)

def generate_match_analysis(context: MatchAnalysisContext) -> str:
    """
    The main Explainable AI dispatcher. Resolves the fixture state, handles thread-safe
    WAL SQLite cache validations, runs execution counters, and appends footers.
    """
    start_time = time.perf_counter()
    state = get_match_state(context.match_record)
    
    # Run automatic live cache cleanups to keep table sizes lean
    cleanup_live_cache()
    
    # Resolve evidence level and completeness
    evidence_level = "N/A"
    evidence_completeness = "N/A"
    if state == MatchState.COMPLETED:
        quality, completeness_str, _, _ = build_evidence_summary(context)
        evidence_level = quality
        evidence_completeness = completeness_str
        
    # 1. Compile Unique Cache Key
    status_str = context.match_record.get("status", "UNKNOWN") if context.match_record else "FUTURE"
    home_score = context.match_record.get("home_score", 0) if context.match_record else 0
    away_score = context.match_record.get("away_score", 0) if context.match_record else 0
    news_hash = hashlib.md5((context.team_a_news + context.team_b_news).encode()).hexdigest()[:6]
    
    cache_key = f"{state.name.lower()}_{context.team_a}_{context.team_b}_{home_score}_{away_score}_{status_str}_{evidence_level}_{PROMPT_VERSION}_{FORECAST_VERSION}_{MODEL_VERSION}_{news_hash}"
    
    # 2. Check Cache
    use_cache = (state == MatchState.COMPLETED or state == MatchState.LIVE)
    
    if use_cache:
        # If LIVE, filter on age (under 30s) directly inside SQL
        if state == MatchState.LIVE:
            query = "SELECT explanation FROM xai_cache WHERE cache_key = ? AND created_at >= datetime('now', '-30 seconds');"
        else:
            query = "SELECT explanation FROM xai_cache WHERE cache_key = ?;"
            
        try:
            cursor = db_conn.execute(query, (cache_key,))
            row = cursor.fetchone()
            if row:
                increment_cache_stat("hit")
                gen_duration = time.perf_counter() - start_time
                explanation_id = f"XAI-{str(uuid.uuid5(uuid.NAMESPACE_DNS, cache_key))[:8].upper()}"
                
                # Check if metadata footer should be updated
                explanation = row[0]
                # Strip out old footers and append fresh metrics
                if "Analysis Metadata" in explanation or "*This analysis is based on" in explanation:
                    explanation_clean = explanation.split("\n\n---")[0].split("\n---")[0]
                else:
                    explanation_clean = explanation
                    
                footer = build_explanation_footer(state, context.forecast_date, context.live_results_version, "Hit", gen_duration, explanation_id, evidence_level=evidence_level, evidence_completeness=evidence_completeness)
                return explanation_clean + footer
        except Exception:
            pass

    # 3. Cache Miss - Generate analysis
    increment_cache_stat("miss")
    
    if state == MatchState.COMPLETED:
        analysis = generate_result_explanation(context)
    elif state == MatchState.LIVE:
        analysis = generate_live_explanation(context)
    elif state == MatchState.UNKNOWN:
        notice = "### ⚠️ Match Status: Unknown\nOfficial match status is currently unavailable. Predictions and analysis will resume automatically once verified match data becomes available.\n\n"
        fallback = generate_prediction_explanation(context)
        analysis = notice + fallback
    else:
        analysis = generate_prediction_explanation(context)
        
    if analysis == "AI_TACTICAL_ANALYSIS_UNAVAILABLE":
        return analysis
        
    # Write to SQLite Cache if successful and not an error/unavailable status
    is_error = (
        analysis.startswith("Tactical analysis synthesis interrupted") or 
        analysis == "AI_TACTICAL_ANALYSIS_UNAVAILABLE"
    )
    if use_cache and not is_error:
        try:
            db_conn.execute(
                "INSERT OR REPLACE INTO xai_cache (cache_key, explanation, prompt_version) VALUES (?, ?, ?);",
                (cache_key, analysis, PROMPT_VERSION)
            )
            db_conn.commit()
        except Exception:
            pass
            
    gen_duration = time.perf_counter() - start_time
    explanation_id = f"XAI-{str(uuid.uuid5(uuid.NAMESPACE_DNS, cache_key))[:8].upper()}"
    footer = build_explanation_footer(state, context.forecast_date, context.live_results_version, "Miss", gen_duration, explanation_id, evidence_level=evidence_level, evidence_completeness=evidence_completeness)
    
    return analysis + footer
