from dataclasses import dataclass
from typing import Optional

@dataclass
class CalibrationMetrics:
    confidence: str
    upset_level: str
    is_correct: str
    pred_winner: str
    actual_winner: str
    t1_champ_change: str
    t2_champ_change: str
    t1_qual_change: str
    t2_qual_change: str
    t1_base_champ: float
    t1_live_champ: float
    t2_base_champ: float
    t2_live_champ: float
    t1_base_qual: float
    t1_live_qual: float
    t2_base_qual: float
    t2_live_qual: float

def get_calibration_metrics(t1_name, t2_name, t1_prob, t2_prob, match_record, baseline_probs, live_probs) -> CalibrationMetrics:
    h_score = match_record.get("home_score")
    a_score = match_record.get("away_score")
    winner = match_record.get("winner")
    
    # Margin-based confidence
    margin = abs(t1_prob - t2_prob)
    if margin >= 0.20:
        confidence = "High"
    elif margin >= 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"
        
    actual_winner = winner if winner != "Draw" and winner is not None else "Draw"
    
    # Prediction correctness
    if t1_prob > t2_prob:
        pred_winner = t1_name
    elif t2_prob > t1_prob:
        pred_winner = t2_name
    else:
        pred_winner = "Draw"
        
    is_correct = "Correct" if pred_winner == actual_winner else "Incorrect"
    
    # Upset level
    fav_prob = max(t1_prob, t2_prob)
    fav_team = t1_name if t1_prob > t2_prob else t2_name
    
    if actual_winner == "Draw":
        if fav_prob >= 0.70:
            upset_level = "Moderate Upset"
        else:
            upset_level = "Expected"
    elif actual_winner == fav_team:
        upset_level = "Expected"
    else:
        # Underdog won
        if fav_prob >= 0.70:
            upset_level = "Major Upset"
        elif fav_prob >= 0.55:
            upset_level = "Moderate Upset"
        else:
            upset_level = "Expected"
            
    # Probability Change (Before vs After)
    t1_base_champ = baseline_probs.get(t1_name, {}).get("champion", 0.0)
    t1_live_champ = live_probs.get(t1_name, {}).get("champion", 0.0)
    t1_champ_delta = t1_live_champ - t1_base_champ
    
    t2_base_champ = baseline_probs.get(t2_name, {}).get("champion", 0.0)
    t2_live_champ = live_probs.get(t2_name, {}).get("champion", 0.0)
    t2_champ_delta = t2_live_champ - t2_base_champ
    
    t1_base_qual = baseline_probs.get(t1_name, {}).get("group_qual", 0.0)
    t1_live_qual = live_probs.get(t1_name, {}).get("group_qual", 0.0)
    t1_qual_delta = t1_live_qual - t1_base_qual
    
    t2_base_qual = baseline_probs.get(t2_name, {}).get("group_qual", 0.0)
    t2_live_qual = live_probs.get(t2_name, {}).get("group_qual", 0.0)
    t2_qual_delta = t2_live_qual - t2_base_qual
    
    def format_qual_odds(base_val, live_val, delta_val):
        if live_val >= 0.999:
            return "Already Qualified"
        if live_val <= 0.001:
            return "Eliminated"
        sign = "+" if delta_val > 0 else ""
        return f"{sign}{delta_val*100:.1f}%"
        
    def format_champ_odds(delta_val):
        sign = "+" if delta_val > 0 else ""
        return f"{sign}{delta_val*100:.2f}%"
        
    return CalibrationMetrics(
        confidence=confidence,
        upset_level=upset_level,
        is_correct=is_correct,
        pred_winner=pred_winner,
        actual_winner=actual_winner,
        t1_champ_change=format_champ_odds(t1_champ_delta),
        t2_champ_change=format_champ_odds(t2_champ_delta),
        t1_qual_change=format_qual_odds(t1_base_qual, t1_live_qual, t1_qual_delta),
        t2_qual_change=format_qual_odds(t2_base_qual, t2_live_qual, t2_qual_delta),
        t1_base_champ=t1_base_champ * 100,
        t1_live_champ=t1_live_champ * 100,
        t2_base_champ=t2_base_champ * 100,
        t2_live_champ=t2_live_champ * 100,
        t1_base_qual=t1_base_qual * 100,
        t1_live_qual=t1_live_qual * 100,
        t2_base_qual=t2_base_qual * 100,
        t2_live_qual=t2_live_qual * 100
    )
