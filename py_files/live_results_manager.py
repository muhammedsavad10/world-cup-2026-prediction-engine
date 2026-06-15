import json
import os
import pandas as pd
from const import WCGroups

group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

def load_live_results(file_path):
    if not os.path.exists(file_path):
        return {"last_updated": "2026-06-11T00:00:00Z", "matches": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "matches" in data:
                return data
    except Exception:
        pass
    return {"last_updated": "2026-06-11T00:00:00Z", "matches": []}

def get_team_group_mapping():
    mapping = {}
    for idx, grp in enumerate(WCGroups):
        letter = group_letters[idx]
        for team in grp:
            mapping[team] = letter
    return mapping

def calculate_group_tables(completed_matches, rankings_df=None):
    """
    Computes standings for each of the 12 groups.
    Tiebreakers applied in order:
    1. Points (Pts)
    2. Goal Difference (GD)
    3. Goals Scored (GS)
    4. FIFA Rank fallback (lower rank number/higher ranking)
    """
    mapping = get_team_group_mapping()
    
    # Initialize stats for all 48 teams
    tables = {letter: {} for letter in group_letters}
    for team, letter in mapping.items():
        tables[letter][team] = {
            "team": team,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "goal_difference": 0,
            "points": 0,
            "matches_played": 0
        }
        
    # Process completed group stage matches
    for match in completed_matches:
        if match.get("stage") != "group_stage":
            continue
            
        home = match["home_team"]
        away = match["away_team"]
        h_score = int(match["home_score"])
        a_score = int(match["away_score"])
        
        home_grp = mapping.get(home)
        away_grp = mapping.get(away)
        
        if not home_grp or not away_grp or home_grp != away_grp:
            continue  # Safety check
            
        h_stats = tables[home_grp][home]
        a_stats = tables[away_grp][away]
        
        h_stats["matches_played"] += 1
        a_stats["matches_played"] += 1
        h_stats["goals_scored"] += h_score
        h_stats["goals_conceded"] += a_score
        a_stats["goals_scored"] += a_score
        a_stats["goals_conceded"] += h_score
        
        if h_score > a_score:
            h_stats["wins"] += 1
            h_stats["points"] += 3
            a_stats["losses"] += 1
        elif h_score == a_score:
            h_stats["draws"] += 1
            h_stats["points"] += 1
            a_stats["draws"] += 1
            a_stats["points"] += 1
        else:
            a_stats["wins"] += 1
            a_stats["points"] += 3
            h_stats["losses"] += 1
            
    # Compute goal differences
    for grp in tables:
        for team in tables[grp]:
            t_stats = tables[grp][team]
            t_stats["goal_difference"] = t_stats["goals_scored"] - t_stats["goals_conceded"]

    # Pre-build dictionary of ranks to avoid slow pandas filter in loop
    team_ranks = {}
    if rankings_df is not None:
        if isinstance(rankings_df, dict):
            team_ranks = rankings_df
        else:
            # Vectorized extraction of the latest rank for each country
            latest_ranks = rankings_df.drop_duplicates(subset=['country_full'], keep='last')
            team_ranks = dict(zip(latest_ranks['country_full'], latest_ranks['rank']))

    # Helper to get FIFA rank for tiebreakers
    def get_team_rank(team_name):
        return team_ranks.get(team_name, 999)

    # Sort each group table
    sorted_tables = {}
    for grp in group_letters:
        team_list = list(tables[grp].values())
        # Sort by points (desc), GD (desc), GS (desc), then FIFA Rank (asc)
        team_list.sort(key=lambda x: (x["points"], x["goal_difference"], x["goals_scored"], -get_team_rank(x["team"])), reverse=True)
        sorted_tables[grp] = team_list
        
    return sorted_tables

def calculate_rolling_form(completed_matches):
    """
    Computes the rolling tournament form (e.g. 'WWD', 'WL') for each team.
    """
    form_dict = {}
    
    # Sort matches chronologically if date is present
    sorted_matches = sorted(completed_matches, key=lambda x: x.get("date", ""))
    
    for match in sorted_matches:
        home = match["home_team"]
        away = match["away_team"]
        h_score = int(match["home_score"])
        a_score = int(match["away_score"])
        
        form_dict.setdefault(home, [])
        form_dict.setdefault(away, [])
        
        if h_score > a_score:
            form_dict[home].append("W")
            form_dict[away].append("L")
        elif h_score == a_score:
            form_dict[home].append("D")
            form_dict[away].append("D")
        else:
            form_dict[home].append("L")
            form_dict[away].append("W")
            
    # Convert lists to strings
    result_form = {}
    for team, history in form_dict.items():
        result_form[team] = "".join(history)
        
    return result_form
