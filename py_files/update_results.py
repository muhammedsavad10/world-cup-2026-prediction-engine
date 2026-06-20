import os
import sys
import json
import datetime
import requests
import time
import logging
import numpy as np
import math

from const import WCGroups, data_dir_path
from preprocess import load_data
import preprocess
from tournament_simulator import TournamentSimulator

# 1. Structured Ingest Logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger()

# 2. Team Name Normalization Map
TEAM_MAP = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "IR Iran": "Iran",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "UAE": "United Arab Emirates"
}

def normalize_team_name(name):
    name = name.strip()
    return TEAM_MAP.get(name, name)

# Flatten WCGroups for quick verification
WCGroups_flat = {team for grp in WCGroups for team in grp}
group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

VALID_STAGES = {"group_stage", "round_of_32", "round_of_16", "quarter_final", "semi_final", "third_place", "final"}

def get_stage_by_match_num(match_num):
    if match_num <= 72:
        return "group_stage"
    elif match_num <= 88:
        return "round_of_32"
    elif match_num <= 96:
        return "round_of_16"
    elif match_num <= 100:
        return "quarter_final"
    elif match_num <= 102:
        return "semi_final"
    elif match_num == 103:
        return "third_place"
    else:
        return "final"

# 3. Model Inference Engine (JSON Coefficients Evaluator)
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

# 4. Strict Validation Pipeline
def validate_live_results(data):
    matches = data.get("matches", [])
    if not matches:
        return True
        
    # Check match continuity
    match_numbers = [m["match_number"] for m in matches]
    if len(match_numbers) != len(set(match_numbers)):
        raise ValueError("Data Validation Error: Duplicate match numbers detected.")
        
    sorted_nums = sorted(match_numbers)
    if sorted_nums[0] != 1:
        raise ValueError("Data Validation Error: Match numbers must start from 1.")
        
    for idx, num in enumerate(sorted_nums):
        if num != idx + 1:
            raise ValueError(f"Data Validation Error: Match number gap detected: expected {idx + 1}, found {num}.")
            
    # Check bounds and roster integrity
    for m in matches:
        home = m.get("home_team")
        away = m.get("away_team")
        h_score = m.get("home_score")
        a_score = m.get("away_score")
        stage = m.get("stage")
        winner = m.get("winner")
        
        if home not in WCGroups_flat or away not in WCGroups_flat:
            raise ValueError(f"Data Validation Error: Invalid team name: '{home}' vs '{away}'")
            
        if not isinstance(h_score, int) or not isinstance(a_score, int):
            raise ValueError(f"Data Validation Error: Invalid score types in match {m['match_number']}")
            
        if not (0 <= h_score <= 20) or not (0 <= a_score <= 20):
            raise ValueError(f"Data Validation Error: Score out of bounds in match {m['match_number']}")
            
        if stage not in VALID_STAGES:
            raise ValueError(f"Data Validation Error: Invalid stage: '{stage}'")
            
        if winner not in [home, away, "Draw", None]:
            raise ValueError(f"Data Validation Error: Invalid winner '{winner}' for match {m['match_number']}")
            
    return True

# 5. Ingest Retry and Backoff Logic
def verify_league_id_1(api_key, max_retries=3):
    url = "https://v3.football.api-sports.io/leagues"
    headers = {
        "x-apisports-key": api_key
    }
    params = {
        "id": "1"
    }
    backoff = 2
    logger.info("Verifying API-Football League ID 1 metadata on startup...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "errors" in data and data["errors"]:
                    err_msg = json.dumps(data["errors"])
                    logger.warning(f"Leagues verification returned errors: {err_msg}. Retrying...")
                else:
                    leagues = data.get("response", [])
                    if not leagues:
                        raise ValueError("League ID 1 not found in API response.")
                        
                    league_name = leagues[0].get("league", {}).get("name")
                    if league_name != "World Cup":
                        raise ValueError(f"Startup verification failed: League ID 1 corresponds to '{league_name}', expected 'World Cup'.")
                    logger.info("Startup verification succeeded: League ID 1 verified as 'World Cup'.")
                    return
            elif response.status_code == 429:
                logger.warning(f"Rate limit hit (429). Retrying in {backoff} seconds...")
            else:
                logger.warning(f"API returned status {response.status_code}. Retrying in {backoff} seconds...")
        except ValueError as ve:
            # Re-raise explicit schema/value check failures immediately
            raise ve
        except Exception as e:
            logger.warning(f"Leagues verification request failed: {e}. Retrying in {backoff} seconds...")
            
        if attempt < max_retries - 1:
            time.sleep(backoff)
            backoff *= 2
            
    raise RuntimeError("Failed to verify league ID 1 from API-Football after multiple retries.")

def fetch_api_football_with_retry(api_key, max_retries=3):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        "x-apisports-key": api_key
    }
    params = {
        "league": "1",
        "season": "2026"
    }
    backoff = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching API-Football fixtures (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "errors" in data and data["errors"]:
                    err_msg = json.dumps(data["errors"])
                    logger.warning(f"API-Football returned errors: {err_msg}")
                else:
                    return data
            elif response.status_code == 429:
                logger.warning(f"Rate limit hit (429). Retrying in {backoff} seconds...")
            else:
                logger.warning(f"API returned status {response.status_code}. Retrying in {backoff} seconds...")
        except Exception as e:
            logger.warning(f"Request failed: {e}. Retrying in {backoff} seconds...")
        time.sleep(backoff)
        backoff *= 2
    raise RuntimeError("Failed to fetch fixtures from API-Football after multiple retries.")

def resolve_match_numbers(raw_data):
    fixtures = raw_data.get("response", [])
    completed_fixtures = []
    
    # Filter completed fixtures based on status
    COMPLETED_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}
    for f in fixtures:
        status = f.get("fixture", {}).get("status", {}).get("short")
        if status in COMPLETED_STATUSES:
            completed_fixtures.append(f)
            
    group_fixtures = []
    ko_fixtures = []
    
    for f in completed_fixtures:
        round_name = f.get("league", {}).get("round", "")
        if "Group Stage" in round_name:
            group_fixtures.append(f)
        else:
            ko_fixtures.append(f)
            
    # Sort chronologically by timestamp
    group_fixtures = sorted(group_fixtures, key=lambda x: x.get("fixture", {}).get("timestamp", 0))
    ko_fixtures = sorted(ko_fixtures, key=lambda x: x.get("fixture", {}).get("timestamp", 0))
    
    completed_matches = []
    
    # 1. Map Group Stage Matches (1 to N)
    for idx, f in enumerate(group_fixtures):
        fixture_info = f.get("fixture", {})
        teams = f.get("teams", {})
        home_name = normalize_team_name(teams.get("home", {}).get("name", ""))
        away_name = normalize_team_name(teams.get("away", {}).get("name", ""))
        goals = f.get("goals", {})
        h_score = goals.get("home")
        a_score = goals.get("away")
        
        if h_score is None or a_score is None or not home_name or not away_name:
            continue
            
        h_score = int(h_score)
        a_score = int(a_score)
        
        winner = None
        if h_score > a_score:
            winner = home_name
        elif a_score > h_score:
            winner = away_name
        else:
            winner = "Draw"
            
        group_letter = "A"
        for grp_idx, grp in enumerate(WCGroups):
            if home_name in grp:
                group_letter = group_letters[grp_idx]
                break
                
        completed_matches.append({
            "match_number": idx + 1,
            "date": fixture_info.get("date", "2026-06-11")[:10],
            "home_team": home_name,
            "away_team": away_name,
            "home_score": h_score,
            "away_score": a_score,
            "winner": winner,
            "group": group_letter,
            "stage": "group_stage"
        })
        
    if not ko_fixtures:
        return completed_matches
        
    # 2. Map Knockout Matches (73 to 104) dynamically using a bracket walk
    for f in ko_fixtures:
        fixture_info = f.get("fixture", {})
        teams = f.get("teams", {})
        home_name = normalize_team_name(teams.get("home", {}).get("name", ""))
        away_name = normalize_team_name(teams.get("away", {}).get("name", ""))
        goals = f.get("goals", {})
        h_score = goals.get("home")
        a_score = goals.get("away")
        
        if h_score is None or a_score is None or not home_name or not away_name:
            continue
            
        h_score = int(h_score)
        a_score = int(a_score)
        
        winner = None
        if h_score > a_score:
            winner = home_name
        elif a_score > h_score:
            winner = away_name
        else:
            # Penalty shootout
            penalty = f.get("score", {}).get("penalty", {})
            pen_home = penalty.get("home")
            pen_away = penalty.get("away")
            if pen_home is not None and pen_away is not None:
                if int(pen_home) > int(pen_away):
                    winner = home_name
                elif int(pen_away) > int(pen_home):
                    winner = away_name
            # Fallback to winner boolean
            if winner is None:
                home_win = teams.get("home", {}).get("winner")
                if home_win is True:
                    winner = home_name
                elif home_win is False:
                    winner = away_name
            if winner is None:
                winner = "Draw"
                
        # Walk the bracket to identify the match_id (73-104) playing these teams
        completed_group_lookup = {}
        completed_ko_lookup = {}
        for m in completed_matches:
            if m["stage"] == "group_stage":
                completed_group_lookup[(m["home_team"], m["away_team"])] = m
                completed_group_lookup[(m["away_team"], m["home_team"])] = m
            else:
                completed_ko_lookup[int(m["match_number"])] = m
                
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]])
                
        temp_sim = TournamentSimulator(datetime.date(2026, 6, 11), DummyModel(), WCGroups, lambda t1, t2: [0]*7)
        temp_sim.probabilistic = False
        
        group_results, advancing_thirds_mapping = temp_sim.playGroupStage(
            start_date=temp_sim.tournamentStartDate, 
            completed_lookup=completed_group_lookup
        )
        
        knockout_schedule = {
            73: ("RU_A", "RU_B"), 74: ("W_E", "3rd_E"), 75: ("W_F", "RU_C"), 76: ("W_C", "RU_F"),
            77: ("W_I", "3rd_I"), 78: ("RU_E", "RU_I"), 79: ("W_A", "3rd_A"), 80: ("W_L", "3rd_L"),
            81: ("W_D", "3rd_D"), 82: ("W_G", "3rd_G"), 83: ("RU_K", "RU_L"), 84: ("W_H", "RU_J"),
            85: ("W_B", "3rd_B"), 86: ("W_J", "RU_H"), 87: ("W_K", "3rd_K"), 88: ("RU_D", "RU_G"),
            89: ("W74", "W77"), 90: ("W73", "W75"), 91: ("W76", "W78"), 92: ("W79", "W80"),
            93: ("W83", "W84"), 94: ("W81", "W82"), 95: ("W86", "W88"), 96: ("W85", "W87"),
            97: ("W89", "W90"), 98: ("W93", "W94"), 99: ("W91", "W92"), 100: ("W95", "W96"),
            101: ("W97", "W98"), 102: ("W99", "W100"), 103: ("L101", "L102"), 104: ("W101", "W102")
        }
        
        match_winners = {}
        match_losers = {}
        match_id_found = None
        
        play_order = list(range(73, 103)) + [104, 103]
        for match_id in play_order:
            home_key, away_key = knockout_schedule[match_id]
            
            def resolve_team(key):
                if key.startswith("W_"):
                    return group_results[key.split("_")[1]]['winner']
                elif key.startswith("RU_"):
                    return group_results[key.split("_")[1]]['runner_up']
                elif key.startswith("3rd_"):
                    return advancing_thirds_mapping[key.split("_")[1]]
                elif key.startswith("W"):
                    return match_winners[int(key[1:])]
                elif key.startswith("L"):
                    return match_losers[int(key[1:])]
                    
            t1 = resolve_team(home_key)
            t2 = resolve_team(away_key)
            
            if (t1 == home_name and t2 == away_name) or (t1 == away_name and t2 == home_name):
                match_id_found = match_id
                
            match_record = completed_ko_lookup.get(match_id)
            if match_record is not None:
                winner_t = match_record["winner"]
            else:
                winner_t = winner if match_id_found == match_id else t1
                
            match_winners[match_id] = winner_t
            match_losers[match_id] = t2 if winner_t == t1 else t1
            
            if match_id_found == match_id:
                break
                
        if match_id_found is None:
            logger.warning(f"Could not map knockout fixture: {home_name} vs {away_name}")
            continue
            
        completed_matches.append({
            "match_number": match_id_found,
            "date": fixture_info.get("date", "2026-06-11")[:10],
            "home_team": home_name,
            "away_team": away_name,
            "home_score": h_score,
            "away_score": a_score,
            "winner": winner,
            "group": None,
            "stage": get_stage_by_match_num(match_id_found)
        })
        
    return completed_matches

def fetch_and_update():
    logger.info("=== STARTING LIVE RESULTS INGESTION ENGINE ===")
    
    local_json = os.path.join(data_dir_path, "world_cup_2026_live_results.json")
    local_probs_json = os.path.join(data_dir_path, "world_cup_2026_live_probabilities.json")
    model_json_path = os.path.join(data_dir_path, "logistic_regression_model.json")
    
    # 1. Load existing results
    existing_data = {"last_updated": "2026-06-11T00:00:00Z", "matches": []}
    if os.path.exists(local_json):
        try:
            with open(local_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read local results database: {e}")
            return False
            
    # 2. Fetch matches from API-Football
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key or api_key == "MOCK":
        logger.info("API_FOOTBALL_KEY is missing or set to MOCK. Using local mock payload for dry run.")
        completed_fetched = existing_data.get("matches", [])
    else:
        try:
            # Startup league verification
            verify_league_id_1(api_key)
            
            raw_payload = fetch_api_football_with_retry(api_key)
            if not raw_payload or "response" not in raw_payload or not raw_payload["response"]:
                raise ValueError("API-Football response is empty or invalid.")
            if "errors" in raw_payload and raw_payload["errors"]:
                raise ValueError(f"API-Football returned errors: {json.dumps(raw_payload['errors'])}")
                
            completed_fetched = resolve_match_numbers(raw_payload)
        except Exception as e:
            logger.error(f"Ingestion crashed during API-Football sync: {e}")
            return False
            
    # 3. Check for updates / merges
    existing_matches = {m["match_number"]: m for m in existing_data.get("matches", [])}
    updated_count = 0
    
    for match in completed_fetched:
        match_num = match["match_number"]
        if match_num not in existing_matches:
            existing_matches[match_num] = match
            updated_count += 1
        else:
            old = existing_matches[match_num]
            if (old["home_score"] != match["home_score"] or 
                old["away_score"] != match["away_score"] or 
                old["winner"] != match["winner"]):
                existing_matches[match_num] = match
                updated_count += 1
                
    # If there are no new updates, and the probabilities file exists, short-circuit immediately
    if updated_count == 0 and os.path.exists(local_probs_json):
        logger.info("No updates detected. Local results database and probabilities are fully up to date. Short-circuiting.")
        return True
        
    # Merge matches list
    merged_matches = sorted(existing_matches.values(), key=lambda x: x["match_number"])
    new_results_data = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "matches": merged_matches
    }
    
    # Run validation checks on structural data integrity
    try:
        validate_live_results(new_results_data)
    except Exception as val_error:
        logger.error(f"Transactional Validation Failed: {val_error}")
        return False
        
    # 4. Monte Carlo Re-Simulation (Deterministic 10,000 Runs)
    logger.info("Executing 10,000 Monte Carlo Re-Simulations inside GitHub runner...")
    try:
        # Load pre-trained JSON model coefficients
        if not os.path.exists(model_json_path):
            raise FileNotFoundError(f"Model coefficients JSON not found at {model_json_path}")
            
        with open(model_json_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
            
        model_wrapper = JSONLogisticRegressionModel(model_data)
        
        # Load datasets and calculate features
        X_train, y_train, rankings_df = load_data()
        
        # Lock qualified teams
        qualified_teams = set()
        for grp in WCGroups:
            for team in grp:
                qualified_teams.add(team)
                
        preprocess.final_team_stats = {
            team: stats for team, stats in preprocess.final_team_stats.items() if team in qualified_teams
        }
        rankings_df = rankings_df[rankings_df['country_full'].isin(qualified_teams)].copy()
        
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
            
        # Seed simulation for exact reproducibility
        np.random.seed(42)
        
        simulator = TournamentSimulator(datetime.date(2026, 6, 11), model_wrapper, WCGroups, feature_generator)
        simulator.gamma = 0.15
        simulator.prob_cap = 0.12
        
        # Run MC (10,000 runs)
        live_probs = simulator.run_monte_carlo_simulation(merged_matches, num_runs=10000)
        
        # Validate mathematical invariants
        champ_sum = sum(t["champion"] for t in live_probs.values())
        if not (0.99990 <= champ_sum <= 1.00010):
            raise ValueError(f"Invariant Violated: Champion probability sum = {champ_sum} (expected 1.0)")
            
        qual_sum = sum(t["group_qual"] for t in live_probs.values())
        if not (31.9990 <= qual_sum <= 32.0010):
            raise ValueError(f"Invariant Violated: Group qualification sum = {qual_sum} (expected 32.0)")
            
        # Structure the probabilities payload
        probabilities_data = {
            "metadata": {
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                "model_version": "1.0.0",
                "simulation_runs": 10000,
                "seed": 42,
                "schema_version": "1.1.0"
            },
            "probabilities": {
                team: {
                    k: round(v, 5) if isinstance(v, float) else v
                    for k, v in metrics.items()
                }
                for team, metrics in live_probs.items()
            }
        }
    except Exception as sim_error:
        logger.error(f"Simulation execution failed: {sim_error}")
        return False
        
    # 5. Transactional Atomic File Swaps
    try:
        temp_results = local_json + ".tmp"
        temp_probs = local_probs_json + ".tmp"
        
        with open(temp_results, "w", encoding="utf-8") as f:
            json.dump(new_results_data, f, indent=2, sort_keys=True)
            
        with open(temp_probs, "w", encoding="utf-8") as f:
            json.dump(probabilities_data, f, indent=2, sort_keys=True)
            
        os.replace(temp_results, local_json)
        os.replace(temp_probs, local_probs_json)
        logger.info(f"Database transaction successfully committed. Added/Updated {updated_count} matches.")
    except Exception as io_error:
        logger.error(f"Transactional write replacement failed: {io_error}")
        # Cleanup temp files if they exist
        for p in [temp_results, temp_probs]:
            if os.path.exists(p):
                os.remove(p)
        return False
        
    return True

if __name__ == "__main__":
    success = fetch_and_update()
    if not success:
        logger.error("PIPELINE EXECUTION FAILED.")
        sys.exit(1)
    logger.info("PIPELINE EXECUTION SUCCEEDED.")
    sys.exit(0)
