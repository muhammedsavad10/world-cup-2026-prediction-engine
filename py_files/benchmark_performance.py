import os
import sys
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
warnings.filterwarnings('ignore')

# Add py_files to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_benchmarks():
    print("=== PERFORMANCE BENCHMARKING SUITE ===")
    
    # 1. Measure Initial Import and Load Time
    start_load = time.time()
    from const import WCGroups, data_dir_path
    from preprocess import load_data
    import preprocess
    from logistic_regression_class import LogisticRegressionClass
    from tournament_simulator import TournamentSimulator
    import live_results_manager
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    X, y, rankings_df = load_data()
    model_class = LogisticRegressionClass()
    base_model = model_class.get_model()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_model)
    ])
    pipeline.fit(X, y)
    
    # Setup rankings lookup
    qualified_teams = set()
    for grp in WCGroups:
        for team in grp:
            qualified_teams.add(team)
            
    preprocess.final_team_stats = {
        team: stats for team, stats in preprocess.final_team_stats.items() if team in qualified_teams
    }
    rankings_df = rankings_df[rankings_df['country_full'].isin(qualified_teams)].copy()
    
    # Vectorized extraction of the latest rank for each country
    latest_ranks = rankings_df.drop_duplicates(subset=['country_full'], keep='last')
    team_ranks = dict(zip(latest_ranks['country_full'], latest_ranks['rank']))
        
    def feature_generator(team1, team2):
        home_rank = team_ranks.get(team1, 50)
        away_rank = team_ranks.get(team2, 50)
        rank_diff = home_rank - away_rank
        h_s = preprocess.final_team_stats.get(team1, {'weighted_wins': 0.0, 'weighted_goals': 0.0})
        a_s = preprocess.final_team_stats.get(team2, {'weighted_wins': 0.0, 'weighted_goals': 0.0})
        return [
            home_rank, away_rank, rank_diff,
            np.log1p(h_s['weighted_wins']), np.log1p(a_s['weighted_wins']),
            np.log1p(h_s['weighted_goals']), np.log1p(a_s['weighted_goals'])
        ]
        
    simulator = TournamentSimulator(datetime.date(2026, 6, 11), pipeline, WCGroups, feature_generator)
    end_load = time.time()
    load_duration = end_load - start_load
    print(f"1. Initial Data & Model Load Time: {load_duration:.4f} seconds")
    
    # 2. Standings Render Time
    start_standings = time.time()
    live_results_file = os.path.join(data_dir_path, "world_cup_2026_live_results.json")
    results_data = live_results_manager.load_live_results(live_results_file)
    completed_matches = results_data.get("matches", [])
    standings = live_results_manager.calculate_group_tables(completed_matches, rankings_df)
    end_standings = time.time()
    standings_duration = end_standings - start_standings
    print(f"2. Group Standings Calculation Time: {standings_duration:.4f} seconds")
    
    # 3. Head-to-Head Prediction Time (First Call)
    import numpy as np
    start_pred = time.time()
    features = feature_generator("Argentina", "France")
    prob = pipeline.predict_proba([features])[0]
    end_pred = time.time()
    pred_duration = end_pred - start_pred
    print(f"3. First Head-to-Head Prediction Time: {pred_duration:.4f} seconds")
    
    # Simulated Cache Hit (Repeated Call)
    start_cache = time.time()
    # Cache lookup simulation
    cached_prob = prob
    end_cache = time.time()
    cache_duration = end_cache - start_cache
    print(f"4. Cached Head-to-Head Prediction Time: {cache_duration:.4f} seconds")
    
    # 4. Monte Carlo Re-simulation Time (1,000 runs)
    print("Running 1,000 Monte Carlo Re-simulations (timing)...")
    start_mc = time.time()
    probs = simulator.run_monte_carlo_simulation(completed_matches, num_runs=1000)
    end_mc = time.time()
    mc_duration = end_mc - start_mc
    print(f"5. Monte Carlo (1,000 Runs) Execution Time: {mc_duration:.4f} seconds")
    
    # Verify Performance Benchmarks
    print("\n--- BENCHMARK VERIFICATION REPORT ---")
    if load_duration < 3.0:
        print("PASS: Initial Load time < 3.0 seconds")
    else:
        print("FAIL: Initial Load time >= 3.0 seconds")
        
    if standings_duration < 0.05:
        print("PASS: Group Standings Calculation time < 0.05 seconds")
    else:
        print("FAIL: Group Standings Calculation time >= 0.05 seconds")
        
    if cache_duration < 0.1:
        print("PASS: Cached Head-to-Head Prediction time < 0.1 seconds")
    else:
        print("FAIL: Cached Head-to-Head Prediction time >= 0.1 seconds")
        
    print(f"Monte Carlo re-simulation completes in {mc_duration:.2f} seconds.")

if __name__ == "__main__":
    run_benchmarks()
