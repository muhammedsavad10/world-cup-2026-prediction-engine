import pandas as pd
import numpy as np
import os.path as osp

from const import data_dir_path

# Global stats to be used by the tournament simulator
final_team_stats = {}

def load_data():
    global final_team_stats
    results = pd.read_csv(osp.join(data_dir_path, 'results.csv'))
    rankings = pd.read_csv(osp.join(data_dir_path, 'fifa_ranking-2024-06-20.csv'))
    
    # Standardize country names to match WCGroups and each other
    results['home_team'] = results['home_team'].replace({'United States': 'USA', 'Czech Republic': 'Czechia'})
    results['away_team'] = results['away_team'].replace({'United States': 'USA', 'Czech Republic': 'Czechia'})
    results['country'] = results['country'].replace({'United States': 'USA', 'Czech Republic': 'Czechia'})

    rankings['country_full'] = rankings['country_full'].replace({
        'Korea Republic': 'South Korea',
        'China PR': 'China',
        'IR Iran': 'Iran'
    })
    
    results['date'] = pd.to_datetime(results['date'])
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    
    # strictly filter out unplayed fixtures
    results = results.dropna(subset=['home_score', 'away_score']).copy()
    
    # Filter by Date: Drop any rows where the date is greater than June 10, 2026
    results = results[results['date'] <= '2026-06-10'].copy()
    
    # Hard Cutoff: drop matches before 2018
    results = results[results['date'] >= '2018-01-01'].copy()
    

    
    # Filter Out Friendlies:
    # matches where the tournament column equals "Friendly" are completely dropped
    results = results[results['tournament'] != 'Friendly'].copy()
    
    # Sort for merge_asof
    results = results.sort_values('date')
    rankings = rankings.sort_values('rank_date')
    
    # Merge Home Rank
    results = pd.merge_asof(results, rankings[['rank_date', 'country_full', 'rank']], 
                            left_on='date', right_on='rank_date', 
                            left_by='home_team', right_by='country_full', direction='backward')
    results = results.rename(columns={'rank': 'home_rank'})
    
    # Merge Away Rank
    results = pd.merge_asof(results, rankings[['rank_date', 'country_full', 'rank']], 
                            left_on='date', right_on='rank_date', 
                            left_by='away_team', right_by='country_full', direction='backward')
    results = results.rename(columns={'rank': 'away_rank'})
    
    results = results.dropna(subset=['home_rank', 'away_rank']).copy()
    
    # Create simple features
    results['rank_diff'] = results['home_rank'] - results['away_rank']
    results['home_match_for_home_team'] = (results['country'] == results['home_team']).astype(int)
    
    # Calculate Quality-Weighted Wins and Goals Scored
    team_history = {}
    
    home_weighted_wins_list = []
    away_weighted_wins_list = []
    home_weighted_goals_list = []
    away_weighted_goals_list = []
    
    for idx, row in results.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        home_rank = row['home_rank']
        away_rank = row['away_rank']
        
        # 1. Fetch rolling stats of home team (prior matches)
        home_hist = team_history.get(home_team, [])
        recent_home = home_hist[-15:]
        h_wins = sum(h['weighted_win'] for h in recent_home) if recent_home else 0.0
        h_goals = sum(h['weighted_goals'] for h in recent_home) if recent_home else 0.0
        
        # 2. Fetch rolling stats of away team (prior matches)
        away_hist = team_history.get(away_team, [])
        recent_away = away_hist[-15:]
        a_wins = sum(h['weighted_win'] for h in recent_away) if recent_away else 0.0
        a_goals = sum(h['weighted_goals'] for h in recent_away) if recent_away else 0.0
        
        home_weighted_wins_list.append(h_wins)
        away_weighted_wins_list.append(a_wins)
        home_weighted_goals_list.append(h_goals)
        away_weighted_goals_list.append(a_goals)
        
        # 3. Calculate this match's stats to record in history
        years_ago = max(0, (pd.to_datetime('2026-06-01') - row['date']).days / 365.25)
        decay_multiplier = 0.5 ** (years_ago / 2)

        w_opp_rank = max(0.0, min(5.0, 100.0 / away_rank))
        w_home_rank = max(0.0, min(5.0, 100.0 / home_rank))

        h_win_val = (w_opp_rank if home_score > away_score else 0.0) * decay_multiplier
        h_goals_val = (home_score * w_opp_rank) * decay_multiplier
        
        a_win_val = (w_home_rank if away_score > home_score else 0.0) * decay_multiplier
        a_goals_val = (away_score * w_home_rank) * decay_multiplier
        
        if home_team not in team_history:
            team_history[home_team] = []
        team_history[home_team].append({
            'weighted_win': h_win_val,
            'weighted_goals': h_goals_val
        })
        
        if away_team not in team_history:
            team_history[away_team] = []
        team_history[away_team].append({
            'weighted_win': a_win_val,
            'weighted_goals': a_goals_val
        })
    
    # Store the final team stats at the end of the history
    final_team_stats.clear()
    for team, hist in team_history.items():
        recent = hist[-15:]
        final_team_stats[team] = {
            'weighted_wins': sum(h['weighted_win'] for h in recent),
            'weighted_goals': sum(h['weighted_goals'] for h in recent)
        }
        
    results['home_weighted_wins'] = home_weighted_wins_list
    results['away_weighted_wins'] = away_weighted_wins_list
    results['home_weighted_goals'] = home_weighted_goals_list
    results['away_weighted_goals'] = away_weighted_goals_list
    
    # Filter out draws to center probabilities at 0.5 (model only learns from decisive matches)
    results = results[results['home_score'] != results['away_score']].copy()
    
    # Double the dataset to create a perfectly symmetric training set
    p1 = pd.DataFrame()
    p1['home_rank'] = results['home_rank']
    p1['away_rank'] = results['away_rank']
    p1['rank_diff'] = results['rank_diff']
    p1['home_match_for_home_team'] = results['home_match_for_home_team']
    p1['home_weighted_wins'] = results['home_weighted_wins']
    p1['away_weighted_wins'] = results['away_weighted_wins']
    p1['home_weighted_goals'] = results['home_weighted_goals']
    p1['away_weighted_goals'] = results['away_weighted_goals']
    y1 = (results['home_score'] > results['away_score']).astype(int)
    
    p2 = pd.DataFrame()
    p2['home_rank'] = results['away_rank']
    p2['away_rank'] = results['home_rank']
    p2['rank_diff'] = -results['rank_diff']
    p2['home_match_for_home_team'] = (results['country'] == results['away_team']).astype(int)
    p2['home_weighted_wins'] = results['away_weighted_wins']
    p2['away_weighted_wins'] = results['home_weighted_wins']
    p2['home_weighted_goals'] = results['away_weighted_goals']
    p2['away_weighted_goals'] = results['home_weighted_goals']
    y2 = (results['away_score'] > results['home_score']).astype(int)
    
    features = [
        'home_rank', 'away_rank', 'rank_diff',
        'home_weighted_wins', 'away_weighted_wins',
        'home_weighted_goals', 'away_weighted_goals'
    ]
    X = pd.concat([p1, p2], ignore_index=True)[features]
    
    # Apply log1p transformation to prevent outlier amplification
    for col in ['home_weighted_wins', 'away_weighted_wins', 'home_weighted_goals', 'away_weighted_goals']:
        X[col] = np.log1p(X[col])
        
    y = pd.concat([y1, y2], ignore_index=True)
    
    return X, y, rankings

def generate_features():
    # Only for compatibility with old code if needed
    pass

def get_match_features(home_team, away_team, rankings_df):
    home_rank = rankings_df[rankings_df['country_full'] == home_team]['rank'].values
    away_rank = rankings_df[rankings_df['country_full'] == away_team]['rank'].values
    
    home_rank = home_rank[-1] if len(home_rank) > 0 else 50
    away_rank = away_rank[-1] if len(away_rank) > 0 else 50
    
    rank_diff = home_rank - away_rank
    
    h_stats = final_team_stats.get(home_team, {'weighted_wins': 0.0, 'weighted_goals': 0.0})
    a_stats = final_team_stats.get(away_team, {'weighted_wins': 0.0, 'weighted_goals': 0.0})
    
    return [
        home_rank, away_rank, rank_diff,
        np.log1p(h_stats['weighted_wins']), np.log1p(a_stats['weighted_wins']),
        np.log1p(h_stats['weighted_goals']), np.log1p(a_stats['weighted_goals'])
    ]
if __name__ == '__main__':
    import os.path as osp
    import pandas as pd
    from const import data_dir_path
    
    match_results_df = pd.read_csv(osp.join(data_dir_path, 'results.csv'))
    match_results_df['date'] = pd.to_datetime(match_results_df['date'])
    
    # Strict Purge for the diagnostic print
    match_results_df = match_results_df.dropna(subset=['home_score', 'away_score'])
    match_results_df = match_results_df[match_results_df['date'] <= '2026-06-10']
    
    print(f"The latest played match in the dataset is from: {match_results_df['date'].max()}")