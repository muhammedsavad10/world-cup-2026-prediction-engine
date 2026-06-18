import datetime
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

class TournamentSimulator:
    def __init__(self, start_date, model, groups, feature_generator_func):
        self.tournamentStartDate = start_date
        self.model = model
        self.groups = groups
        self.model_name = "Logistic Regression"
        self.feature_generator_func = feature_generator_func
        self.probabilistic = False
        self.avg_form = {}
        self.match_counts = {}
        self.gamma = 0.03

    def predictWinner(self, team1, team2, date, stage):
        if not hasattr(self, 'prediction_cache'):
            self.prediction_cache = {}
        
        cache_key = (team1, team2)
        if cache_key in self.prediction_cache:
            home_win_prob = self.prediction_cache[cache_key]
        else:
            X = self.feature_generator_func(team1, team2)
            prob = self.model.predict_proba([X])[0]
            home_win_prob = prob[1]
            
            # Apply Dynamic Tournament Form Layer using log-odds (logit) scaling
            if hasattr(self, 'avg_form') and self.avg_form and hasattr(self, 'match_counts') and self.match_counts:
                n1 = self.match_counts.get(team1, 0)
                n2 = self.match_counts.get(team2, 0)
                
                def get_weight(n):
                    if n <= 0: return 0.0
                    elif n == 1: return 0.25
                    elif n == 2: return 0.60
                    else: return 1.00
                    
                w1 = get_weight(n1)
                w2 = get_weight(n2)
                
                f1 = self.avg_form.get(team1, 0.0) * w1
                f2 = self.avg_form.get(team2, 0.0) * w2
                
                # Convert baseline probability to logit (clip to avoid division by zero / log(0))
                p_base = max(0.0001, min(0.9999, home_win_prob))
                logit_base = np.log(p_base / (1.0 - p_base))
                
                # Apply gamma and scaling in log-odds space
                gamma = getattr(self, 'gamma', 0.03)
                logit_adj = gamma * (f1 - f2)
                logit_adjusted = logit_base + logit_adj
                
                # Convert back to probability
                p_adjusted_uncapped = 1.0 / (1.0 + np.exp(-logit_adjusted))
                
                # Apply hard win-probability adjustment cap of +/- 5%
                prob_change = p_adjusted_uncapped - home_win_prob
                prob_change_capped = max(-0.05, min(0.05, prob_change))
                
                home_win_prob = home_win_prob + prob_change_capped
                home_win_prob = max(0.01, min(0.99, home_win_prob))
            
            self.prediction_cache[cache_key] = home_win_prob
            self.prediction_cache[(team2, team1)] = 1.0 - home_win_prob
            
        if self.probabilistic:
            if np.random.rand() < home_win_prob:
                return team1, home_win_prob
            else:
                return team2, 1.0 - home_win_prob
        else:
            if home_win_prob > 0.5:
                return team1, home_win_prob
            else:
                return team2, 1.0 - home_win_prob

    def playGroupStage(self, start_date, completed_lookup=None):
        num_grps = 12
        group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        group_results = {}
        third_place_candidates = []
        
        for grp in range(num_grps):
            g_letter = group_letters[grp]
            teams = self.groups[grp]
            team_stats = {team: {'wins': 0.0, 'prob_score': 0.0} for team in teams}
            
            for i in range(0, len(teams)):
                for j in range(i + 1, len(teams)):
                    dateOfMatch = start_date + datetime.timedelta(days=i * j)
                    
                    match_record = None
                    if completed_lookup:
                        match_record = completed_lookup.get((teams[i], teams[j]))
                        
                    if match_record is not None:
                        winner = match_record.get("winner")
                        h_score = int(match_record.get("home_score", 0))
                        a_score = int(match_record.get("away_score", 0))
                        
                        if winner == "Draw" or winner is None or h_score == a_score:
                            team_stats[teams[i]]['wins'] += 0.5
                            team_stats[teams[j]]['wins'] += 0.5
                            team_stats[teams[i]]['prob_score'] += 0.5
                            team_stats[teams[j]]['prob_score'] += 0.5
                        else:
                            team_stats[winner]['wins'] += 1.0
                            team_stats[winner]['prob_score'] += 1.0
                            loser = teams[j] if winner == teams[i] else teams[i]
                            team_stats[loser]['prob_score'] += 0.0
                    else:
                        winner, win_prob = self.predictWinner(teams[i], teams[j], dateOfMatch, 'group_stage')
                        team_stats[winner]['wins'] += 1.0
                        team_stats[winner]['prob_score'] += win_prob
                        loser = teams[j] if winner == teams[i] else teams[i]
                        team_stats[loser]['prob_score'] += (1.0 - win_prob)

            sorted_teams = sorted(team_stats.keys(), key=lambda x: (team_stats[x]['wins'], team_stats[x]['prob_score']), reverse=True)
            
            group_results[g_letter] = {
                'winner': sorted_teams[0],
                'runner_up': sorted_teams[1],
                'third_place': sorted_teams[2]
            }
            
            third_place_candidates.append({
                'team': sorted_teams[2],
                'group': g_letter,
                'wins': team_stats[sorted_teams[2]]['wins'],
                'prob_score': team_stats[sorted_teams[2]]['prob_score']
            })

        # Sort the 12 third-place candidates to find the 8 best
        best_thirds = sorted(third_place_candidates, key=lambda x: (x['wins'], x['prob_score']), reverse=True)[:8]
        qualified_third_groups = sorted([t['group'] for t in best_thirds])
        
        # Resolve Annex C mapping
        import os
        import json
        current_dir = os.path.dirname(__file__)
        json_path = os.path.join(current_dir, "annex_c_lookup.json")
        with open(json_path, "r", encoding="utf-8") as f:
            annex_c_lookup = json.load(f)
            
        combo_key = "".join(qualified_third_groups)
        assignments = annex_c_lookup[combo_key]
        
        advancing_thirds_mapping = {}
        for winner_group, opp_group in assignments.items():
            opp_team = group_results[opp_group]['third_place']
            advancing_thirds_mapping[winner_group] = opp_team
            
        return group_results, advancing_thirds_mapping

    def play(self, previousWinners, stage='round_of_32', start_date=datetime.date(2026, 6, 28), labels=[], odds=[]):
        winners = []
        i = 0
        print(f"\nStage: {stage}")
        
        while i < len(previousWinners):
            opp = i + 1
            dateOfMatch = start_date + datetime.timedelta(days=i)
            winner, win_prob = self.predictWinner(previousWinners[i], previousWinners[opp], dateOfMatch, stage)
            
            winners.append(winner)
            
            if winner == previousWinners[i]:
                t1_prob = win_prob
                t2_prob = 1 - win_prob
            else:
                t1_prob = 1 - win_prob
                t2_prob = win_prob
                
            labels.append(f"{previousWinners[i]}({np.round(t1_prob,2)}) vs. {previousWinners[opp]}({np.round(t2_prob,2)})")
            odds.append([t1_prob, t2_prob])
            
            i += 2
            
        print(f'\n{stage} winners: {winners}')
        return winners, dateOfMatch, labels, odds

    def playKnockOuts(self, group_results=None, advancing_thirds_mapping=None, completed_lookup=None):
        if group_results is None or advancing_thirds_mapping is None:
            group_results, advancing_thirds_mapping = self.playGroupStage(
                start_date=self.tournamentStartDate, 
                completed_lookup=completed_lookup
            )
        
        knockout_schedule = {
            73: ("RU_A", "RU_B"),
            74: ("W_E", "3rd_E"),
            75: ("W_F", "RU_C"),
            76: ("W_C", "RU_F"),
            77: ("W_I", "3rd_I"),
            78: ("RU_E", "RU_I"),
            79: ("W_A", "3rd_A"),
            80: ("W_L", "3rd_L"),
            81: ("W_D", "3rd_D"),
            82: ("W_G", "3rd_G"),
            83: ("RU_K", "RU_L"),
            84: ("W_H", "RU_J"),
            85: ("W_B", "3rd_B"),
            86: ("W_J", "RU_H"),
            87: ("W_K", "3rd_K"),
            88: ("RU_D", "RU_G"),
            89: ("W74", "W77"),
            90: ("W73", "W75"),
            91: ("W76", "W78"),
            92: ("W79", "W80"),
            93: ("W83", "W84"),
            94: ("W81", "W82"),
            95: ("W86", "W88"),
            96: ("W85", "W87"),
            97: ("W89", "W90"),
            98: ("W93", "W94"),
            99: ("W91", "W92"),
            100: ("W95", "W96"),
            101: ("W97", "W98"),
            102: ("W99", "W100"),
            103: ("L101", "L102"),
            104: ("W101", "W102")
        }
        
        match_winners = {}
        match_losers = {}
        
        labels = []
        odds = []
        round_results = {
            'round_of_32': [],
            'round_of_16': [],
            'quarter_final': [],
            'semi_final': [],
            'third_place': [],
            'final': []
        }
        
        last_date = self.tournamentStartDate + datetime.timedelta(days=15)
        
        # Play Final (104) before Third-place Playoff (103) to align labels index 30 for the final
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
                else:
                    raise ValueError(f"Unknown key: {key}")
                    
            team1 = resolve_team(home_key)
            team2 = resolve_team(away_key)
            
            dateOfMatch = last_date + datetime.timedelta(days=match_id - 73)
            
            if match_id <= 88:
                stage = 'round_of_32'
            elif match_id <= 96:
                stage = 'round_of_16'
            elif match_id <= 100:
                stage = 'quarter_final'
            elif match_id <= 102:
                stage = 'semi_final'
            elif match_id == 103:
                stage = 'third_place'
            else:
                stage = 'final'
                
            if hasattr(self, 'recorded_matchups'):
                self.recorded_matchups.append((stage, team1, team2))
                
            match_record = None
            if completed_lookup:
                match_record = completed_lookup.get(match_id)
                
            if match_record is not None:
                winner = match_record.get("winner")
                if winner not in (team1, team2):
                    winner, win_prob = self.predictWinner(team1, team2, dateOfMatch, stage)
                else:
                    win_prob = 1.0
            else:
                winner, win_prob = self.predictWinner(team1, team2, dateOfMatch, stage)
                
            loser = team2 if winner == team1 else team1
            
            match_winners[match_id] = winner
            match_losers[match_id] = loser
            
            round_results[stage].append(winner)
            
            t1_prob = win_prob if winner == team1 else 1 - win_prob
            t2_prob = 1 - t1_prob
            
            labels.append(f"{team1}({np.round(t1_prob,2)}) vs. {team2}({np.round(t2_prob,2)})")
            odds.append([t1_prob, t2_prob])
            
        return round_results, labels, odds

    def run_monte_carlo_simulation(self, completed_matches, num_runs=1000):
        # Clear prediction cache to ensure correct recalibration with latest completed matches and gamma settings
        self.prediction_cache = {}
        
        # 1. Build lookup tables
        completed_group_lookup = {}
        completed_ko_lookup = {}
        
        for m in completed_matches:
            if m.get("stage") == "group_stage":
                t1, t2 = m["home_team"], m["away_team"]
                completed_group_lookup[(t1, t2)] = m
                completed_group_lookup[(t2, t1)] = m
            else:
                match_num = m.get("match_number")
                if match_num:
                    completed_ko_lookup[int(match_num)] = m
                    
        # 2. Get all teams
        all_teams = set()
        for grp in self.groups:
            for team in grp:
                all_teams.add(team)
                
        # Compute Dynamic Form Factors & Match Counts from completed matches
        raw_deltas = {}
        self.match_counts = {team: 0 for team in all_teams}
        
        for m in completed_matches:
            t1, t2 = m.get("home_team"), m.get("away_team")
            if not t1 or not t2 or t1 not in all_teams or t2 not in all_teams:
                continue
                
            h_score = int(m.get("home_score", 0))
            a_score = int(m.get("away_score", 0))
            winner = m.get("winner")
            
            # Predict baseline (uncapped, unadjusted raw ML probability)
            try:
                X = self.feature_generator_func(t1, t2)
                p_home = self.model.predict_proba([X])[0][1]
            except Exception:
                p_home = 0.5
                
            E_home = 2.0 * p_home + 1.0
            E_away = 3.0 - E_home
            
            if winner == t1 or h_score > a_score:
                A_home, A_away = 3.0, 0.0
            elif winner == t2 or a_score > h_score:
                A_home, A_away = 0.0, 3.0
            else:
                A_home, A_away = 1.0, 1.0
                
            raw_deltas.setdefault(t1, []).append(A_home - E_home)
            raw_deltas.setdefault(t2, []).append(A_away - E_away)
            
            self.match_counts[t1] += 1
            self.match_counts[t2] += 1
            
        self.avg_form = {}
        for team in all_teams:
            deltas = raw_deltas.get(team, [])
            self.avg_form[team] = float(np.mean(deltas)) if deltas else 0.0
            
        metrics = {team: {
            "group_qual": 0,
            "r32": 0,
            "r16": 0,
            "qf": 0,
            "sf": 0,
            "finalist": 0,
            "champion": 0
        } for team in all_teams}
        
        # Detailed tracking structures
        group_positions = {team: {1: 0, 2: 0, 3: 0, 4: 0} for team in all_teams}
        matchup_frequencies = {team: {stage: {} for stage in ['round_of_32', 'round_of_16', 'quarter_final', 'semi_final', 'final']} for team in all_teams}
        opponent_ranks_faced = {team: [] for team in all_teams}
        
        # Precompute team ranks to keep Monte Carlo lookup fast
        team_ranks = {}
        for team in all_teams:
            try:
                team_ranks[team] = self.feature_generator_func(team, "USA")[0]
            except Exception:
                team_ranks[team] = 50.0
                
        group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        self.recorded_matchups = []
        
        # Enable probabilistic mode
        self.probabilistic = True
        
        for _ in range(num_runs):
            self.recorded_matchups.clear()
            
            # Run one simulation iteration
            group_results, advancing_thirds_mapping = self.playGroupStage(
                start_date=self.tournamentStartDate, 
                completed_lookup=completed_group_lookup
            )
            
            # Record group finishes (1st, 2nd, 3rd, 4th)
            for grp_letter, res in group_results.items():
                w = res['winner']
                ru = res['runner_up']
                tp = res['third_place']
                
                grp_idx = group_letters.index(grp_letter)
                teams = self.groups[grp_idx]
                fp = [t for t in teams if t not in (w, ru, tp)][0]
                
                group_positions[w][1] += 1
                group_positions[ru][2] += 1
                group_positions[tp][3] += 1
                group_positions[fp][4] += 1
                
            # Record who qualified from group stage
            qualified_teams = set()
            for grp_letter, res in group_results.items():
                qualified_teams.add(res['winner'])
                qualified_teams.add(res['runner_up'])
            for third_team in advancing_thirds_mapping.values():
                qualified_teams.add(third_team)
                
            for team in qualified_teams:
                metrics[team]["group_qual"] += 1
                metrics[team]["r32"] += 1
                
            # Play Knockouts
            round_results, labels, odds = self.playKnockOuts(
                group_results=group_results,
                advancing_thirds_mapping=advancing_thirds_mapping,
                completed_lookup=completed_ko_lookup
            )
            
            # Record progressions
            for stage, winners in round_results.items():
                for team in winners:
                    if stage == 'round_of_16':
                        metrics[team]["r16"] += 1
                    elif stage == 'quarter_final':
                        metrics[team]["qf"] += 1
                    elif stage == 'semi_final':
                        metrics[team]["sf"] += 1
                    elif stage == 'final':
                        metrics[team]["finalist"] += 1
            
            # Champion
            champion = round_results['final'][0]
            metrics[champion]["champion"] += 1
            
            # Record matchup frequencies and opponent ranks
            for stage, t1, t2 in self.recorded_matchups:
                if stage in matchup_frequencies[t1]:
                    matchup_frequencies[t1][stage][t2] = matchup_frequencies[t1][stage].get(t2, 0) + 1
                if stage in matchup_frequencies[t2]:
                    matchup_frequencies[t2][stage][t1] = matchup_frequencies[t2][stage].get(t1, 0) + 1
                
                r1 = team_ranks.get(t1, 50.0)
                r2 = team_ranks.get(t2, 50.0)
                opponent_ranks_faced[t1].append(r2)
                opponent_ranks_faced[t2].append(r1)
            
        # Disable probabilistic mode after run is complete
        self.probabilistic = False
        if hasattr(self, 'recorded_matchups'):
            del self.recorded_matchups
        
        # 3. Calculate probabilities and package detailed stats
        probabilities = {}
        for team, counts in metrics.items():
            probabilities[team] = {k: v / num_runs for k, v in counts.items()}
            
            # Inject nested metadata for path explanation engine
            probabilities[team]["group_positions"] = {
                pos: group_positions[team][pos] / num_runs for pos in [1, 2, 3, 4]
            }
            probabilities[team]["matchup_frequencies"] = {
                stage: {
                    opp: count / num_runs for opp, count in stage_matchups.items()
                } for stage, stage_matchups in matchup_frequencies[team].items()
            }
            probabilities[team]["avg_opponent_rank"] = float(np.mean(opponent_ranks_faced[team])) if opponent_ranks_faced[team] else 50.0
            
        return probabilities

    def visualizeKnockOuts(self):
        model_name = ' '.join(model.capitalize() for model in self.model_name.split('_'))
        round_results, labels, odds = self.playKnockOuts()
        winner = round_results['final']
        node_sizes = pd.DataFrame(list(reversed(odds)))
        scale_factor = 0.3 
        
        G = nx.balanced_tree(2, 4) 
        pos = graphviz_layout(G, prog='twopi')
        centre = pd.DataFrame(pos).mean(axis=1).mean()

        plt.figure(figsize=(18, 18))
        ax = plt.subplot(1,1,1)
        
        circle_positions = [(290, 'grey'), (235, 'black'), (180, 'blue'), (120, 'red'), (60, 'yellow')]
        [ax.add_artist(plt.Circle((centre, centre), cp, color='grey', alpha=0.2)) for cp, c in circle_positions]

        nx.draw(G, pos, 
                node_color=node_sizes.diff(axis=1)[1].abs().pow(scale_factor), 
                alpha=1, cmap='Reds', edge_color='black', width=5, with_labels=False)

        shifted_pos = {k:[(v[0]-centre)*0.9+centre,(v[1]-centre)*0.9+centre] for k,v in pos.items()}
        nx.draw_networkx_labels(G, pos=shifted_pos, 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5, alpha=1), font_size=7,
                                labels=dict(zip(reversed(range(len(labels))), labels)))

        texts = ((10, 'Best 32', 'grey'), (70, 'Best 16', 'black'), (130, 'Quarter-\nfinal', 'blue'), (190, 'Semifinal', 'red'), (250, 'Final', 'yellow'))
        [plt.text(p, centre+20, t, fontsize=12, color='black', va='center', ha='center') for p,t,c in texts]
        
        plt.axis('equal')
        plt.title(f'2026 Format Predictions ({model_name})\nWinner= {winner[0]}', fontsize=18)
        # plt.show() # Prevent blocking if it runs
        print("Visualization complete!")