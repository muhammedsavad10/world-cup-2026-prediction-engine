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

    def predictWinner(self, team1, team2, date, stage):
        X = self.feature_generator_func(team1, team2)
        prob = self.model.predict_proba([X])[0]
        home_win_prob = prob[1]
        if home_win_prob > 0.5:
            return team1, home_win_prob
        else:
            return team2, 1 - home_win_prob

    def playGroupStage(self, start_date):
        num_grps = 12
        group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        group_results = {}
        third_place_candidates = []
        
        print(f"Groups for this tournament:\n{self.groups}\n\n")
        print('Group Stage:')
        
        for grp in range(num_grps):
            g_letter = group_letters[grp]
            teams = self.groups[grp]
            team_stats = {team: {'wins': 0, 'prob_score': 0.0} for team in teams}
            
            for i in range(0, len(teams)):
                for j in range(i + 1, len(teams)):
                    dateOfMatch = start_date + datetime.timedelta(days=i * j)
                    winner, win_prob = self.predictWinner(teams[i], teams[j], dateOfMatch, 'group_stage')
                    
                    team_stats[winner]['wins'] += 1
                    team_stats[winner]['prob_score'] += win_prob
                    
                    loser = teams[j] if winner == teams[i] else teams[i]
                    team_stats[loser]['prob_score'] += (1 - win_prob)

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
        
        print(f"\nTop 8 Third-Place Teams Advancing: {[t['team'] for t in best_thirds]} from groups {qualified_third_groups}\n")
        
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

    def playKnockOuts(self):
        group_results, advancing_thirds_mapping = self.playGroupStage(start_date=self.tournamentStartDate)
        
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
                
            winner, win_prob = self.predictWinner(team1, team2, dateOfMatch, stage)
            loser = team2 if winner == team1 else team1
            
            match_winners[match_id] = winner
            match_losers[match_id] = loser
            
            round_results[stage].append(winner)
            
            t1_prob = win_prob if winner == team1 else 1 - win_prob
            t2_prob = 1 - t1_prob
            
            labels.append(f"{team1}({np.round(t1_prob,2)}) vs. {team2}({np.round(t2_prob,2)})")
            odds.append([t1_prob, t2_prob])
            
        print(f"\nKnockout winners:")
        for stg, wins in round_results.items():
            print(f"{stg}: {wins}")
            
        return round_results, labels, odds

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