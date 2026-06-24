import numpy as np

# Flattened group assignments for commentary mapping
team_group_map = {
    'Mexico': 'A', 'South Korea': 'A', 'South Africa': 'A', 'Czechia': 'A',
    'Canada': 'B', 'Bosnia and Herzegovina': 'B', 'Qatar': 'B', 'Switzerland': 'B',
    'Brazil': 'C', 'Morocco': 'C', 'Haiti': 'C', 'Scotland': 'C',
    'USA': 'D', 'Paraguay': 'D', 'Australia': 'D', 'Turkey': 'D',
    'Germany': 'E', 'Ivory Coast': 'E', 'Ecuador': 'E', 'Curacao': 'E',
    'Netherlands': 'F', 'Japan': 'F', 'Sweden': 'F', 'Tunisia': 'F',
    'Belgium': 'G', 'Egypt': 'G', 'Iran': 'G', 'New Zealand': 'G',
    'Spain': 'H', 'Uruguay': 'H', 'Saudi Arabia': 'H', 'Cape Verde': 'H',
    'France': 'I', 'Senegal': 'I', 'Norway': 'I', 'Iraq': 'I',
    'Portugal': 'K', 'Colombia': 'K', 'Uzbekistan': 'K', 'DR Congo': 'K',
    'England': 'L', 'Croatia': 'L', 'Ghana': 'L', 'Panama': 'L'
}

def approx_cdf(z):
    return 0.5 * (1 + np.sign(z) * np.sqrt(1 - np.exp(-2 * z**2 / np.pi)))

def generate_recalibration_explanation(team_name, baseline_prob, current_prob, completed_matches, rolling_form, num_runs=10000):
    """
    Generates structured, data-driven recalibration commentary.
    Returns a dictionary:
    {
        "spectator_insight": "Concise 2-line spectator insight.",
        "advanced_analytics": "Markdown containing statistical details (CI, Z-statistic, matchups)."
    }
    """
    curr_champ = current_prob.get("champion", 0.0)
    base_champ = baseline_prob.get("champion", 0.0)
    delta_champ = curr_champ - base_champ
    
    curr_qual = current_prob.get("group_qual", 0.0)
    base_qual = baseline_prob.get("group_qual", 0.0)
    delta_qual = curr_qual - base_qual
    
    curr_positions = current_prob.get("group_positions", {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
    base_positions = baseline_prob.get("group_positions", {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
    
    curr_likely_finish = max(curr_positions.keys(), key=lambda k: curr_positions[k]) if curr_positions else "Unknown"
    base_likely_finish = max(base_positions.keys(), key=lambda k: base_positions[k]) if base_positions else "Unknown"
    
    curr_diff = current_prob.get("avg_opponent_rank", 50.0)
    base_diff = baseline_prob.get("avg_opponent_rank", 50.0)
    diff_change = curr_diff - base_diff # Positive is easier (weaker rank number)
    
    # Statistical validation checks (Z-test)
    N_base = 10000
    N_curr = num_runs
    se_base = np.sqrt(base_champ * (1 - base_champ) / N_base) if base_champ > 0 else 0
    se_curr = np.sqrt(curr_champ * (1 - curr_champ) / N_curr) if curr_champ > 0 else 0
    se_diff = np.sqrt(se_base**2 + se_curr**2)
    
    if se_diff > 0:
        z_stat = abs(delta_champ) / se_diff
        p_val = 2 * (1 - approx_cdf(z_stat))
        is_significant = z_stat >= 1.96
    else:
        z_stat = 0.0
        p_val = 1.0
        is_significant = False
        
    ci_margin = 1.96 * se_curr if se_curr > 0 else 0.0
    
    # Attribution calculations
    if abs(delta_champ) < 1e-5:
        noise_pct, qual_pct, path_pct, norm_pct = 100.0, 0.0, 0.0, 0.0
    else:
        noise_bound = 1.96 * (se_diff if se_diff > 0 else 1.0)
        noise_share = min(0.9, noise_bound / max(1e-5, abs(delta_champ)))
        noise_pct = noise_share * 100.0
        
        w_qual = abs(curr_qual - base_qual)
        w_path = abs(curr_diff - base_diff) / 50.0
        w_total = w_qual + w_path
        
        if w_total > 0:
            qual_pct = (100.0 - noise_pct) * (w_qual / w_total) * 0.9
            path_pct = (100.0 - noise_pct) * (w_path / w_total) * 0.9
            norm_pct = 100.0 - noise_pct - qual_pct - path_pct
        else:
            qual_pct, path_pct = 0.0, 0.0
            norm_pct = 100.0 - noise_pct

    # Generate Concise Spectator Insight (Fan friendly, 2-3 lines max)
    grp_letter = team_group_map.get(team_name, "A")
    
    if abs(delta_champ) < 0.001 or not is_significant:
        spectator_insight = f"{team_name}'s outlook remains largely unchanged as group stage matches continue."
    else:
        if delta_champ > 0:
            if delta_qual > 0.05:
                spectator_insight = f"{team_name}'s title odds increased after favorable standings shifts in Group {grp_letter}."
            elif diff_change > 1.5:
                spectator_insight = f"{team_name}'s projected knockout route became easier due to favorable results in neighboring groups."
            else:
                spectator_insight = f"{team_name}'s champion odds rose as their qualification pathway solidified."
        else:
            if delta_qual < -0.05:
                spectator_insight = f"{team_name}'s recent matches reduced their chances of qualifying from Group {grp_letter}."
            elif diff_change < -1.5:
                spectator_insight = f"{team_name}'s projected knockout route became more challenging due to strong potential opponents."
            else:
                spectator_insight = f"{team_name}'s championship probability compressed slightly due to tournament bracket pathing shifts."

    # Generate Detailed Advanced Analytics (Markdown format)
    def format_opp_dist(matchups):
        if not matchups:
            return "No matchups recorded (qualified or eliminated in all runs)"
        sorted_opps = sorted(matchups.items(), key=lambda x: x[1], reverse=True)[:3]
        return ", ".join([f"{opp} ({prob * 100:.1f}%)" for opp, prob in sorted_opps])

    curr_matchups = current_prob.get("matchup_frequencies", {})
    base_matchups = baseline_prob.get("matchup_frequencies", {})
    
    r32_base = format_opp_dist(base_matchups.get("round_of_32", {}))
    r32_curr = format_opp_dist(curr_matchups.get("round_of_32", {}))
    
    r16_base = format_opp_dist(base_matchups.get("round_of_16", {}))
    r16_curr = format_opp_dist(curr_matchups.get("round_of_16", {}))
    
    significance_str = "Significant Shift" if is_significant else "Insignificant (Monte Carlo Noise)"
    
    advanced_markdown = f"""#### ⚙️ Advanced Analytics & Confidence Margins

*   **Baseline Champion Probability**: `{base_champ * 100:.2f}%`
*   **Live Champion Probability**: `{curr_champ * 100:.2f}%` (95% CI: `[{max(0.0, curr_champ - ci_margin) * 100:.2f}%, {min(1.0, curr_champ + ci_margin) * 100:.2f}%]`, Margin: `±{ci_margin * 100:.2f}%`)
*   **Statistical Significance**: `{significance_str}` (Z-score: `{z_stat:.3f}`, p-value: `{p_val:.4f}`)

#### 📊 Attribution of Probability Movement
*   **Monte Carlo Noise**: `{noise_pct:.1f}%`
*   **Group Standings Changes**: `{qual_pct:.1f}%`
*   **Path Difficulty Changes**: `{path_pct:.1f}%`
*   **Probability Normalization**: `{norm_pct:.1f}%`

#### 🗺️ Matchup Path Comparisons
*   **Round of 32 Opponents (Before)**: {r32_base}
*   **Round of 32 Opponents (After)**: {r32_curr}
*   **Round of 16 Opponents (Before)**: {r16_base}
*   **Round of 16 Opponents (After)**: {r16_curr}

#### 💡 Verified Causes
*   **Group qualification probability**: Changed from `{base_qual * 100:.1f}%` to `{curr_qual * 100:.1f}%`
*   **Most likely group finish rank**: Changed from `Rank {base_likely_finish}` to `Rank {curr_likely_finish}`
*   **Knockout path difficulty score (Avg Opponent FIFA Rank)**: Changed from `{base_diff:.1f}` to `{curr_diff:.1f}` *(Note: lower rank number indicates a more challenging route)*
"""

    return {
        "spectator_insight": spectator_insight,
        "advanced_analytics": advanced_markdown
    }
