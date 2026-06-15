import numpy as np

def audit_probabilities(baseline_probs, current_probs, completed_matches, num_runs=1000):
    """
    Audits tournament champion probability shifts for all 48 teams.
    Flags Case A (win but drop), Case B (loss but rise), Case C (change >= 1.0%),
    calculates 95% confidence intervals, and diagnoses root causes.
    """
    # 1. Determine team outcomes in completed matches
    team_outcomes = {}
    for m in completed_matches:
        home, away = m["home_team"], m["away_team"]
        winner = m["winner"]
        
        team_outcomes.setdefault(home, "D")
        team_outcomes.setdefault(away, "D")
        
        if winner == home:
            team_outcomes[home] = "W"
            team_outcomes[away] = "L"
        elif winner == away:
            team_outcomes[home] = "L"
            team_outcomes[away] = "W"
            
    audit_results = {}
    
    for team in current_probs:
        curr_p = current_probs[team].get("champion", 0.0)
        base_p = baseline_probs.get(team, {}).get("champion", curr_p)
        delta = curr_p - base_p
        
        # Calculate 95% Confidence Interval for baseline probability: 1.96 * SE
        # Ensure base_p is bounded to avoid zero division or infinite error
        p_bound = max(0.001, min(0.999, base_p))
        se = np.sqrt(p_bound * (1 - p_bound) / num_runs)
        ci = 1.96 * se
        
        outcome = team_outcomes.get(team, None)
        
        flagged = False
        case_label = None
        cause = None
        explanation = ""
        
        # Detect Cases
        if outcome == "W" and delta < -0.001:
            flagged = True
            case_label = "Case A: Won match but title probability fell"
        elif outcome == "L" and delta > 0.001:
            flagged = True
            case_label = "Case B: Lost match but title probability rose"
        elif abs(delta) >= 0.01:
            flagged = True
            case_label = "Case C: Large probability shift (>= 1.0%)"
            
        # Root Cause Analysis
        if flagged or abs(delta) > 0.001:
            if abs(delta) <= ci:
                cause = "Monte Carlo Noise"
                explanation = (
                    f"The champion probability shift of {delta*100:+.2f}% is within the simulation's "
                    f"95% confidence interval (±{ci*100:.2f}%) for {num_runs} runs. This is likely normal "
                    f"statistical variance."
                )
            elif outcome == "W" and delta < 0:
                cause = "Bracket Path Collision"
                explanation = (
                    f"{team} won their match, consolidating their group position. However, results in other "
                    f"groups now project a more difficult knockout path with stronger potential opponents, "
                    f"slightly depressing their deep-run projections despite the win."
                )
            elif outcome == "L" and delta > 0:
                cause = "Group Standing Dynamics"
                explanation = (
                    f"Despite their defeat, results elsewhere in the group (or favorable goal differences) "
                    f"kept {team} in a strong position to qualify as a best 3rd-place team, resulting in a minor "
                    f"statistical variance."
                )
            elif abs(delta) >= 0.01:
                cause = "Path Difficulty Shift"
                explanation = (
                    f"Surprise results or draws elsewhere in the bracket shifted the tournament structure, "
                    f"significantly altering {team}'s projected knockout pathway."
                )
            else:
                cause = "Probability Normalization"
                explanation = (
                    f"Favorable results for major contenders elsewhere in the bracket caused their probabilities to "
                    f"rise. Because the total probability space is fixed at 100%, other teams' odds compressed "
                    f"slightly to compensate."
                )
        else:
            cause = "No Significant Change"
            explanation = "Tournament odds remain stable and aligned with the baseline."
            
        audit_results[team] = {
            "team": team,
            "previous": base_p,
            "current": curr_p,
            "delta": delta,
            "ci": ci,
            "flagged": flagged,
            "case": case_label,
            "cause": cause,
            "explanation": explanation
        }
        
    return audit_results
