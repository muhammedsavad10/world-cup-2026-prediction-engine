import os
import json
from groq import Groq
from duckduckgo_search import DDGS

# Initialize the Groq client safely to handle missing key at start
client = None
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    client = Groq(api_key=api_key)

# Load team intelligence dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "data", "world_cup_2026_team_intelligence.json")

team_intelligence = {}
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for item in raw_data:
                team_intelligence[item["team"].lower().strip()] = item
    except Exception as e:
        print(f"Error loading team intelligence dataset: {e}")

def get_team_info(team_name):
    """
    Looks up a team in the local team intelligence dictionary,
    mapping abbreviations like 'USA' to 'United States'.
    """
    name_lookup = team_name.lower().strip()
    if name_lookup == "usa":
        name_lookup = "united states"
    return team_intelligence.get(name_lookup)

def fetch_live_team_news(team_name):
    """
    Scrapes the live web with tight chronological parameters to enforce 
    current 2026 manager and squad ground truths.
    """
    try:
        with DDGS() as ddgs:
            # Query heavily restricted to current active cycle to prevent legacy coach leaks
            query = f"{team_name} national football team current manager squad June 2026"
            search_results = list(ddgs.text(query, max_results=3))
            
            snippets = []
            for result in search_results:
                snippets.append(result.get('body', ''))
            
            return "\n".join(snippets)
    except Exception as e:
        return f"Live search data currently unavailable for {team_name}."

def format_team_context(info, team_name):
    """
    Formats the team intelligence dictionary into a concise context block.
    """
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

def generate_match_analysis(team_a, team_b, prob_a, prob_b, rank_diff, form_diff, goals_diff, current_phase="pre_tournament"):
    """
    Search-Augmented Agentic AI engineered to ignore data science jargon and justify prediction outcomes.
    """
    global client
    if not client:
        # Try initializing client again in case the key was configured late
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            client = Groq(api_key=api_key)
        else:
            return "Tactical analysis synthesis interrupted. Error: GROQ_API_KEY environment variable is not set."

    # Look up teams in dataset
    info_a = get_team_info(team_a)
    info_b = get_team_info(team_b)

    # Restrict live search usage: only search when the team is missing from the JSON dataset
    live_news_a = ""
    if not info_a:
        live_news_a = fetch_live_team_news(team_a)
        
    live_news_b = ""
    if not info_b:
        live_news_b = fetch_live_team_news(team_b)
    
    # Format local dataset info context
    team_a_context = format_team_context(info_a, team_a)
    team_b_context = format_team_context(info_b, team_b)

    # Force clean, explicit 2-decimal rounding before feeding the prompt context
    rounded_prob_a = round(float(prob_a), 2)
    rounded_prob_b = round(float(prob_b), 2)
    rounded_rank = round(float(rank_diff), 2)
    rounded_form = round(float(form_diff), 2)
    rounded_goals = round(float(goals_diff), 2)
    
    # Determine the predicted outcome based on model win probabilities
    if rounded_prob_a > rounded_prob_b:
        prediction_headline = f"{team_a} defeats {team_b}"
    elif rounded_prob_b > rounded_prob_a:
        prediction_headline = f"{team_b} defeats {team_a}"
    else:
        prediction_headline = f"{team_a} and {team_b} draw"

    math_context = f"""
    PREDICTION AND MODEL PERFORMANCE METRICS:
    - Prediction: {prediction_headline}
    - {team_a} Win Probability: {rounded_prob_a}%
    - {team_b} Win Probability: {rounded_prob_b}%
    - FIFA Rank Gap: {rounded_rank}
    - Form Advantage: {rounded_form}
    - Goal Efficiency Lead: {rounded_goals}
    """
    
    dataset_context = f"""
    AUTHORITATIVE DATASET INFO:
    Team A ({team_a}) Context:
    {team_a_context}
    
    Team B ({team_b}) Context:
    {team_b_context}
    """

    scraped_web_context = ""
    if live_news_a or live_news_b:
        scraped_web_context = f"""
        LIVE SEARCH DATA (Used ONLY because team(s) were missing from the authoritative JSON dataset):
        Team A ({team_a}) Live News:
        {live_news_a}
        
        Team B ({team_b}) Live News:
        {live_news_b}
        """

    system_prompt = (
        "You are an elite professional sports football analyst and presenter writing for publications like FIFA, ESPN, Opta, and The Athletic.\n"
        "Your task is to analyze and write a realistic football justification for a pre-calculated prediction model output using the team intelligence dataset. You must strictly follow these rules:\n\n"
        "1. NO PREDICTION DRIFT: Do NOT modify, alter, override, challenge, recalculate, or replace the pre-calculated prediction. Do not suggest other winners or tie scores. Your sole job is to explain why this outcome is plausible.\n"
        "2. EXPLANATION-ONLY BEHAVIOR: Explain why the prediction makes football sense. Never use phrases like 'I would predict...', 'The model could be wrong...', or 'A more likely outcome...'. Treat the prediction outcome as final.\n"
        "3. AUTHORITATIVE DATA ONLY: Rely on the provided AUTHORITATIVE DATASET INFO. Reference actual coaches, captains, players, form, injuries, and tactics as stated. Do NOT invent players, statistics, or facts not present in the provided dataset.\n"
        "4. NO DATA SCIENCE JARGON: Do NOT use terms like 'delta', 'coefficient', 'variance', 'matrix', 'regressor', or 'intercept'. Use clean football terms like 'advantage', 'gap', 'differential', or 'lead'.\n"
        "5. DECIMAL CAP: Any statistics, probabilities, or metrics you quote must be strictly formatted to a maximum of 2 decimal places.\n"
        "6. REQUIRED SECTIONS: Your output MUST contain exactly the following sections in Markdown, with no other text before or after:\n\n"
        "### Prediction\n"
        "[Display the supplied prediction headline exactly as received, along with win probabilities]\n\n"
        "### Explanation\n"
        "[Write 2 to 6 paragraphs of professional football reasoning justifying the prediction. Reference squad strength, coach impacts, captain and leadership, and tactical matchups.]\n\n"
        "### Key Factors\n"
        "* [Factor 1]\n"
        "* [Factor 2]\n"
        "* [Factor 3]\n"
        "* [Factor 4]\n\n"
        "### Players Likely To Influence The Outcome\n"
        "[List relevant players from the dataset with their positions, clubs, or statistics to justify how they will affect the outcome.]\n\n"
        "### Tactical Story\n"
        "[Describe a realistic, coherent narrative of how the match is likely to unfold based on the pre-calculated prediction.]"
    )

    try:
        user_prompt_content = f"Matchup: {team_a} vs {team_b}\n\n{math_context}\n\n{dataset_context}\n\n{scraped_web_context}"
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3, # Low temperature minimizes creative fabrications
            max_tokens=800,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Tactical analysis synthesis interrupted. Error: {str(e)}"
