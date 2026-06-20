import os
import pickle
import json

def convert():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_dir, "..", "data", "logistic_regression_model.pkl")
    json_path = os.path.join(base_dir, "..", "data", "logistic_regression_model.json")
    
    if not os.path.exists(pkl_path):
        print(f"Error: Pickle file {pkl_path} not found.")
        return
        
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)
        
    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']
    
    model_data = {
        "scaler": {
            "mean": list(scaler.mean_),
            "scale": list(scaler.scale_),
            "var": list(scaler.var_),
            "n_samples_seen": int(scaler.n_samples_seen_)
        },
        "classifier": {
            "coef": list(classifier.coef_[0]),
            "intercept": float(classifier.intercept_[0])
        },
        "features": [
            "home_rank", "away_rank", "rank_diff",
            "home_weighted_wins", "away_weighted_wins",
            "home_weighted_goals", "away_weighted_goals"
        ]
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)
        
    print(f"Model successfully converted and saved to {json_path}")

if __name__ == "__main__":
    convert()
