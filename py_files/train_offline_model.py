import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocess import load_data
from logistic_regression_class import LogisticRegressionClass

def train_and_save():
    print("Loading data...")
    X, y, rankings_df = load_data()
    print("Training Logistic Regression Model...")
    model_class = LogisticRegressionClass()
    base_model = model_class.get_model()
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_model)
    ])
    pipeline.fit(X, y)
    
    # Apply 60% Calibration to Historical Goals Coefficients (index 5 & 6)
    pipeline.named_steps['classifier'].coef_[0][5] *= 0.60
    pipeline.named_steps['classifier'].coef_[0][6] *= 0.60
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "data", "logistic_regression_model.pkl")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_and_save()
