import argparse
import datetime
import warnings
warnings.filterwarnings('ignore')

from const import models, WCGroups
from preprocess import load_data, get_match_features
from tournament_simulator import TournamentSimulator
from logistic_regression_class import LogisticRegressionClass
from sklearn.preprocessing import StandardScaler

def main():
    print("Loading datasets and merging FIFA rankings...")
    X, y, rankings_df = load_data()
    
    from sklearn.pipeline import Pipeline
    
    print("Training Logistic Regression Pipeline...")
    model_class = LogisticRegressionClass()
    base_model = model_class.get_model()
    
    # Wrap model in Pipeline to explicitly scale X_train and X_test
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_model)
    ])
    pipeline.fit(X, y)
    
    # Feature generator func for the simulator
    def feature_generator(team1, team2):
        features = get_match_features(team1, team2, rankings_df)
        return features # No manual scaling needed, pipeline handles it
    
    print("Starting Tournament Simulation...")
    simulator = TournamentSimulator(datetime.date(2026, 6, 11), pipeline, WCGroups, feature_generator)
    simulator.playKnockOuts()

if __name__ == '__main__':
    main()
