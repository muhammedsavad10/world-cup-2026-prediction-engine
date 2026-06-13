from sklearn.preprocessing import StandardScaler

from model.adaboost import AdaptiveBoostingClassifier
from model.decision_tree import DecisionTree
from model.ensemble_classifier import EnsembleClassifier
from model.gradient_boost import GradientBoost
from model.knn import KNearestNeighbours
from model.naive_bayes import NaiveBayesClassifier
from model.random_forest import RandomForest
from model.pca import PCATransform
from const import project_dir_path, data_dir_path, individual_window_size, head_to_head_window_size, last_n_data, \
    pca_n_components
from feature.data_split import split_feature_and_labels, get_train_test_split
from model.semi_supervised import SemiSupervisedClassifier
from model.svm import SVM
from model.logistic_regression_class import LogisticRegressionClass
from utils.classification_stats import ClassificationStatistics
from feature.feature_selector import FeatureSelector

import pandas as pd
import os.path as osp
import numpy as np

from utils.preprocess import generate_features, generate_artificial_matches


def train(model_name, do_pca=False, tune=False, select_features=True, semi_supervised=False):
    model = None
    X, y = load_data(artificial=False)
    print('Data loaded')
    if model_name == 'random_forest':
        model = RandomForest()
    elif model_name == 'gradient_boost':
        model = GradientBoost()
    elif model_name == 'support_vector_machine':
        model = SVM()
    elif model_name == 'decision_tree':
        model = DecisionTree()
    elif model_name == 'logistic_regression':
        model = LogisticRegressionClass()
    elif model_name == 'k_nearest_neighbours':
        model = KNearestNeighbours()
    elif model_name == 'naive_bayes_classifier':
        model = NaiveBayesClassifier()
    elif 'adaptive_boost_' in model_name:
        model = AdaptiveBoostingClassifier(model_name=model_name)
    elif model_name == 'ensemble_classifier':
        model = EnsembleClassifier()
    if select_features:
        print('Selecting features')
        feature_selector = FeatureSelector(model, X, y)
        selected_features = feature_selector.select_features()
    if do_pca:
        print('Performing PCA')
        pca = PCATransform(n_components=pca_n_components)
        pca.fit(X)
        pca_X = pca.transform(X)
    X = X[selected_features] if select_features else X
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    x_train, x_test, y_train, y_test = get_train_test_split(pca_X if do_pca else X, y)
    if model_name == 'ensemble_classifier':
        print('Ensemble classifier already loads tuned individual models')
    elif tune:
        model.tune(x_train, y_train)
    else:
        hyperparameters = load_model(model_name)
        if hyperparameters is not None:
            model.initialize_model_hyperparameters(**hyperparameters)
        else:
            model.tune(x_train, y_train)
    train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test,do_pca, select_features)
    if semi_supervised:
        model_name = model_name + '_semi_supervised'
        model = SemiSupervisedClassifier(base_model=model, model_name=model_name, threshold=0.75)
        artificial_X, artificial_y = load_data(artificial=True)
        scalar = StandardScaler()
        scalar.fit(artificial_X)
        artificial_X = scalar.transform(artificial_X)
        x_train = np.concatenate((x_train, artificial_X))
        artificial_y.loc[:] = -1
        y_train = pd.concat([y_train, artificial_y], ignore_index=True)
        train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test,do_pca, select_features
                           , semi_supervised=True)


def train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test,do_pca, select_features,
                       semi_supervised=False):
    if model_name != 'ensemble_classifier':
        print('Training model: ' + model_name)
        model.fit(x_train, y_train)
        print('Model trained')
    print('Evaluating Model')
    y_hat_test = model.predict(x_test)
    y_hat_test_proba = model.predict_proba(x_test)
    if semi_supervised:
        y_train = model.get_self_training_model().transduction_
    evaluator = ClassificationStatistics(model, model_name, x_train, y_train, x_test, y_test, y_hat_test,
                                         y_hat_test_proba, do_pca, select_features)
    model.save_model()
    evaluator.evaluate_model(extract_learning_curve=not semi_supervised)


def load_model(model_name):
    try:
        hyperparameters = pd.read_pickle(project_dir_path + '/model_hyperparameters/' + model_name + '.pkl')
        return hyperparameters
    except FileNotFoundError:
        print('Hyperparameters for the model not found. Will need to tune the model.')
    return None


def load_data(artificial):
    print('Loading data')
    prefix = 'artificial_features_last_' if artificial else 'features_last_'
    try:
        features = pd.read_csv(osp.join(data_dir_path, prefix + str(last_n_data) + '_windows_' + str(
            individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'))
    except FileNotFoundError:
        print('Features file not found.')
        if not artificial:
            generate_features()
        else:
            generate_artificial_matches()
        features = pd.read_csv(osp.join(data_dir_path, prefix + str(last_n_data) + '_windows_' + str(
            individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'))
    X, y = split_feature_and_labels(features)
    y = y.iloc[:, :1].squeeze()
    return X, y
