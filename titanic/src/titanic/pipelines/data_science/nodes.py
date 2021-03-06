import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    features = data.drop(columns=['PassengerId','Survived'])
    X = features.values
    y = data["Survived"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return [X_train, X_test, y_train, y_test, features.columns] 

def train_lr(X: np.ndarray, y: np.ndarray, parameters: Dict, feature_names: List) -> LogisticRegression:
    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    return model

def evaluate_model(model, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)  
    score = r2_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model r2score ___{round(score,4)}___, accuracy ___{round(acc,4)}___")    

    from sklearn.metrics import roc_curve, auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    logger.info(f"ROC/AUC: {roc_auc}")


def output_guesses(model, data: pd.DataFrame) -> pd.DataFrame:
    y_pred = model.predict(data.iloc[:,1:])
    df = data.iloc[:,0:1].copy()
    df['Survived'] = pd.DataFrame(y_pred)
    df = df.astype({'Survived': 'int32'})
    return df

def train_xgb(X: np.ndarray, y: np.ndarray, parameters: Dict, feature_names: List)  -> GradientBoostingClassifier:
    paramDict = { 'n_estimators' : [1, 2, 4, 8, 16, 32, 64, 100, 200],
                  'max_depth' : [2, 3, 4, 5, 6, 10, 18, 30], 
                #   'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                #   'min_samples_split': [0.1, 0.3, 0.5, 0.7, 0.9],
                #   'min_samples_leaf': [0.1,0.2,0.3,0.4,0.5]
    }
    model = GradientBoostingClassifier()
    print(f'trying GridSearchCV ranges {paramDict}')
    clf = GridSearchCV(estimator=model, param_grid=paramDict, n_jobs=10, verbose=1)
    clf.fit(X, y)
    print(f'best params: {clf.best_params_}, best score:  {clf.best_score_}')

    model = GradientBoostingClassifier(**clf.best_params_)
    model.fit(X, y)

    ic = pd.DataFrame([model.feature_importances_], columns=feature_names).T
    ic.columns=['Rank']
    print(ic.sort_values(by='Rank', ascending=False))

    return model

def train_dt(X: np.ndarray, y: np.ndarray, parameters: Dict, feature_names: List)  -> DecisionTreeClassifier:
    paramDict = {
                "max_depth": range(1, 50),
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
     }
    
    model = DecisionTreeClassifier()
    print(f'trying GridSearchCV ranges {paramDict}')
    clf = GridSearchCV(estimator=model, param_grid=paramDict, n_jobs=10, verbose=1)
    clf.fit(X, y)
    print(f'best params: {clf.best_params_}, best score:  {clf.best_score_}')

    model = DecisionTreeClassifier(**clf.best_params_)
    model.fit(X, y)

    ic = pd.DataFrame([model.feature_importances_], columns=feature_names).T
    ic.columns=['Rank']
    print(ic.sort_values(by='Rank', ascending=False))

    return model

def train_rf(X: np.ndarray, y: np.ndarray, parameters: Dict, feature_names: List)  -> RandomForestClassifier:
    paramDict = { 'n_estimators' : [5, 7, 10,  15,  20, 100, 200, 400],
                  'max_depth' : [2,3,4, 5,6],   }
    model = RandomForestClassifier(n_jobs = 8)
    print(f'trying GridSearchCV ranges {paramDict}')
    clf = GridSearchCV(estimator=model, param_grid=paramDict, n_jobs=10)
    clf.fit(X, y)
    print(f'best params: {clf.best_params_}, best score:  {clf.best_score_}')

    model = RandomForestClassifier(**clf.best_params_)
    model.fit(X, y)

    ic = pd.DataFrame([model.feature_importances_], columns=feature_names).T
    ic.columns=['Rank']
    print(ic.sort_values(by='Rank', ascending=False))

    return model

def train_gridcv(X: np.ndarray, y: np.ndarray, parameters: Dict):
    paramDict = { 'n_estimators' : [5, 10, 25, 50, 75, 100, 200, 500] }
    model = AdaBoostClassifier()
    print(f'trying GridSearchCV ranges {paramDict}')
    clf = GridSearchCV(estimator=model, param_grid=paramDict, n_jobs=10)
    clf.fit(X, y)
    print(f'best params: {clf.best_params_}, best score:  {clf.best_score_}')

    model = AdaBoostClassifier(**clf.best_params_)
    model.fit(X, y)

    return model

