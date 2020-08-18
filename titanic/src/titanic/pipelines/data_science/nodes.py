import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    X = data[["Pclass","Name","Age","SibSp","Parch","Fare","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S","Cabin"]].values
    y = data["Survived"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]


def split_data2(data: pd.DataFrame, parameters: Dict) -> List:
    X = data[["Pclass","Name","Age","SibSp","Parch","Fare","Sex","Embarked"]].values
    y = data["Survived"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]    

def train_model(X: np.ndarray, y: np.ndarray, parameters: Dict) -> LogisticRegression:
    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    return model

def evaluate_model(model, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)  # skip PassengerId
    score = r2_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model r2score ___{round(score,4)}___, accuracy ___{round(acc,4)}___")

def output_guesses(model: LogisticRegression, data: pd.DataFrame) -> pd.DataFrame:
    y_pred = model.predict(data.iloc[:,1:])
    df = data.iloc[:,0:1].copy()
    df['Survived'] = pd.DataFrame(y_pred)
    df = df.astype({'Survived': 'int32'})
    return df

def train_xgb(X: np.ndarray, y: np.ndarray, parameters: Dict)  -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(X, y)
    return model
