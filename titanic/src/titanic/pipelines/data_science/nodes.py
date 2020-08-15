import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


def train_model(data: pd.DataFrame) -> LogisticRegression:
    """Train the model.
        Args:
            data: features + answer.
        Returns:  Trained model.
    """
    model = LogisticRegression(random_state=0)
    X = data.iloc[:,2:]
    y = data.iloc[:,1:2]
    model.fit(X, y)
    return model

def evaluate_model(model: LogisticRegression, data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the coefficient of determination and log the result.
        Args:
            regressor: Trained model.
            data: features
    """
    # y_pred = model.predict(data.iloc[:,1:])  # skip PassengerId
    # score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model eval skip for now")


def output_guesses(model: LogisticRegression, data: pd.DataFrame) -> pd.DataFrame:
    y_pred = model.predict(data.iloc[:,1:])
    df = data.iloc[:,0:1]
    df['Survived'] = pd.DataFrame(y_pred)
    df = df.astype({'Survived': 'int32'})
    return df
