from kedro.pipeline import Pipeline, node
from sklearn.ensemble import GradientBoostingClassifier

from titanic.pipelines.data_science.nodes import (
    split_data,
    train_lr,
    train_xgb,
    train_dt,
    train_rf,
    train_gridcv,
    evaluate_model,
    output_guesses,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data, 
                inputs=["final_train_out","parameters"], 
                outputs=["X_train", "X_test", "y_train", "y_test", "features_names"]
            ),
            node(
                func=train_dt, 
                inputs=["X_train","y_train","parameters","features_names"], 
                outputs="model"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_train", "y_train"],
                outputs=None,
                name="evaluate xgb"
            ),
            node(
                func=output_guesses,
                inputs=["model", "final_test_out"],
                outputs="final_submission",
            ),
            
        ]
    )
