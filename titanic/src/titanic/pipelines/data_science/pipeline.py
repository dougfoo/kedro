from kedro.pipeline import Pipeline, node

from titanic.pipelines.data_science.nodes import (
    split_data,
    train_model,
    evaluate_model,
    output_guesses,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data, 
                inputs=["final_train_out","parameters"], 
                outputs=["X_train", "X_test", "y_train", "y_test"]
            ),
            node(
                func=train_model, 
                inputs=["X_train","y_train"], 
                outputs="model"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_train", "y_train"],
                outputs=None,
            ),
            node(
                func=output_guesses,
                inputs=["model", "final_test_out"],
                outputs="final_submission",
            ),
            
        ]
    )
