from kedro.pipeline import Pipeline, node

from titanic.pipelines.data_science.nodes import (
    evaluate_model,
    train_model,
    output_guesses,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model, 
                inputs=["final_train_out"], 
                outputs="model"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "final_test_out"],
                outputs=None,
            ),
            node(
                func=output_guesses,
                inputs=["model", "final_test_out"],
                outputs="final_submission",
            ),
            
        ]
    )
