from kedro.pipeline import Pipeline, node

from titanic.pipelines.data_science.nodes import (
    split_data,
    split_data2,
    train_model,
    train_xgb,
    evaluate_model,
    output_guesses,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data, 
                inputs=["final_train_out","parameters"], 
                outputs=["X_train2", "X_test2", "y_train2", "y_test2"]
            ),
            # node(
            #     func=train_model, 
            #     inputs=["X_train","y_train","parameters"], 
            #     outputs="model"
            # ),
            node(
                func=train_xgb, 
                inputs=["X_train2","y_train2","parameters"], 
                outputs="model2"
            ),
            # node(
            #     func=evaluate_model,
            #     inputs=["model", "X_train", "y_train"],
            #     outputs=None,
            #     name="evaluate logreg"
            # ),
            node(
                func=evaluate_model,
                inputs=["model2", "X_train2", "y_train2"],
                outputs=None,
                name="evaluate xgb"
            ),
            node(
                func=output_guesses,
                inputs=["model2", "final_test_out"],
                outputs="final_submission",
            ),
            
        ]
    )
