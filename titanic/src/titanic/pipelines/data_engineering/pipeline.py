from kedro.pipeline import node, Pipeline
from titanic.pipelines.data_engineering.nodes import (
    preprocess,
    final_train,
    final_test,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess,
                inputs="titanic_train",
                outputs=["preprocessed_train","scaler"],
                name="preprocess_train_name",  # also used by run by name ?
            ),
            node(
                func=preprocess,
                inputs="titanic_test",
                outputs=["preprocessed_test","scaler2"],
                name="preprocess_test_name",  
            ),
            node(
                func=final_train,
                inputs=["titanic_train", "preprocessed_train", "titanic_refdata"],
                outputs="final_train_out",
                name="final_train_name",
            ),
            node(
                func=final_test,
                inputs=["titanic_test", "preprocessed_test", "titanic_refdata"],
                outputs="final_test_out",
                name="final_test_name",
            ),
        ]
    )
