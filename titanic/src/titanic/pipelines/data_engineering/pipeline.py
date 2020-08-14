from kedro.pipeline import node, Pipeline
from titanic.pipelines.data_engineering.nodes import (
    preprocess_train,
    preprocess_test,
    final_train,
    final_test,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_train,
                inputs="titanic_train",
                outputs="preprocessed_train",
                name="preprocess_train_name",  # also used by run by name ?
            ),
            node(
                func=preprocess_test,
                inputs="titanic_test",
                outputs="preprocessed_test",
                name="preprocess_test_name",  
            ),
            node(
                func=final_train,
                inputs=["preprocessed_train", "titanic_refdata"],
                outputs="final_train_out",
                name="final_train_name",
            ),
            node(
                func=final_test,
                inputs=["preprocessed_test", "titanic_refdata"],
                outputs="final_test_out",
                name="final_test_name",
            ),
        ]
    )
