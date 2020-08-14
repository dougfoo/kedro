
"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import node, Pipeline
from kedro_tutorial.pipelines.data_engineering import pipeline as de
from kedro_tutorial.pipelines.data_science import pipeline as ds


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    de_pipeline = de.create_pipeline()
    ds_pipeline = ds.create_pipeline()

    return {
        "de": de_pipeline,
        "ds": ds_pipeline,
        "__default__": de_pipeline + ds_pipeline,
    }

