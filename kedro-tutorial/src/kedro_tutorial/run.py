"""Application entry point."""
from pathlib import Path
from typing import Dict

from kedro_tutorial.pipeline import create_pipelines
from kedro.framework.context import KedroContext, load_package_context
from kedro.pipeline import Pipeline

class ProjectContext(KedroContext):
    """Users can override the remaining methods from the parent class here,
    or create new ones (e.g. as required by plugins)
    """

    project_name = "Kedro Tutorial"
    # `project_version` is the version of kedro used to generate the project
    project_version = "0.16.1"
    package_name = "kedro_tutorial"

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return create_pipelines()


def run_package():
    # Entry point for running a Kedro project packaged with `kedro package`
    # using `python -m <project_package>.run` command.
    project_context = load_package_context(
        project_path=Path.cwd(),
        package_name=Path(__file__).resolve().parent.name
    )
    project_context.run()


if __name__ == "__main__":
    run_package()
