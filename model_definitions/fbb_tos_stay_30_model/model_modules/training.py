import shutil
from pathlib import Path

from tmo import (
    tmo_create_context,
    ModelContext
)


def train(context: ModelContext, **kwargs):
    tmo_create_context()

    print("Starting training...")

    print("Moving model artifacts to artifact_output_path...")

    source_path = Path("./model_modules")
    destination_path = Path(context.artifact_output_path)

    files = ['cascade_config.json', 'stage1_model.joblib', 'stage2_model.joblib', 
             'tos_baseline_model.joblib', 'tos_pipeline_artifacts.pkl', 'scoring_config.json']

    for name in files:
        src_path = source_path / name
        if src_path.is_file():
            shutil.move(str(src_path), destination_path / name)
            print(f"Moved: {src_path} to {destination_path / name}")
        else:
            print(f"Skipped (not found): {src_path}")

    print("Finished training")