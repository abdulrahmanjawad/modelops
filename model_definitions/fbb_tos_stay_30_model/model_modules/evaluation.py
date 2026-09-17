from tmo import (
    tmo_create_context,
    ModelContext
)


def evaluate(context: ModelContext, **kwargs):
    tmo_create_context()

    print("Starting evaluation...")
    print("Finished evaluation")