from tmo import (
    tmo_create_context,
    ModelContext
)


def train(context: ModelContext, **kwargs):
    tmo_create_context()

    print("Starting training...")
    print("Finished training")