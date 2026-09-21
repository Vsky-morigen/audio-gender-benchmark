"""Template for integrating a local audio model with the benchmark runner."""


class LocalModelAdapter:
    model_id = "replace-with-model-name"

    def __init__(self, config):
        # Load the tokenizer, processor and model here once.
        self.config = config

    def predict(self, audio_path: str, prompt: str, item: dict[str, str]) -> str:
        # Run local inference here. Return "male" or "female".
        raise NotImplementedError("Implement local model inference in examples/local_model_adapter.py")


def create_backend(config):
    return LocalModelAdapter(config)
