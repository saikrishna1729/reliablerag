import requests
from typing import Any
from config import ExperimentConfig

class OllamaNotAvailableError(Exception):
    pass

class MockGenerator:
    """Mock generator that returns a placeholder response."""
    def generate(self, prompt: str, context: str) -> str:
        return f"MOCK RESPONSE: placeholder for pipeline testing."


class OllamaGenerator:
    """Ollama generator that uses local Ollama REST API."""
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate(self, prompt: str, context: str = "") -> str:
        if context:
            formatted_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            formatted_prompt = prompt
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.exceptions.ConnectionError:
            raise OllamaNotAvailableError(
                f"Ollama server is not running at {self.host}. Please start it and pull '{self.model_name}'"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


class HuggingFaceGenerator:
    """Hugging Face local generator using the transformers library."""
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None

    def _init_pipeline(self):
        if self.pipeline is None:
            import torch
            from transformers import pipeline
            
            if self.device == "auto":
                device_idx = 0 if torch.cuda.is_available() else -1
            else:
                device_idx = 0 if self.device == "cuda" else -1
                
            # Load text2text-generation pipeline (suitable for flan-t5)
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=device_idx,
                max_length=512
            )

    def generate(self, prompt: str, context: str = "") -> str:
        self._init_pipeline()
        if context:
            input_text = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            input_text = prompt
        outputs = self.pipeline(input_text)
        return outputs[0]["generated_text"].strip()


def get_generator(config: ExperimentConfig):
    if config.generator == "mock":
        return MockGenerator()
    elif config.generator == "ollama":
        return OllamaGenerator("gemma3:4b-it-q4_K_M")
    elif config.generator == "hf-small":
        return HuggingFaceGenerator("google/flan-t5-base")
    elif config.generator == "hf-large":
        return HuggingFaceGenerator("google/flan-t5-xl")
    else:
        raise ValueError(f"Unknown generator: {config.generator}")
