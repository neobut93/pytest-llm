import time

import pytest
import requests

class OllamaTestClient:
    def __init__(self, model="tinyllama", host="localhost", port=11434):
        self.model = model
        self.host = host
        self.endpoint = f"http://{host}:{port}/api/generate"

    def check_model_available(self):
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": "test",
                "stream": False
            })
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt, temperature=0.8, max_tokens=1000):
        # todo double check with claude on latency
        start_time = time.time()
        response = requests.post(self.endpoint, json={
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        })
        end_time = time.time()
        if response.status_code != 200:
            raise RuntimeError(f"Request failed with status code {response.status_code}")
        result = response.json()
        return {
            "text": result["response"],
            # todo poor way, tokens != words,use either response or library to calculate it correctly
            # ex: from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained("model_name")
            # prompt_tokens = len(tokenizer.encode(prompt))
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(result["response"].split()),
            "latency": end_time - start_time,
        }

@pytest.fixture
def llm_client():
    client = OllamaTestClient(model = "tinyllama")
    if not client.check_model_available():
        pytest.skip("Model unavailable, please run 'Ollama pull tinyllama'")

    return client