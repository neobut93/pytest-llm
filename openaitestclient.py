import openai
import time

class OpenAITestClient:
    def __init__(self, model_name = "gpt-4o-mini", api_key=None):
        self.model_name = model_name
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # will use env.var by default
            self.client = openai.OpenAI() 

    def check_model_available(self) -> bool:
        try:
            self.client.chat.completions.create(
                model = self.model_name, 
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
                )
            return True
        except openai.error.OpenAIError as e:
            print(f"Model {self.model_name} not available: {e}")
            return False


    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 1000):
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )   
            end_time = time.time()  
        
            generated_text = response.choices[0].message.content

            return {
            "text": generated_text,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency": end_time - start_time,
            "model": self.model_name
            }
    
        except openai.error.OpenAIError as e:
            raise Exception(f"Open API error: {e}")