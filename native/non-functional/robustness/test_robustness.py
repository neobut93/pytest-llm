from llamatestclient import llm_client 

def test_input_variations(llm_client):
    variations = ["what is the capital of France?",
                  "What's the capital of france?",
                  "capital of France?"]
    
    for prompt in variations:
        response = llm_client.generate(prompt)
        contains_paris = "paris" in response['text'].lower() or "Paris" in response['text']

        assert contains_paris, f"Expected 'Paris' in response for prompt '{prompt}', got: {response['text']}"
        print(f"Input variation test passed for prompt: '{prompt}'")