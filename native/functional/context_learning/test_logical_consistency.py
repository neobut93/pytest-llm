from llamatestclient import llm_client

def test_logical_consistency(llm_client):
    equialent_queations = [
        ("What is the capital of Japan?", "Tokyo is the capital of which country?")]
    
    for q1, q2 in equialent_queations:
        response1 = llm_client.generate(q1)
        response2 = llm_client.generate(q2)

        # [:100] limit response
        answer1 = response1['text'].lower()[:100]
        answer2 = response2['text'].lower()[:100]

        words1 = set(answer1.split())
        print(words1)
        words2 = set(answer2.split())
        print(words2)

        if words1 and words2:
            #create dict of same words
            overlap = len(words1.intersection(words2))

            #merge 2 response words
            union = len(words1.union(words2))

            overlap_ratio = overlap / union 

            assert overlap_ratio > 0.2, f"Low overlap ratio ({overlap_ratio}) between answers: '{answer1}' and '{answer2}'"
        print(f"Logical consistency test passed for: '{q1}' and '{q2}'")        

def test_error_handling(llm_client):
    error_inpute = ["What is the color of silence", "How may corners does a circle have?"]

    for prompt in error_inpute:
        response = llm_client.generate(prompt) 
        print(f"Response for prompt '{prompt}': '{response['text']}'")      

        assert len(response["text"]) > 20, f"Response too short for prompt: '{prompt}'"
        print(f"Error handling test passed for prompt: '{prompt}' with response: '{response['text']}'")

        error_handling_indicators = ["undefined", "not applicable", "no answer", "cannot answer", "doesn't make sense", "doesn't have", "not exist"]

        has_indicator =  any(indicator in response["text"].lower() for indicator in error_handling_indicators)
        assert has_indicator, f"Response does not indicate error handling for prompt: '{prompt}'"
        print(f"Error handling indicator found in response for prompt: '{prompt}'")