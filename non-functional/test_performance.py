import time
from llamatestclient import llm_client

def test_basic_throughput(llm_client):
    prompt = "hello"

    start_time = time.time()
    responses = []

    for _ in range(5):
        response = llm_client.generate(prompt, max_tokens=20)
        responses.append(response)

    end_time = time.time()
    total_time = end_time - start_time

    #todo calculate throughput correctly
    total_tokens = sum(r["completion_tokens"] for r in responses)
    tokens_per_second = total_tokens / total_time
    requests_per_munite = (5 / total_time * 60)
    
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"Request rate: {requests_per_munite:.2f} requests/minute")
    print(f"Average response size: {total_tokens / 5:.2f} tokens")