from llamatestclient import llm_client

def test_basic_response(llm_client):
    response = llm_client.generate("Hello,how are you?")

    # asser not empty
    assert response['text'].strip() != ""
    assert len(response['text'].strip()) > 10, "response is too short"

    print(f"Basic response test - response length: {len(response['text'].strip())}")