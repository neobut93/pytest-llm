from llamatestclient import llm_client

def test_basic_response(llm_client):
    response = llm_client.generate("Hello,how are you?")

    # asser not empty
    assert response['text'].strip() != ""
    assert len(response['text'].strip()) > 10, "response is too short"

    print(f"Basic response test - response length: {len(response['text'].strip())}")

def test_instruction_following(llm_client):
    response = llm_client.generate("Name three colosrs. Just list of colors")
    response_text = response['text'].lower()

    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]

    colot_count = sum(1 for color in colors if color in response_text)

    assert colot_count >= 3, f"Expected at least 3 colors, got {colot_count}"
    print(f"Colors found in response: {colot_count}")

def test_simple_qa(llm_client):
    response = llm_client.generate("What is the capital of France?")
    response_text = response['text'].lower()

    assert "paris" in response_text, f"Expected 'Paris' in response, got: {response['text']}"
    print("Simple QA test passed.")

def test_multi_turn_basic(llm_client):
    prompt = """
    User: My name is Sam
    Assistant: Hello Sam! How can I assist you today?
    User: What is my name?
    """

    response = llm_client.generate(prompt)
    assert "sam" in response['text'].lower(), f"Expected 'Sam' in response, got: {response['text']}"
    print("Multi-turn basic test passed.")
