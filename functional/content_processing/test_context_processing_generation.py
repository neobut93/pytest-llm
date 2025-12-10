import re
from llamatestclient import llm_client

def test_list_format(llm_client):
    prompt = "List 3 fruits. Format the response as a numbered list."

    response = llm_client.generate(prompt)

    list_pattern = r'[1-3][.)]'
    matches = re.findall(list_pattern, response['text'])

    assert len(matches) >= 3, f"Expected at least 3 list items, got: {response['text']}"
    print(f"List format test passed with {len(matches)} items found.")

def test_simple_json(llm_client):
    prompt = "Cretae a JSON with persons's name and age. Use the nae is John and age 30."    

    response = llm_client.generate(prompt)
    
    contains_braces = '{' in response['text'] and '}' in response['text']
    contains_name = 'john' in response['text'].lower() and 'name' in response['text'].lower()
    contains_age = '30' in response['text'].lower() and 'age' in response['text'].lower()
    assert contains_braces, "Response does not contain JSON braces."
    assert contains_name, "Response does not contain the correct name field."
    assert contains_age, "Response does not contain the correct age field."
    print("Simple JSON test passed.")
    print(f"Response: {response['text']}")