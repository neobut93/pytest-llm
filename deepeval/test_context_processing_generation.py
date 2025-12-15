from deepeval import assert_test
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient()
evaluation_model = GPTModel(model = "gpt-4o")


def test_list_format():
    prompt = "List 3 fruits. Format the response as a numbered list."
    response_text = openai_client.generate(prompt)

    test_case = LLMTestCase(input=prompt, actual_output=response_text["text"])

    list_format_metric = GEval(
        name = "List Format", 
        criteria="""Determine if the response is formatted as a numbered list with at least three items:
        1. Contains excplicit numbering (e.g., "1.", "2.", "3.").
        2. Includes at least three distinct fruit names.
        3. Proper formatting with clear separation between items.
            
        Award full points only if all criteria are met.
        """,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model = evaluation_model
        )
    
    assert_test(test_case, [list_format_metric])


def test_simple_json():
        prompt = "Create a JSON with person's name and age. Use the name as John and age 30."
        response_text = openai_client.generate(prompt)

        test_case = LLMTestCase(input=prompt, actual_output=response_text["text"])

        json_format_metric = GEval(
            name = "Simple JSON", 
            criteria="""Determine if the response is a valid JSON object containing:
            1. A "name" field with the value "John".
            2. An "age" field with the value 30.
            3. Proper JSON syntax including braces and key-value pairs.

            Deduct points for any syntax errors or missing fields.
            """,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,
            model = evaluation_model
            )
        
        assert_test(test_case, [json_format_metric])