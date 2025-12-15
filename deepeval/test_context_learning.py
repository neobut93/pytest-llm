from deepeval import assert_test
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from openaitestclient import OpenAITestClient

openai_client = OpenAITestClient()
evaluation_model = GPTModel(model = "gpt-4o")


def test_logical_consistency():
    equivalent_questions = [
        ("What is the capital of Japan?", "Tokyo is the capital of which country?")]
    
    for q1, q2 in equivalent_questions:
        response1 = openai_client.generate(q1)
        response2 = openai_client.generate(q2)

        consistency_metric = GEval(
            name = "Logical Consistency", 
            criteria="""Evaluate the logical consistency between two responses to equivalent questions.
            1. Check if both responses refer to the same entity (e.g., "Tokyo" and "Japan").
            2. Assess the overlap in key information provided in both answers.
            3. No contradictions should be present.
            4. Ensure that both answers do not contradict each other.
            Award full points if the responses are logically consistent and refer to the same facts.
            """,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,
            model = evaluation_model
            )  

        response1 = response1["text"]
        response2 = response2["text"]
        combined_input = f"Question1: {q1}\nQuestion2: {q2}"
        combined_output = f"Response 1: {response1}\nResponse 2: {response2}"

        test_case = LLMTestCase(
                input=combined_input, 
                actual_output=combined_output    
            )

        assert_test(test_case, [consistency_metric])