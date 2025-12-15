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
            5. Provide sufficient explanation(not just "I don't know").
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


def test_error_handling():

    error_inputs= [
        "What is the color of silence?",
        "How many corners does a circle have"
    ]

    for prompt in error_inputs:
        response_text = openai_client.generate(prompt)

        error_handling_metric = GEval(
            name = "Error Handling", 
            criteria="""Evaluate how well the model handles nonsensical or unanswerable questions.
            1. Check if the response acknowledges the ambiguity or impossibility of the question.
            2. Assess whether the model refrains from providing incorrect or fabricated information.
            3. Look for polite refusals or requests for clarification.
            Award full points if the response appropriately addresses the nonsensical nature of the input.
            """,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.75,
            model = evaluation_model
            )  

        test_case = LLMTestCase(input=prompt, actual_output=response_text["text"])
             
        
        print("Testing prompt:", prompt)
        print("Response:", response_text["text"])
        assert_test(test_case, [error_handling_metric])

