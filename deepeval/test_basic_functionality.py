from deepeval import assert_test
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from openaitestclient import OpenAITestClient

def test_basic_response():
    openai_client = OpenAITestClient()
    evaluation_model = GPTModel(model = "gpt-4o")
    prompt = "Hello, how are you?"
    response_text = openai_client.generate(prompt)

    test_case = LLMTestCase(input=prompt, actual_output=response_text["text"])

    response_quality_metric = GEval(
        name = "Response Quality", 
        criteria="Determine if the response is sustantial and appropriate, shoud be not empty and have more than 10 characters. Should be reasanable greeting response", 
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model = evaluation_model
        )
    
    assert_test(test_case, [response_quality_metric])


def test_instruction_following():
    openai_client = OpenAITestClient()
    evaluation_model = GPTModel(model = "gpt-4o")
    prompt = "Name three colors. Just list of colors"
    response_text = openai_client.generate(prompt)

    test_case = LLMTestCase(input=prompt, actual_output=response_text["text"])

    instruction_following_metric = GEval(
        name = "Instruction Following", 
        criteria="Determine if the response correctly follows the instruction to list at least three colors. The reponse should include names of colors only.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model = evaluation_model
        )
    
    assert_test(test_case, [instruction_following_metric])     


def test_simple_qa():
    openai_client = OpenAITestClient()
    evaluation_model = GPTModel(model = "gpt-4o")
    prompt = "What is the capital of France?"
    response_text = openai_client.generate(prompt)

    test_case = LLMTestCase(
        input=prompt, 
        actual_output=response_text["text"],
        expected_output="The capital of Farance is Paris."
        )

    qa_metric = GEval(
        name = "Simple QA", 
        criteria="Determine if the response correctly answers the question about the capital of France. The correct answer is 'Paris'.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8,
        model = evaluation_model
        )
    
    assert_test(test_case, [qa_metric])
    


