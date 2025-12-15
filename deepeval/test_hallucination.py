from deepeval import assert_test
from deepeval.models import GPTModel
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from openaitestclient import OpenAITestClient

def test_hallucination_detection():
    openai_client = OpenAITestClient()
    evaluation_model = GPTModel(model = "gpt-4o")

    impossible_questions = ["What is the name of the president of Wakanda?",
                            "What was the color of Napoleon's favourite cell phone?"]

    for question in impossible_questions:                    
        response_text = openai_client.generate(question)


        hallucination_metric = HallucinationMetric(
            threshold=0.7,
            model = evaluation_model
        )

        if "wakanda" in question.lower():
            context = "Wakanda is a fictional country from the Marvel Universe."
        elif "napoleon" in question.lower():
            context = "Napoleon Bonaparte lived in the 18th and 19th centuries, long before cell phones were invented."
        else:
            context = "This question is about an impossible or fictional scenario."

        test_case = LLMTestCase(
            input=question, 
            actual_output=response_text["text"],
            context=[context]
            )
    
    
        assert_test(test_case, [hallucination_metric])