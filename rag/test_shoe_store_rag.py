import os

from rag.shoe_store_rag import ShoeStoreRAG
from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

#todo check why only last test case is evaluated
def test_shoe_store_with_deepeval():

    pinecone_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    rag = ShoeStoreRAG(pinecone_key, openai_key)

    test_scenarios = [
        "What if these shoes do not fit?",
        "Do you offer student discounts?",
        "How long does shipping take",
        "What brands do you carry?"
    ]

    #todo check syntax again and loop iterations(try to debug it)
    for query in test_scenarios:
        retrieval_context = rag.retreive_context(query)
        actual_output = rag.generate_answer(query, retrieval_context)

        test_cases = []
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)

    contextual_relevency_metric =  ContextualRelevancyMetric(
            threshold=0.7,
            model = "gpt-4o",
            include_reason = True
        )

    evaluate(test_cases=test_cases, metrics=[contextual_relevency_metric])