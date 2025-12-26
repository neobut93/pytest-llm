from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

evaluation_model = GPTModel(model="gpt-4o")

actual_output = "We offer a 30-day full refund policy for all our products."

retrieval_context = [
    "Our company provides a 30-day money-back guarantee on all purchases."]

# error scenario
# retrieval_context = ["Out stores close at 9 PM on weekdays."]

context_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=evaluation_model,
    include_reason=True
)

test_case = LLMTestCase(
    input="What is your refund policy?",
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

evaluate([test_case], [context_relevancy_metric])