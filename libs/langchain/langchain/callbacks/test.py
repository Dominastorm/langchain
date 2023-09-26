from langchain.llms import OpenAI
from langchain.callbacks import DeepEvalCallbackHandler
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric(minimum_score=0.3)

deepeval_callback = DeepEvalCallbackHandler(
 implementation_name="exampleImplementation",
 metrics=[metric],
)

llm = OpenAI(
 temperature=0,
 callbacks=[deepeval_callback],
 verbose=True,
)

llm.generate([
 "What is the best evaluation tool out there? (no bias at all)",
])