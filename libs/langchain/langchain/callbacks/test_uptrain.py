from langchain.llms import OpenAI
from langchain.callbacks import UpTrainCallbackHandler
from uptrain.operators import LanguageCritique, ResponseCompleteness, ResponseRelevance

language_critique = LanguageCritique()
response_completeness = ResponseCompleteness()

uptrain_callback = UpTrainCallbackHandler(
 checks=[language_critique, response_completeness],
)

llm = OpenAI(
 temperature=0,
 callbacks=[uptrain_callback],
 verbose=True,
)

response = llm.generate([
 "What is the best evaluation tool out there? (no bias at all)",
 "What does the fox say?",
])

# print(response)