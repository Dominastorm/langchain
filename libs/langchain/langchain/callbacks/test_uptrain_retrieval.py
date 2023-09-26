from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks import UpTrainCallbackHandler
from uptrain.operators import LanguageCritique, ResponseCompleteness, ResponseRelevance

language_critique = LanguageCritique()
response_completeness = ResponseCompleteness()

uptrain_callback = UpTrainCallbackHandler(
 checks=[language_critique, response_completeness],
)

loader = TextLoader("/home/insatanic/Downloads/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

return_source = False
return_source = True

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=return_source,
)

query = "What did the president say about Ketanji Brown Jackson"
if return_source:
    response = qa({"query": query, "source_documents": texts}, callbacks=[uptrain_callback])
else:
    response = qa.run(query, callbacks=[uptrain_callback])

print(response)