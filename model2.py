from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl

db_path = "vectorstores/db_faiss"

llm = CTransformers(
  model = 'llama-2-7b-chat.ggmlv3.q8_0.bin', 
  model_type = 'llama',
  max_new_tokens = 512,
  temperature = 0.5
)

embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local(db_path, embeddings)

def qa(query):
  chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type= 'stuff',
    retriever = db.as_retriever(search_kwargs={'k':2}),
    return_source_documents = True,
  )
  result = chain({"query":query})
  return result['result']

print(qa('how durable are the mattresses?'))


