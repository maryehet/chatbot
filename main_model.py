from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl

#  custom prompt
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# path to vector db
db_path = "vectorstores/db_faiss"

# quantized Llama 2 LLM
llm = CTransformers(
  model = 'TheBloke/Llama-2-7B-Chat-GGML', 
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
    chain_type_kwargs= {'prompt': prompt}
  )
  result = chain({"query":query})
  return result['result']

print(qa('What are the features of the Saatva classic?'))
