from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl

db_path = "vectorstores/db_faiss"

custom_prompt_template = """Use following information to answer user question. 
If you don't know the answer, please say you don't know and don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else. 
Answer:
"""
##test##
# llm = CTransformers(
#     model = 'llama-2-7b-chat.ggmlv3.q8_0.bin',
#     callbacks=[StreamingStdOutCallbackHandler()]
#   )

# def test_qa(q):
#   prompt = PromptTemplate(template =  custom_prompt_template, input_variables=["question"] )
#   llm_chain = LLMChain(prompt= prompt, llm = llm)
#   response = llm_chain.run(q)
#   return response
####
def set_custom_prompt():
  '''
  prompt template for QA retrieval for each vector stores
  '''
  prompt = PromptTemplate(template =  custom_prompt_template, input_variables=['context','question'] )
  return prompt 

llm = CTransformers(
  model = 'llama-2-7b-chat.ggmlv3.q8_0.bin',
  model_type = 'llama',
  max_new_tokens = 512,
  temperature = 0.5
  )


def retrieval_qa_chain(llm, prompt, db):
  qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type= 'stuff',
    retriever = db.as_retriever(search_kwargs={'k':2}),
    return_source_documents = True,
    chain_type_kwargs= {'prompt': prompt}
  )
  return qa_chain

def qa_bot():
  embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', 
                                     model_kwargs = {'device': 'cpu'})
  db = FAISS.load_local(db_path, embeddings)
  #db =  FAISS.from_documents( ,embeddings)
  qa_prompt = set_custom_prompt()
  qa = retrieval_qa_chain(llm, qa_prompt, db)
  return qa

def final_result(query):
  qa_result = qa_bot()
  response = qa_result({'query': query})
  return response

print(final_result('how durable are the mattresses?'))


## chainlit code
# @cl.on_chat_start
# async def start():
#   chain = qa_bot()
#   msg = cl.Message(content='Bot starting...')
#   await msg.send()
#   msg.content = 'Hi welcome to the Staava bot. What product(s) do you want to know about?'
#   await msg.update()
#   cl.user_session.set('chain', chain)

# @cl.on_message
# async def main(message):
#   chain = cl.user_session.get('chain')
#   callback = cl.AsyncLangchainCallbackHandler(
#     stream_final_answer = True, answer_prefix_tokens= ['FINAL', 'ANSWER']
#   )
#   callback.answer_reached = True
#   result = await chain.acall(message, callbacks = [callback])
#   answer = result['result']
#   sources = result['source_documents']

#   if sources:
#     answer += f"\nSources:" + str(sources)
#   else:
#     answer += f"\nNo sources found"

#   await cl.Message(content = answer).send()
