
import streamlit as st
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def get_similar_docs(query, k=5):    
  embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

  api_key = "3c07ceec-454f-4d3d-bfa0-41d923abb7ed"
  pinecone = Pinecone(api_key = api_key)

  text_field = "text"
  index_name = "langchain-chatbot"
  index = pinecone.Index(index_name)

  vectorstore = PineconeStore(
      index, embeddings, text_field
  )

  similar_docs = vectorstore.similarity_search( query,  # our search query
                                                k=k,  # return k most relevant docs
                                                namespace=index_name)
  return similar_docs


def init_page() -> None:
  st.set_page_config(
      page_title = "Langchain RAG Chatbot"
  )
  st.header("Langchain RAG Chatbot")
  st.sidebar.title("Options")

def init_messages() -> None:
  clear_button = st.sidebar.button("Clear Conversation", key="clear")
  if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content = " You are a helpful AI assistant. Reply your answer in markdown format."
        )
    ]

def get_answer(llm, messages) -> str:
  #response = llm.complete(messages)
  #Q-A  
  chain = load_qa_chain(llm, chain_type="stuff")

  query = messages
  similar_docs = get_similar_docs(query)
  response = chain.run(input_documents=similar_docs, question=query)
  
  return response


def main() -> None:
  init_page()
  init_messages()
  llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_lenght":512})


  if user_input := st.chat_input("Input your question!"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("Bot is typing..."):
      answer = get_answer(llm,user_input)
      print(answer)    
    st.session_state.messages.append(AIMessage(content=answer))

  messages = st.session_state.get("messages",[])
  for message in messages:
    if isinstance(message, AIMessage):
      with st.chat_message("assistant"):
        st.markdown(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("user"):
        st.markdown(message.content)

if __name__  == "__main__":
   main()




