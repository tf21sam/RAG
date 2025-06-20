import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from config import load_config, get_groq_api_key
import tempfile
import time

load_config()
st.set_page_config(layout='wide', page_title="Groq for RAG")
groq_api_key = get_groq_api_key()

if 'vector' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    persist_directory = tempfile.mkdtemp()
    st.session_state.vectorstore = Chroma.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        persist_directory=persist_directory
    )

st.title('Groq for RAG')

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"  # or "llama3-8b-8192"
)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write(response['answer'])
    st.write(f'Response time: {(time.process_time() - start):.2f} sec')

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
