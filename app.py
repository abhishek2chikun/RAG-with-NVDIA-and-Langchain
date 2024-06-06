import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import pdfplumber,faiss



from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")




def save_uploaded_files(uploaded_files):
    # Remove previous files from the directory
    if os.path.exists("Docu"):
        for file_name in os.listdir("Docu"):
            file_path = os.path.join("Docu", file_name)
            os.remove(file_path)

    # Save the new uploaded files
    if not os.path.exists("Docu"):
        os.makedirs("Docu")
    for uploaded_file in uploaded_files:
        file_path = os.path.join("Docu", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())


def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./Docu") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


#Upload you document 

st.title("RAG with NVIDIA and Langchain")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# File uploader allows multiple files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Save the uploaded files
    if st.button("Save Uploaded Files"):
        save_uploaded_files(uploaded_files)
        st.success("File Saved successfully performing embedding")
        with st.spinner('Wait for it...'):
            vector_embedding()
        st.write("Vector embedding completed.")
        # st.write(f"Total documents processed: {len(st.session_state.final_documents)}")


prompt1=st.text_input("Enter Your Question From Documents")


if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")