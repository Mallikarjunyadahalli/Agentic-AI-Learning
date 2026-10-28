
# Load the requried Modules.
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class State(TypedDict):
    programee: str
    messages: Annotated[list, add_messages ]
    query_type: str
    retrived_context: str

# Build a retrived context that is RAG
def retrive_context(file_path: str):
    # load the files
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    # Create the chunks for each documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents=documents)

    vector_db = FAISS(chunks, embedding_model)

    return vector_db.as_retriever(search_kwargs={"k" : 3})


# Classify the programee
def classifier_node(state: State):
    """ Look at the lastest user messages and decide which path to take. """
    last_message = state['messages'][-1].content
    