from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

def embed_and_store_in_chroma(chunks : List[Document]) -> Chroma:
    """
    Converts documents into embeddings and stores them in a persistent Chroma DB.

    Args:
        documents (List[Document]): Chunked documents.

    Returns:
        Chroma: The Chroma vector store object.
    """
     # Initialize the embedding model
    print("OPENAI_API_KEY: ", os.getenv("OPENAI_API_KEY"))
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    # Create Chroma vector store from documents

    # vectorstore = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embedding_model,
    #     persist_directory="chroma_db"
    # )

    #  # Save vectorstore to disk
    # vectorstore.persist()

     # Create FAISS index
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Save locally
    vectorstore.save_local("faiss_db")

    print(f"✅ Stored {len(chunks)} chunks in FAISS DB")

    return vectorstore

    