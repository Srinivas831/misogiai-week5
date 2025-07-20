from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from config.pinecone_init import initialize_pinecone, INDEX_NAME

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def embed_and_store(chunks):
    # Step 1: Init Pinecone client
    pinecone_index = initialize_pinecone()
    
    # Step 2: Initialize embeddings
    embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    # Step 3: Use LangChain's Pinecone VectorStore wrapper
    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embed,
        text_key="text"  # Default text key
    )
    
    # Add documents to the vectorstore
    vectorstore.add_documents(chunks)

    print("✅ Chunks embedded and stored in Pinecone.")
    return vectorstore




# if __name__ == "__main__":
#     embed_and_store()