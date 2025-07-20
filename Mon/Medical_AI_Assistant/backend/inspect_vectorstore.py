from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

FAISS_DIR = "faiss_db"

def inspect_faiss_index(top_n=5):
    """Load FAISS index and print a few stored documents (metadata + content)"""
    print("🔍 Inspecting FAISS DB...")

    # Load embedding model and FAISS DB
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.load_local(FAISS_DIR, embedding_model,allow_dangerous_deserialization=True)

    # Get internal documents (stored as langchain Document objects)
    docs = vectorstore.docstore._dict  # this is a dictionary of {id: Document}
    
    print(f"📦 Total documents stored: {len(docs)}\n")

    # Print top N docs
    for i, (doc_id, doc) in enumerate(docs.items()):
        if i >= top_n:
            break
        print(f"📄 Document #{i+1}")
        print(f"🆔 ID: {doc_id}")
        print(f"📄 Content: {doc.page_content[:200]}...")  # Preview
        print(f"🧾 Metadata: {doc.metadata}\n")

if __name__ == "__main__":
    inspect_faiss_index()