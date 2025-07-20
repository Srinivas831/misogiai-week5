import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()  

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") 
INDEX_NAME = os.getenv("PINECONE_INDEX")


def initialize_pinecone():
     print("initing")
     # Initialize Pinecone with the new API
     pc = Pinecone(api_key=PINECONE_API_KEY)
     
     # Get the index
     return pc.Index(INDEX_NAME)

# if __name__ == "__main__":
#      print(initialize_pinecone())
    
