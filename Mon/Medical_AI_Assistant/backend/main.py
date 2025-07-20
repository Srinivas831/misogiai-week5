from laoders.csv_loader import load_csv_data
from splitters.text_splitter import split_documents
from vectorstores.vdb_chroma_db import embed_and_store_in_chroma


if __name__ == "__main__":
      # Step 1: Load CSVs
    docs = load_csv_data("data")

     # Step 2: Split into chunks
    chunks = split_documents(docs) 
    # limit chunks list of Document to less and test it
    # chunks = chunks[:20]
    chunks = chunks[21:40]
    # Step 3: Embed & store in ChromaDB
    embed_and_store_in_chroma(chunks)





