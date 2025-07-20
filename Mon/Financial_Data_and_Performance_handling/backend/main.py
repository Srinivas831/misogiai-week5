from loaders.pdf_loader import load_pdf_files_and_split
from vectorstore.store import embed_and_store

if __name__ == "__main__":
    print("Gettin started")

    all_chunks = load_pdf_files_and_split("data")

    embed_and_store(all_chunks)




