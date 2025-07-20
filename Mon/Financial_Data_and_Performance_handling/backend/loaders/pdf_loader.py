import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def load_pdf_files_and_split(folder):
    all_docs_chunks=[]
    for i, file in enumerate(os.listdir(folder)):
        print("index", i, "file",file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder,file))
            documents = loader.load()
            # print("documetns type", type(documents))
            # print("number of documents loaded:", len(documents))
            # print("sample document[0]:", documents[0])


            # splitting it
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            )

            chunks = text_splitter.split_documents(documents)
            # print("chunks", type(chunks))
            # print("---------------------")
            # print("chunks size", len(chunks))
            # print("---------------------")
            # print("chunks[0]", chunks[0])
            # for i, chunk in enumerate(chunks[:5]):  # Just printing first 5
            #  print(f"\n--- Chunk {i} ---")
            #  print(json.dumps({
            #  "page_content": chunk.page_content,
            #  "metadata": chunk.metadata
            #   }, indent=4))
            all_docs_chunks.extend(chunks)
    return all_docs_chunks



            



