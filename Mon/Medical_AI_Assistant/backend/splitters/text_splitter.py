from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def split_documents(documents:List[Document]):
    """
    Splits the input documents into smaller chunks using recursive splitting.
    
    Args:
        documents (List[Document]): Raw documents to split.

    Returns:
        List[Document]: A flat list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    # Perform the actual splitting
    split_documents = text_splitter.split_documents(documents)
    print(f"✅ Total documents split: {len(split_documents)}")
    print("-------------------------")
    print("split_documents type: ", type(split_documents))
    print("-------------------------")
    print("split_documents[0]: ", split_documents[0])
    print("-------------------------")
    print("split_documents[0] type: ", type(split_documents[0]))
    return split_documents




