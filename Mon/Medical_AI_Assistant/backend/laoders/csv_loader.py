from langchain_community.document_loaders import CSVLoader
import os


def load_csv_data(folder):
    """
    Loads all CSV files from the specified folder and returns a combined list of LangChain documents.
    
    Args:
        data_folder_path (str): Path to the folder containing CSV files.

    Returns:
        List[Document]: List of all documents extracted from all CSVs.
    """
    documents = []  # This will store all the loaded documents

    for index,file in enumerate(os.listdir(folder)):
        if file.endswith(".csv"):
            # print("file: ", file) 
            # print("file path: ", os.path.join(folder, file))
            loader = CSVLoader(file_path=os.path.join(folder, file))
            docs = loader.load()
            # if index == 0:
                # print("docs: ", docs)
                # print("-------------------------")
                # print("docs length: ", len(docs))
                # print("-------------------------")
                # print("docs type: ", type(docs))
                # print("-------------------------")
                # print("docs[0]: ", docs[0])
            documents.extend(docs)
            # print(f"✅ Total documents loaded: {len(documents)}") 
            # print("-------------------------")
            # print("documents[0]: ", documents[0])
            # print("-------------------------")
            # print("documents[0] type: ", type(documents[0]))
            # print("-------------------------")

    return documents
   