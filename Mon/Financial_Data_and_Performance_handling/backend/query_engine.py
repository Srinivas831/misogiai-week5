from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from config.pinecone_init import initialize_pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key =os.getenv("OPENAI_API_KEY"))
# Set up retriever and model
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

pinecone_index = initialize_pinecone()
vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embedder, text_key="text")
print("vectorstore type",type(vectorstore))
print("----------------------")

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
print("retriever type",type(retriever))
print("----------------------")

retrieve_fn = RunnableParallel({
    "context":retriever,
    "question":RunnablePassthrough()
})

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant in giving answers related to this context {context}"),
    ("user","{question}")
])

if __name__ == "__main__":
        while True:
         user_query = input("Enter the query: ")
         if user_query.lower().strip() == "quit":
            break
         
         req_retriever = retrieve_fn.invoke(user_query)
         print("req_retriever type",type(req_retriever))
         print("----------------------")

         full_prompt = prompt.invoke(req_retriever)
         print("full_prompt type",type(full_prompt))
         print("----------------------")

         res = llm.invoke(full_prompt)
         print("res type",type(res))
         print(res.content)


def get_answer(user_query: str) -> str:
    req_retriever = retrieve_fn.invoke(user_query)
    full_prompt = prompt.invoke(req_retriever)
    res = llm.invoke(full_prompt)
    return res.content

