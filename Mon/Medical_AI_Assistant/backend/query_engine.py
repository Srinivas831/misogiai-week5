# 1st method direct lcel chain

# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# import os
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings

# load_dotenv()

# vectorStore = FAISS.load_local("faiss_db",OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")),allow_dangerous_deserialization=True)

# retriever = vectorStore.as_retriever(search_kwargs={"k":5})

# # -- 4️⃣ Build LCEL Pipeline --

# # Step A: run retriever and pass question through
# # RunnableParallel creates dict {"context": docs, "question": original_input}

# retrieval_step = RunnableParallel({
#     "context":retriever,
#     "question":RunnablePassthrough()
# })

# prompt = ChatPromptTemplate.from_messages([
#     ('system',"You are a helpful assistant that can answer questions about the following context: {context}"),
#     ('user',"{question}")
# ])


# # -- 3️⃣ Define LLM and Output Parser --
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# output_parser = StrOutputParser()


# # Step B: Chain the prompt, LLM, parser in sequence
# pipeline = (retrieval_step | prompt | llm | output_parser)

# # -- 5️⃣ Run the Pipeline --
# if __name__ == "__main__":
#     while True:
#         q = input("\n🔍 Ask a medical question (or 'exit'): ")
#         if q.lower().strip() == "exit":
#             break

#         # Execute LCEL chain
#         answer = pipeline.invoke(q)  # returns just the answer string

#         print("\n💬 Answer:\n", answer)







# 2nd method deeply with logging

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_community.vectorstores import FAISS
# import os

# # Load environment
# load_dotenv()

# # 1️⃣ Load Vector DB
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# vectorStore = FAISS.load_local(
#     "faiss_db",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

# # 2️⃣ Define Prompt
# prompt = ChatPromptTemplate.from_messages([
#     ('system', "You are a helpful assistant that can answer questions about the following context:\n\n{context}"),
#     ('user', "{question}")
# ])

# # 3️⃣ Define LLM and Parser
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
# output_parser = StrOutputParser()

# # 🧠 Step A: Retrieval
# retrieval_step = RunnableParallel({
#     "context": retriever,
#     "question": RunnablePassthrough()
# })

# # 🧠 Step B: Chain together (manual)
# if __name__ == "__main__":
#     while True:
#         q = input("\n🔍 Ask a medical question (or 'exit'): ")
#         if q.lower().strip() == "exit":
#             break

#         # 1. Retrieval step
#         retrieval_output = retrieval_step.invoke(q)
#         context_docs = retrieval_output["context"]
#         print("\n📚 Retrieved Context Documents:")
#         for i, doc in enumerate(context_docs):
#             print(f"\n[{i+1}] {doc.page_content[:500]}...")  # print first 500 chars

#         # 2. Prepare full prompt
#         prompt_input = prompt.invoke(retrieval_output)
#         print("\n📨 Constructed Prompt to LLM:\n", prompt_input.to_messages())

#         # 3. LLM call
#         llm_response = llm.invoke(prompt_input)
#         print("\n🤖 Raw LLM Response:\n", llm_response.content)

#         # 4. Final output parsing
#         answer = output_parser.invoke(llm_response)
#         print("\n✅ Final Answer:\n", answer)




# practice
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

vectorstore = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
print("vectorstore type", type(vectorstore))
print("----------------------")

retriever = vectorstore.as_retriever(search_kwargs={'k': 6})
print("retriever type",type(retriever))
print("----------------------")

retrieve_fn = RunnableParallel({
   "context":retriever,
   "question":RunnablePassthrough()
})

prompt = ChatPromptTemplate([
   ("system", "You are an helpful assistant who has knowledge medical realted, so you this context to answer the queries {context}"),
   ("user", "{question}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key = api_key)

if __name__ == "__main__":
    while True:
      q = input("Enter the medical query:(Type \"exit\" for exit) ")
      if q.lower().strip() == "exit":
         break
      
      req_for_prompt_retrieved = retrieve_fn.invoke(q)
    #   print("req_for_prompt_retrieved ype", type(req_for_prompt_retrieved))
      print("----------------------")
    #   print("req_for_prompt_retrieved", req_for_prompt_retrieved)

      to_llm = prompt.invoke(req_for_prompt_retrieved)
      print("to_llm",type(to_llm))
      print("------------------")

      final_ans = llm.invoke(to_llm)
      print("final_answer", type(final_ans))
      print("----------------------")
    #   print("final_answer",final_ans)
      print("final_answer",final_ans.content)







    