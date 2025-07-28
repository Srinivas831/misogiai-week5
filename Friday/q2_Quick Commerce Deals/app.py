# app.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Connect to SQLite
db = SQLDatabase.from_uri("sqlite:///zepto_inventory.db")

# Set up LLM (OpenAI GPT)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Create SQL agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True
)

# db = SQLDatabase.from_uri("sqlite:///zepto_inventory.db")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)

# ---------------- UI ----------------
st.set_page_config(page_title="Zepto SQL Agent", layout="centered")
st.title("🧠 Zepto Natural Language SQL Agent")

user_question = st.text_input("Enter your question about the inventory:", "")

if user_question:
    with st.spinner("Thinking..."):
        try:
            response = agent_executor.run(user_question)
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
