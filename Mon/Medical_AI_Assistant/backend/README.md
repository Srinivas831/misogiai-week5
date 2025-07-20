# Medical RAG Assistant - Latest LangChain Implementation

A comprehensive Retrieval-Augmented Generation (RAG) system for medical AI assistance using the latest LangChain patterns and FAISS vector store.

## 🏥 Overview

This Medical RAG Assistant implements state-of-the-art techniques for medical question answering:

- **Latest LangChain Patterns (2024)**: Uses `create_retrieval_chain` and `create_stuff_documents_chain`
- **FAISS Vector Store**: Fast and efficient similarity search
- **OpenAI Integration**: GPT-4 and text-embedding-3-small models
- **Streaming Responses**: Real-time answer generation
- **Conversational Context**: Multi-turn conversations with history
- **Medical Domain**: Specialized for medical/healthcare queries

## 🚀 Features

### Core Functionality
- ✅ **Modern RAG Pipeline**: Latest LangChain patterns (replaces deprecated RetrievalQA)
- ✅ **FAISS Vector Storage**: Fast similarity search and retrieval
- ✅ **OpenAI Models**: GPT-4o-mini and text-embedding-3-small
- ✅ **Streaming Support**: Real-time response generation
- ✅ **Chat History**: Conversational context management
- ✅ **Source Attribution**: Shows relevant document chunks
- ✅ **Error Handling**: Robust error handling and recovery

### Medical Specialization
- 🏥 **Medical Domain Focus**: Optimized for healthcare queries
- 🔍 **Evidence-Based Responses**: Uses only provided medical context
- ⚠️ **Safety Disclaimers**: Emphasizes professional medical consultation
- 📚 **Medical Document Processing**: Handles medical CSV data
- 🎯 **Accurate Retrieval**: Finds most relevant medical information

## 📁 Project Structure

```
backend/
├── medical_rag_assistant.py    # Main RAG assistant class
├── demo.py                     # Interactive demo script
├── main.py                     # Complete pipeline orchestration
├── data/                       # Medical CSV data files
├── faiss_db/                   # FAISS vector database
├── laoders/
│   └── csv_loader.py          # CSV document loader
├── splitters/
│   └── text_splitter.py       # Text chunking functionality
├── vectorstores/
│   └── vdb_chroma_db.py       # Vector store creation (now uses FAISS)
└── .env                       # Environment variables
```

## 🛠️ Installation & Setup

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv
```

### 2. API Configuration

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Data Preparation

Place your medical CSV files in the `data/` directory:
- `medical_tc_train.csv`
- `medical_tc_test.csv` 
- `medical_tc_labels.csv`
- etc.

## 🏃‍♂️ Quick Start

### Option 1: Complete Pipeline (Recommended)

```bash
# Run the complete pipeline
python main.py

# This will:
# 1. Load medical documents from CSV files
# 2. Split into chunks and create embeddings
# 3. Store in FAISS vector database
# 4. Initialize the RAG assistant
# 5. Provide interactive interface
```

### Option 2: Use Existing Database

```bash
# If you already have a FAISS database
python medical_rag_assistant.py
```

### Option 3: Run Demos

```bash
# Interactive demo with examples
python demo.py
```

### Option 4: Quick Test

```bash
# Test existing setup
python main.py test
```

## 💻 Usage Examples

### Basic Question Answering

```python
from medical_rag_assistant import MedicalRAGAssistant

# Initialize assistant
assistant = MedicalRAGAssistant()

# Ask a medical question
response = assistant.ask_question("What are the symptoms of diabetes?")
print(response['answer'])
```

### Streaming Responses

```python
# Stream responses for real-time interaction
for chunk in assistant.ask_question_stream("What causes hypertension?"):
    if chunk["type"] == "answer_chunk":
        print(chunk["content"], end="", flush=True)
```

### Conversational Context

```python
# Multi-turn conversation with context
assistant.ask_question("What is pneumonia?")
assistant.ask_question("What causes it?")  # Understands "it" refers to pneumonia
assistant.ask_question("How is it treated?")  # Maintains context
```

## 🧠 How It Works

### 1. Document Processing
```python
# Load medical documents from CSV files
docs = load_csv_data("data")

# Split into manageable chunks (1000 chars with 200 overlap)
chunks = split_documents(docs)
```

### 2. Vector Storage
```python
# Create embeddings and store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_db")
```

### 3. Retrieval Chain
```python
# Modern LangChain pattern
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

### 4. Query Processing
```python
# Process user question through complete pipeline
response = retrieval_chain.invoke({
    "input": question,
    "chat_history": formatted_history
})
```

## ⚙️ Configuration

### Model Settings

```python
# Language Model Configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",       # Latest OpenAI model
    temperature=0.1,           # Low for medical accuracy
    streaming=True,            # Enable streaming
    max_tokens=1000           # Response length limit
)

# Embedding Model Configuration  
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # Latest embedding model
)
```

### Retrieval Settings

```python
# Retriever Configuration
retriever = vectorstore.as_retriever(
    search_type="similarity",   # Cosine similarity
    search_kwargs={
        "k": 4,                # Top 4 relevant chunks
        "fetch_k": 20         # Consider top 20 candidates
    }
)
```

## 🔧 API Reference

### MedicalRAGAssistant Class

#### Methods

- `ask_question(question: str) -> Dict[str, Any]`
  - Process a medical question and return complete response
  - Returns: answer, sources, context_used, conversation_turn

- `ask_question_stream(question: str) -> Generator`
  - Stream response chunks for real-time interaction
  - Yields: answer chunks, sources, completion status

- `get_conversation_history() -> List[Dict[str, str]]`
  - Retrieve current conversation history

- `clear_conversation_history()`
  - Clear conversation history to start fresh

- `get_system_info() -> Dict[str, Any]`
  - Get system configuration and status

### Response Format

```python
{
    "answer": "Detailed medical response...",
    "sources": [
        {
            "chunk_id": 1,
            "content": "Relevant document excerpt...",
            "metadata": {"source": "medical_data.csv", "row": 123},
            "relevance_score": "Chunk 1"
        }
    ],
    "context_used": 4,
    "conversation_turn": 1
}
```

## 🎯 Medical Domain Features

### Safety and Accuracy
- **Conservative Responses**: Only uses provided medical context
- **Professional Disclaimers**: Emphasizes consulting healthcare professionals
- **No Diagnoses**: Avoids definitive medical diagnoses
- **Clear Limitations**: States when information is insufficient

### Medical Terminology
- **Appropriate Usage**: Uses medical terms correctly
- **Explanations**: Explains complex medical concepts
- **Evidence-Based**: Relies on document evidence
- **Concise Answers**: Focused 2-3 sentence responses

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   ```bash
   # Set environment variable
   export OPENAI_API_KEY=your_key_here
   # Or add to .env file
   ```

2. **"Could not load FAISS database"**
   ```bash
   # Run complete pipeline first
   python main.py
   ```

3. **"No documents found"**
   ```bash
   # Check data directory
   ls data/
   # Ensure CSV files exist
   ```

4. **Memory/Performance Issues**
   ```python
   # Reduce chunk size for large datasets
   chunks = chunks[:1000]  # Limit to 1000 chunks
   ```

### Performance Optimization

- **Batch Processing**: Process large datasets in smaller batches
- **Chunk Limiting**: Reduce number of chunks for faster processing
- **Model Selection**: Use smaller models for faster responses
- **Caching**: Enable response caching for repeated queries

## 📊 System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: 4GB+ RAM (8GB+ for large datasets)
- **Storage**: 1GB+ for vector database
- **API**: OpenAI API key with sufficient credits

## 🔄 Latest Changes (2024)

### Modernization Updates
- ✅ **Replaced RetrievalQA**: Now uses `create_retrieval_chain`
- ✅ **FAISS Integration**: Switched from ChromaDB to FAISS
- ✅ **LCEL Patterns**: Uses LangChain Expression Language
- ✅ **Streaming Support**: Real-time response generation
- ✅ **Better Error Handling**: Robust error management
- ✅ **Comprehensive Documentation**: Detailed code comments

### Performance Improvements
- 🚀 **Faster Retrieval**: FAISS provides better performance
- 🚀 **Streaming Responses**: Real-time user experience
- 🚀 **Optimized Prompts**: Better medical domain prompts
- 🚀 **Batch Processing**: Handles large datasets efficiently

## 📝 License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and OpenAI usage policies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add comprehensive comments
5. Test thoroughly
6. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify API keys and data files
5. Test with smaller datasets first

---

Built with ❤️ using the latest LangChain patterns for medical AI assistance. 