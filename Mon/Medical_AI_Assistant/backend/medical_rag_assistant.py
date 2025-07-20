# Core imports for LangChain functionality
from langchain.chains import create_retrieval_chain  # Latest retrieval chain constructor
from langchain.chains.combine_documents import create_stuff_documents_chain  # Document combination
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Modern prompt templates
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types
from langchain_core.runnables import RunnablePassthrough, RunnableSequence  # LCEL components
from langchain_core.output_parsers import StrOutputParser  # Output parsing
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI integration
from langchain_community.vectorstores import FAISS  # Vector store
from langchain_core.documents import Document  # Document class
from typing import List, Dict, Any  # Type hints
import os
from dotenv import load_dotenv

# Load environment variables (contains OpenAI API key)
load_dotenv()

class MedicalRAGAssistant:
    """
    A comprehensive Medical RAG Assistant that uses the latest LangChain patterns
    to provide context-aware responses to medical queries.
    
    This class implements:
    - Modern retrieval chains using create_retrieval_chain
    - FAISS vector store for efficient similarity search
    - OpenAI embeddings and chat models
    - Streaming support for real-time responses
    - Chat history management for conversational interactions
    """
    
    def __init__(self, faiss_db_path: str = "faiss_db"):
        """
        Initialize the Medical RAG Assistant.
        
        Args:
            faiss_db_path (str): Path to the FAISS database directory
        """
        # Store the database path for later use
        self.faiss_db_path = faiss_db_path
        
        # Initialize core components
        self._setup_llm()            # Set up the language model
        self._setup_embeddings()     # Set up embedding model
        self._load_vector_store()    # Load the FAISS vector store
        self._setup_retriever()      # Configure the retriever
        self._setup_prompt()         # Create prompt templates
        self._setup_chains()         # Build the RAG chains
        
        # Initialize chat history for conversational context
        self.chat_history: List[Dict[str, str]] = []
        
        print("✅ Medical RAG Assistant initialized successfully!")
    
    def _setup_llm(self):
        """
        Initialize the Large Language Model (LLM) with optimal settings.
        
        Uses OpenAI's GPT-4 model with specific configurations:
        - Temperature 0.1 for focused, factual responses (medical accuracy)
        - Streaming enabled for real-time response generation
        """
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")
        
        # Initialize ChatOpenAI with medical-appropriate settings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",           # Latest, most capable model
            temperature=0.1,               # Low temperature for accurate medical responses
            api_key=api_key,              # OpenAI API authentication
            streaming=True,               # Enable streaming for real-time responses
            max_tokens=1000,              # Reasonable response length limit
        )
        print("🤖 Language Model (GPT-4) initialized")
    
    def _setup_embeddings(self):
        """
        Initialize the embedding model for converting text to vectors.
        
        Uses OpenAI's text-embedding-3-small model which provides:
        - High quality semantic embeddings
        - Good performance for medical domain
        - Cost-effective compared to larger embedding models
        """
        # Create embedding model instance
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Latest small embedding model
            api_key=os.getenv("OPENAI_API_KEY")  # Same API key as LLM
        )
        print("🔢 Embedding Model (text-embedding-3-small) initialized")
    
    def _load_vector_store(self):
        """
        Load the pre-existing FAISS vector store containing medical documents.
        
        FAISS (Facebook AI Similarity Search) provides:
        - Fast similarity search over high-dimensional vectors
        - Memory-efficient storage
        - No external database dependencies
        """
        try:
            # Load the FAISS index from disk with embeddings
            self.vectorstore = FAISS.load_local(
                self.faiss_db_path,        # Path to FAISS database directory
                self.embeddings,           # Embedding model for consistency
                allow_dangerous_deserialization=True  # Required for loading pickled data
            )
            print(f"📊 FAISS Vector Store loaded from '{self.faiss_db_path}'")
        except Exception as e:
            raise FileNotFoundError(f"❌ Could not load FAISS database from '{self.faiss_db_path}': {str(e)}")
    
    def _setup_retriever(self):
        """
        Configure the retriever component for finding relevant documents.
        
        The retriever converts the vector store into a searchable interface
        with optimized parameters for medical query resolution.
        """
        # Convert vector store to retriever with search parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",     # Use cosine similarity for document matching
            search_kwargs={
                "k": 4,                   # Retrieve top 4 most relevant chunks
                "fetch_k": 20,           # Consider top 20 candidates before filtering
            }
        )
        print("🔍 Document Retriever configured (top-4 similarity search)")
    
    def _setup_prompt(self):
        """
        Create sophisticated prompt templates for different interaction patterns.
        
        Uses ChatPromptTemplate for structured conversation management:
        - System prompt defines AI's role and behavior
        - Context injection for retrieved medical information
        - Message history placeholder for conversational continuity
        """
        
        # Define the system prompt that establishes the AI's role
        self.system_prompt = """
        You are a knowledgeable Medical AI Assistant specialized in providing accurate, evidence-based medical information.

        INSTRUCTIONS:
        1. Use ONLY the provided context to answer medical questions
        2. If the context doesn't contain relevant information, clearly state "I don't have enough information to answer that question safely"
        3. Provide clear, concise responses in 2-3 sentences maximum
        4. Always emphasize that users should consult healthcare professionals for medical advice
        5. Never provide definitive diagnoses or treatment recommendations
        6. Use medical terminology appropriately but explain complex terms

        CONTEXT:
        {context}
        
        Remember: This is for informational purposes only and does not replace professional medical consultation.
        """
        
        # Create the main prompt template for question-answering
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),           # System role definition
            MessagesPlaceholder("chat_history"),      # Placeholder for conversation history
            ("human", "{input}"),                     # User's current question
        ])
        
        # Create a simpler prompt for document processing
        self.document_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        print("💬 Prompt Templates configured for medical assistance")
    
    def _setup_chains(self):
        """
        Build the complete RAG chain using modern LangChain patterns.
        
        Creates two main chains:
        1. Document Chain: Processes retrieved documents with the LLM
        2. Retrieval Chain: Combines retrieval + document processing
        
        This replaces the deprecated RetrievalQA class with the latest approach.
        """
        
        # Step 1: Create the document processing chain
        # This chain takes retrieved documents and generates an answer
        self.document_chain = create_stuff_documents_chain(
            llm=self.llm,                    # Language model for generation
            prompt=self.document_prompt,     # Prompt template for document processing
            output_parser=StrOutputParser(), # Parse LLM output to string
        )
        
        # Step 2: Create the complete retrieval chain
        # This chain combines document retrieval + processing
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,        # Document retriever component
            combine_docs_chain=self.document_chain  # Document processing chain
        )
        
        # Step 3: Create a conversational chain for chat history
        # This handles multi-turn conversations with context
        self.conversational_chain = RunnableSequence(
            # First step: extract input and prepare context
            RunnablePassthrough.assign(
                # Combine user input with chat history
                context=lambda x: self._format_chat_history()
            ),
            # Second step: process through retrieval chain
            self.retrieval_chain
        )
        
        print("⛓️  RAG Chains constructed using latest LangChain patterns")
    
    def _format_chat_history(self) -> str:
        """
        Format the chat history into a readable string for context.
        
        Returns:
            str: Formatted chat history for prompt injection
        """
        if not self.chat_history:
            return ""
        
        # Convert chat history to formatted string
        formatted_history = []
        for entry in self.chat_history[-6:]:  # Keep last 6 exchanges (3 rounds)
            formatted_history.append(f"Human: {entry['human']}")
            formatted_history.append(f"Assistant: {entry['assistant']}")
        
        return "\n".join(formatted_history)
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Process a medical question through the complete RAG pipeline.
        
        This method orchestrates the entire process:
        1. Query preprocessing and validation
        2. Document retrieval from FAISS vector store
        3. Context preparation and prompt construction
        4. LLM inference with retrieved context
        5. Response post-processing and formatting
        
        Args:
            question (str): User's medical question
            
        Returns:
            Dict[str, Any]: Complete response with answer, sources, and metadata
        """
        
        print(f"\n🔍 Processing question: '{question[:50]}...'")
        
        try:
            # Step 1: Validate input
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid medical question.",
                    "sources": [],
                    "error": "Empty question"
                }
            
            # Step 2: Execute the retrieval chain
            print("📚 Retrieving relevant medical documents...")
            response = self.retrieval_chain.invoke({
                "input": question.strip(),        # Clean user input
                "chat_history": self._format_chat_history()  # Include conversation context
            })
            
            # Step 3: Extract information from response
            answer = response.get("answer", "I couldn't generate a response.")
            source_documents = response.get("context", [])
            
            # Step 4: Process source documents for user display
            sources = []
            for i, doc in enumerate(source_documents):
                sources.append({
                    "chunk_id": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": f"Chunk {i+1}"
                })
            
            # Step 5: Update chat history for conversational context
            self.chat_history.append({
                "human": question,
                "assistant": answer
            })
            
            # Step 6: Prepare complete response
            result = {
                "answer": answer,
                "sources": sources,
                "context_used": len(source_documents),
                "conversation_turn": len(self.chat_history)
            }
            
            print(f"✅ Response generated using {len(source_documents)} document chunks")
            return result
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "error": error_msg
            }
    
    def ask_question_stream(self, question: str):
        """
        Process a question with streaming response for real-time interaction.
        
        This method provides the same functionality as ask_question but yields
        response chunks as they're generated, allowing for real-time display.
        
        Args:
            question (str): User's medical question
            
        Yields:
            Dict[str, Any]: Response chunks as they're generated
        """
        
        print(f"\n🔄 Streaming response for: '{question[:50]}...'")
        
        try:
            # Retrieve relevant documents first
            docs = self.retriever.invoke(question)
            
            # Yield source information immediately
            yield {
                "type": "sources",
                "sources": [
                    {
                        "chunk_id": i + 1,
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for i, doc in enumerate(docs)
                ]
            }
            
            # Stream the answer generation
            accumulated_answer = ""
            for chunk in self.retrieval_chain.stream({
                "input": question,
                "chat_history": self._format_chat_history()
            }):
                if "answer" in chunk:
                    # Yield each piece of the answer as it's generated
                    accumulated_answer += chunk["answer"]
                    yield {
                        "type": "answer_chunk",
                        "content": chunk["answer"],
                        "accumulated": accumulated_answer
                    }
            
            # Update chat history after completion
            self.chat_history.append({
                "human": question,
                "assistant": accumulated_answer
            })
            
            # Yield completion signal
            yield {
                "type": "complete",
                "final_answer": accumulated_answer
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": f"Streaming error: {str(e)}"
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the current conversation history.
        
        Returns:
            List[Dict[str, str]]: List of conversation exchanges
        """
        return self.chat_history.copy()
    
    def clear_conversation_history(self):
        """
        Clear the conversation history to start fresh.
        """
        self.chat_history.clear()
        print("🗑️  Conversation history cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current system configuration.
        
        Returns:
            Dict[str, Any]: System configuration details
        """
        try:
            # Get vector store statistics
            total_docs = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else "Unknown"
            
            return {
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "vector_store": "FAISS",
                "total_documents": total_docs,
                "retrieval_k": 4,
                "conversation_turns": len(self.chat_history),
                "database_path": self.faiss_db_path,
                "status": "Ready"
            }
        except Exception as e:
            return {
                "status": "Error",
                "error": str(e)
            }


def main():
    """
    Main function to demonstrate the Medical RAG Assistant functionality.
    
    This provides a simple command-line interface for testing the assistant.
    """
    
    print("🏥 Medical RAG Assistant - Latest LangChain Implementation")
    print("=" * 60)
    
    try:
        # Initialize the assistant
        assistant = MedicalRAGAssistant()
        
        # Display system information
        info = assistant.get_system_info()
        print(f"\n📊 System Status: {info['status']}")
        print(f"📚 Documents in database: {info['total_documents']}")
        
        # Interactive loop for testing
        print("\n💬 You can now ask medical questions! (Type 'quit' to exit, 'clear' to reset)")
        
        while True:
            # Get user input
            question = input("\n🤔 Your question: ").strip()
            
            # Handle special commands
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'clear':
                assistant.clear_conversation_history()
                continue
            elif not question:
                print("Please enter a valid question.")
                continue
            
            # Process the question
            try:
                response = assistant.ask_question(question)
                
                # Display the response
                print(f"\n🤖 Assistant: {response['answer']}")
                print(f"\n📊 Used {response['context_used']} document chunks")
                
                # Show sources if available
                if response['sources']:
                    print("\n📚 Source Information:")
                    for source in response['sources'][:2]:  # Show top 2 sources
                        print(f"  • {source['relevance_score']}: {source['content']}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {str(e)}")
        print("Make sure FAISS database exists and OPENAI_API_KEY is set.")


if __name__ == "__main__":
    main() 