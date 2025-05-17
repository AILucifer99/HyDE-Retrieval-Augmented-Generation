import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import sys
from pathlib import Path
import time
import pandas as pd

# Add the current directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import HyDERAG class
try:
    from hyDERAGPipeline import HyDERAG
except ImportError:
    st.error("Failed to import HyDERAG. Make sure hyDERAGPipeline.py is in the same directory as this app.")
    st.stop()

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="HyDERAG - Hypothetical Document Embeddings RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("HyDERAG: Hypothetical Document Embeddings RAG")
st.markdown("Upload documents and ask questions using the HyDE Retrieval-Augmented Generation approach.")

# Initialize session state variables
if "hyde_rag" not in st.session_state:
    st.session_state.hyde_rag = None
if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = False

# Functions
def check_api_keys():
    """Check if all required API keys are set."""
    keys = {
        "OpenAI API Key": os.getenv("OPENAI_API_KEY"),
        "Google API Key": os.getenv("GOOGLE_API_KEY"),
        "Groq API Key": os.getenv("GROQ_API_KEY")
    }
    
    missing_keys = [name for name, key in keys.items() if not key]
    
    if missing_keys:
        return False, missing_keys
    return True, []

def initialize_hyde_rag():
    """Initialize the HyDERAG instance with current settings."""
    with st.spinner("Initializing HyDERAG..."):
        try:
            st.session_state.hyde_rag = HyDERAG(
                verbose=1 if st.session_state.verbose else 0,
                embedding_provider=st.session_state.embedding_provider,
                embedding_model=st.session_state.embedding_model,
                hyde_llm_provider=st.session_state.hyde_llm_provider,
                hyde_llm_model=st.session_state.hyde_llm_model,
                answer_llm_provider=st.session_state.answer_llm_provider,
                answer_llm_model=st.session_state.answer_llm_model,
                temperature=st.session_state.temperature,
                hyde_max_tokens=st.session_state.hyde_max_tokens,
                answer_max_tokens=st.session_state.answer_max_tokens
            )
            return True
        except Exception as e:
            st.error(f"Error initializing HyDERAG: {str(e)}")
            return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the path."""
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def load_document(file_path):
    """Load document and create retriever."""
    with st.spinner("Loading document and creating embeddings..."):
        try:
            success = st.session_state.hyde_rag.load_data_and_create_retriever(
                data_path=file_path,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                retrieval_k=st.session_state.retrieval_k,
                force_reload=True,
                text_splitter=st.session_state.text_splitter
            )
            
            if success:
                st.session_state.current_file_path = file_path
                return True
            else:
                st.error("Failed to load document and create retriever.")
                return False
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return False

def answer_question(question, use_hyde=True):
    """Get answer to question using HyDERAG."""
    with st.spinner("Generating answer..."):
        try:
            start_time = time.time()
            
            # Get answer
            answer = st.session_state.hyde_rag.answer_question(
                question=question,
                disable_hyde=not use_hyde,
                retrieval_k=st.session_state.retrieval_k,
                temperature=st.session_state.temperature
            )
            
            execution_time = time.time() - start_time
            
            # Store in history
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "use_hyde": use_hyde,
                "time": execution_time
            })
            
            st.session_state.last_question = question
            
            return answer, execution_time
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return f"Error: {str(e)}", 0

def compare_approaches(question):
    """Compare HyDE and direct approaches."""
    with st.spinner("Comparing HyDE and direct approaches..."):
        try:
            start_time = time.time()
            
            results = st.session_state.hyde_rag.compare_approaches(
                question=question,
                retrieval_k=st.session_state.retrieval_k
            )
            
            execution_time = time.time() - start_time
            
            # Store in history
            st.session_state.history.append({
                "question": question,
                "answer": "COMPARISON",
                "hyde_answer": results["hyde_approach"],
                "direct_answer": results["direct_approach"],
                "use_hyde": "comparison",
                "time": execution_time
            })
            
            st.session_state.last_question = question
            
            return results, execution_time
        except Exception as e:
            st.error(f"Error comparing approaches: {str(e)}")
            return {"hyde_approach": f"Error: {str(e)}", "direct_approach": f"Error: {str(e)}"}, 0

# Sidebar for API keys and configurations
with st.sidebar:
    st.header("Configuration")
    
    # API Keys section
    with st.expander("API Keys", expanded=not st.session_state.api_keys_set):
        openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        google_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
        groq_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
        
        if st.button("Save API Keys"):
            # Save keys to environment variables
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["GOOGLE_API_KEY"] = google_key
            os.environ["GROQ_API_KEY"] = groq_key
            
            # Check if all keys are set
            keys_ok, missing = check_api_keys()
            if keys_ok:
                st.session_state.api_keys_set = True
                st.success("All API keys set successfully!")
            else:
                st.error(f"Missing API keys: {', '.join(missing)}")
    
    # Model Configuration
    with st.expander("Model Configuration", expanded=True):
        # Embeddings
        st.subheader("Embeddings")
        
        if "embedding_provider" not in st.session_state:
            st.session_state.embedding_provider = "google"
        
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["google", "openai"],
            index=0 if st.session_state.embedding_provider == "google" else 1,
            key="embedding_provider_select"
        )
        st.session_state.embedding_provider = embedding_provider
        
        if embedding_provider == "google":
            st.session_state.embedding_model = st.text_input(
                "Google Embedding Model",
                value=st.session_state.get("embedding_model", "models/embedding-001")
            )
        else:
            st.session_state.embedding_model = st.text_input(
                "OpenAI Embedding Model",
                value=st.session_state.get("embedding_model", "text-embedding-3-large")
            )
        
        # HyDE LLM
        st.subheader("HyDE LLM (Hypothetical Document Generator)")
        
        if "hyde_llm_provider" not in st.session_state:
            st.session_state.hyde_llm_provider = "groq"
        
        hyde_llm_provider = st.selectbox(
            "HyDE LLM Provider",
            options=["groq", "openai", "google"],
            index=["groq", "openai", "google"].index(st.session_state.hyde_llm_provider),
            key="hyde_llm_provider_select"
        )
        st.session_state.hyde_llm_provider = hyde_llm_provider
        
        hyde_model_defaults = {
            "groq": "gemma2-9b-it",
            "openai": "gpt-4o",
            "google": "gemini-1.5-flash-001"
        }
        
        if "hyde_llm_model" not in st.session_state:
            st.session_state.hyde_llm_model = hyde_model_defaults[hyde_llm_provider]
        
        st.session_state.hyde_llm_model = st.text_input(
            f"{hyde_llm_provider.title()} HyDE Model",
            value=st.session_state.get("hyde_llm_model", hyde_model_defaults[hyde_llm_provider])
        )
        
        # Answer LLM
        st.subheader("Answer LLM (Final Response Generator)")
        
        if "answer_llm_provider" not in st.session_state:
            st.session_state.answer_llm_provider = "google"
        
        answer_llm_provider = st.selectbox(
            "Answer LLM Provider",
            options=["groq", "openai", "google"],
            index=["groq", "openai", "google"].index(st.session_state.answer_llm_provider),
            key="answer_llm_provider_select"
        )
        st.session_state.answer_llm_provider = answer_llm_provider
        
        answer_model_defaults = {
            "groq": "gemma2-9b-it",
            "openai": "gpt-4o",
            "google": "gemini-1.5-flash-001"
        }
        
        if "answer_llm_model" not in st.session_state:
            st.session_state.answer_llm_model = answer_model_defaults[answer_llm_provider]
        
        st.session_state.answer_llm_model = st.text_input(
            f"{answer_llm_provider.title()} Answer Model",
            value=st.session_state.get("answer_llm_model", answer_model_defaults[answer_llm_provider])
        )
    
    # RAG Configuration
    with st.expander("RAG Configuration"):
        # Text Splitter
        if "text_splitter" not in st.session_state:
            st.session_state.text_splitter = "recursive"
        
        st.session_state.text_splitter = st.selectbox(
            "Text Splitter",
            options=["recursive", "character"],
            index=0 if st.session_state.text_splitter == "recursive" else 1
        )
        
        # Chunk Size
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 512
        
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=128,
            max_value=2048,
            value=st.session_state.chunk_size,
            step=32
        )
        
        # Chunk Overlap
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 128
        
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=512,
            value=st.session_state.chunk_overlap,
            step=16
        )
        
        # Retrieval K
        if "retrieval_k" not in st.session_state:
            st.session_state.retrieval_k = 5
        
        st.session_state.retrieval_k = st.slider(
            "Retrieval K (# of documents)",
            min_value=1,
            max_value=10,
            value=st.session_state.retrieval_k
        )
    
    # LLM Parameters
    with st.expander("LLM Parameters"):
        # Temperature
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.3
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1
        )
        
        # Max Tokens
        if "hyde_max_tokens" not in st.session_state:
            st.session_state.hyde_max_tokens = 512
        
        st.session_state.hyde_max_tokens = st.slider(
            "HyDE Max Tokens",
            min_value=128,
            max_value=2048,
            value=st.session_state.hyde_max_tokens,
            step=128
        )
        
        if "answer_max_tokens" not in st.session_state:
            st.session_state.answer_max_tokens = 1024
        
        st.session_state.answer_max_tokens = st.slider(
            "Answer Max Tokens",
            min_value=128,
            max_value=4096,
            value=st.session_state.answer_max_tokens,
            step=128
        )
    
    # Verbose logging
    if "verbose" not in st.session_state:
        st.session_state.verbose = True
    
    st.session_state.verbose = st.checkbox("Enable Verbose Logging", value=st.session_state.verbose)
    
    # Initialize button
    if st.button("Initialize HyDERAG"):
        keys_ok, missing = check_api_keys()
        if not keys_ok:
            st.error(f"Missing API keys: {', '.join(missing)}")
        else:
            if initialize_hyde_rag():
                st.success("HyDERAG initialized successfully!")

# Main content area
tab1, tab2, tab3 = st.tabs(["Document Processing", "Question Answering", "History"])

# Tab 1: Document Processing
with tab1:
    st.header("Upload and Process Document")
    
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, CSV)",
        type=["pdf", "txt", "csv"],
        help="Upload a document to process with HyDERAG."
    )
    
    if uploaded_file is not None:
        st.write(f"File uploaded: {uploaded_file.name}")
        
        if st.button("Process Document"):
            if st.session_state.hyde_rag is None:
                st.error("Please initialize HyDERAG first!")
            else:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    if load_document(file_path):
                        st.success(f"Document processed successfully: {uploaded_file.name}")

# Tab 2: Question Answering
with tab2:
    st.header("Ask Questions")
    
    if st.session_state.current_file_path:
        st.info(f"Current document: {os.path.basename(st.session_state.current_file_path)}")
        
        # Ask question
        question = st.text_area("Enter your question", height=100)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Ask with HyDE"):
                if question:
                    answer, exec_time = answer_question(question, use_hyde=True)
                    st.markdown("### Answer (HyDE approach)")
                    st.markdown(answer)
                    st.info(f"Response generated in {exec_time:.2f} seconds")
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("Ask without HyDE"):
                if question:
                    answer, exec_time = answer_question(question, use_hyde=False)
                    st.markdown("### Answer (Direct approach)")
                    st.markdown(answer)
                    st.info(f"Response generated in {exec_time:.2f} seconds")
                else:
                    st.warning("Please enter a question")
        
        # Compare approaches
        if st.button("Compare Approaches"):
            if question:
                results, exec_time = compare_approaches(question)
                
                st.markdown("### Comparison of Approaches")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### HyDE Approach")
                    st.markdown(results["hyde_approach"])
                
                with col2:
                    st.markdown("#### Direct Approach")
                    st.markdown(results["direct_approach"])
                
                st.info(f"Comparison completed in {exec_time:.2f} seconds")
            else:
                st.warning("Please enter a question")
    else:
        st.warning("Please upload and process a document first")

# Tab 3: History
with tab3:
    st.header("Question History")
    
    if st.session_state.history:
        # Convert history to DataFrame for display
        history_data = []
        
        for i, entry in enumerate(st.session_state.history):
            if entry["answer"] == "COMPARISON":
                # This was a comparison
                history_data.append({
                    "Index": i + 1,
                    "Question": entry["question"],
                    "Type": "Comparison",
                    "Time (s)": f"{entry['time']:.2f}"
                })
            else:
                # Regular question
                approach = "HyDE" if entry["use_hyde"] else "Direct"
                history_data.append({
                    "Index": i + 1,
                    "Question": entry["question"],
                    "Type": approach,
                    "Time (s)": f"{entry['time']:.2f}"
                })
        
        # Display history table
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        
        # Display selected history entry
        selected_index = st.selectbox(
            "Select entry to view details",
            options=list(range(1, len(st.session_state.history) + 1)),
            format_func=lambda i: f"{i}. {st.session_state.history[i-1]['question'][:50]}..."
        )
        
        if selected_index:
            entry = st.session_state.history[selected_index - 1]
            
            st.markdown(f"### Question {selected_index}")
            st.markdown(entry["question"])
            
            if entry["answer"] == "COMPARISON":
                # Display comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### HyDE Approach")
                    st.markdown(entry["hyde_answer"])
                
                with col2:
                    st.markdown("#### Direct Approach")
                    st.markdown(entry["direct_answer"])
            else:
                # Display regular answer
                approach = "HyDE" if entry["use_hyde"] else "Direct"
                st.markdown(f"#### Answer ({approach} approach)")
                st.markdown(entry["answer"])
            
            st.info(f"Response generated in {entry['time']:.2f} seconds")
    else:
        st.info("No questions asked yet")

# Footer
st.markdown("---")
st.markdown("HyDERAG - Hypothetical Document Embeddings for Retrieval-Augmented Generation")
