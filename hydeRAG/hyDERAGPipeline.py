from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader, CSVLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.embeddings import Embeddings
from dotenv import load_dotenv, find_dotenv
import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Type
from langchain.schema import Document


class HyDERAG:
    """
    A class for implementing Hypothetical Document Embeddings (HyDE) RAG pipeline.
    """
    
    # Class variables to track vector stores
    _vector_stores = {}
    _current_data_path = None
    
    def __init__(self, 
                verbose: int = 0,
                env_file: Optional[str] = None,
                embedding_provider: str = "google",
                embedding_model: Optional[str] = None,
                hyde_llm_provider: str = "groq",
                hyde_llm_model: Optional[str] = None,
                answer_llm_provider: str = "google",
                answer_llm_model: Optional[str] = None,
                temperature: float = 0.3,
                hyde_max_tokens: int = 512,
                answer_max_tokens: int = 1024):
        """
        Initialize the HyDE RAG system.
        
        Args:
            verbose: Verbosity level (0: minimal, 1: detailed, 2: debug)
            env_file: Path to .env file (if not using find_dotenv)
            embedding_provider: Provider for embeddings ("google" or "openai")
            embedding_model: Specific embedding model to use (None for default)
            hyde_llm_provider: Provider for HyDE LLM ("groq", "openai", "google")
            hyde_llm_model: Specific model for HyDE (None for default)
            answer_llm_provider: Provider for answer LLM ("groq", "openai", "google")
            answer_llm_model: Specific model for answers (None for default)
            temperature: Temperature for LLM generation
            hyde_max_tokens: Max tokens for hypothetical answer generation
            answer_max_tokens: Max tokens for final answer generation
        """
        self.verbose = verbose
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model
        self.hyde_llm_provider = hyde_llm_provider
        self.hyde_llm_model_name = hyde_llm_model
        self.answer_llm_provider = answer_llm_provider
        self.answer_llm_model_name = answer_llm_model
        self.temperature = temperature
        self.hyde_max_tokens = hyde_max_tokens
        self.answer_max_tokens = answer_max_tokens
        
        # Setup components
        self._setup_logging()
        self._load_environment(env_file)
        self._initialize_models()
        
    def _setup_logging(self):
        """Configure logging based on verbosity level."""
        logging_level = logging.INFO if self.verbose > 0 else logging.WARNING
        logging.basicConfig(
            level=logging_level,
            format='[%(levelname)s] %(message)s'
        )
        self.logger = logging
    
    def _load_environment(self, env_file: Optional[str] = None):
        """Load environment variables from .env file."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv(find_dotenv())
        
        # Set API keys from environment variables
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        
        if not all([os.environ["OPENAI_API_KEY"], 
                   os.environ["GOOGLE_API_KEY"], 
                   os.environ["GROQ_API_KEY"]]):
            self.logger.warning("Some API keys may be missing. Check your .env file.")
            
    def _initialize_models(self):
        """Initialize embedding and LLM models."""
        # Initialize embeddings
        if self.embedding_provider.lower() == "google":
            model_name = self.embedding_model_name or "models/embedding-001"
            self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        elif self.embedding_provider.lower() == "openai":
            model_name = self.embedding_model_name or "text-embedding-3-large"
            self.embeddings = OpenAIEmbeddings(model=model_name)
        else:
            self.logger.warning(f"Unknown embedding provider: {self.embedding_provider}. Using Google as default.")
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Initialize LLM models
        self.llm_models = {}
        
        # Groq LLM
        groq_model = self.hyde_llm_model_name if self.hyde_llm_provider == "groq" else "gemma2-9b-it"
        self.llm_models["groq"] = ChatGroq(
            model=groq_model, 
            temperature=self.temperature,
            max_tokens=self.hyde_max_tokens,
            model_kwargs={"top_p": 0.9}
        )
        
        # OpenAI LLM
        openai_model = self.hyde_llm_model_name if self.hyde_llm_provider == "openai" else "gpt-4o"
        self.llm_models["openai"] = ChatOpenAI(
            model=openai_model, 
            max_tokens=self.answer_max_tokens, 
            temperature=self.temperature,
        )
        
        # Google LLM
        google_model = self.hyde_llm_model_name if self.hyde_llm_provider == "google" else "gemini-1.5-flash-001"
        self.llm_models["google"] = ChatGoogleGenerativeAI(
            model=google_model,
            temperature=self.temperature,
            max_output_tokens=self.answer_max_tokens,
        )
        
        # Set default LLMs for different tasks
        self.hyde_llm = self.llm_models[self.hyde_llm_provider]  # For hypothetical answers
        self.answer_llm = self.llm_models[self.answer_llm_provider]  # For final answers

    def set_hyde_llm(self, llm_name: str, model_name: Optional[str] = None) -> bool:
        """
        Change the LLM used for generating hypothetical answers.
        
        Args:
            llm_name: Name of the LLM ("groq", "openai", or "google")
            model_name: Optional specific model name to use
            
        Returns:
            bool: Success status
        """
        if llm_name in self.llm_models:
            self.hyde_llm = self.llm_models[llm_name]
            
            # Update model if specified
            if model_name:
                try:
                    self.llm_models[llm_name].model = model_name
                    self.hyde_llm = self.llm_models[llm_name]
                except Exception as e:
                    self.logger.error(f"Failed to update model: {str(e)}")
                    return False
                    
            self.logger.info(f"HyDE LLM set to {llm_name}" + (f" with model {model_name}" if model_name else ""))
            return True
            
        self.logger.error(f"Invalid LLM name: {llm_name}")
        return False
        
    def set_answer_llm(self, llm_name: str, model_name: Optional[str] = None) -> bool:
        """
        Change the LLM used for generating final answers.
        
        Args:
            llm_name: Name of the LLM ("groq", "openai", or "google")
            model_name: Optional specific model name to use
            
        Returns:
            bool: Success status
        """
        if llm_name in self.llm_models:
            self.answer_llm = self.llm_models[llm_name]
            
            # Update model if specified
            if model_name:
                try:
                    self.llm_models[llm_name].model = model_name
                    self.answer_llm = self.llm_models[llm_name]
                except Exception as e:
                    self.logger.error(f"Failed to update model: {str(e)}")
                    return False
                    
            self.logger.info(f"Answer LLM set to {llm_name}" + (f" with model {model_name}" if model_name else ""))
            return True
            
        self.logger.error(f"Invalid LLM name: {llm_name}")
        return False
        
    def set_embeddings(self, provider: str, model_name: Optional[str] = None) -> bool:
        """
        Change the embeddings model.
        
        Args:
            provider: Provider name ("google" or "openai")
            model_name: Optional specific model name
            
        Returns:
            bool: Success status
        """
        try:
            if provider.lower() == "google":
                model = model_name or "models/embedding-001"
                self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
            elif provider.lower() == "openai":
                model = model_name or "text-embedding-3-large"
                self.embeddings = OpenAIEmbeddings(model=model)
            else:
                self.logger.error(f"Unknown embedding provider: {provider}")
                return False
                
            self.logger.info(f"Embeddings set to {provider}" + (f" with model {model_name}" if model_name else ""))
            # Note: Changing embeddings will require rebuilding vector stores
            self._vector_stores = {}
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update embeddings: {str(e)}")
            return False

    def _get_loader(self, data_path: str) -> BaseLoader:
        """Get appropriate document loader based on file type."""
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.pdf':
            return PyMuPDFLoader(data_path)
        elif file_ext == '.csv':
            return CSVLoader(data_path)
        elif file_ext == '.txt':
            return TextLoader(data_path)
        elif os.path.isdir(data_path):
            # Handle directory of files
            return DirectoryLoader(
                data_path, 
                glob="**/*.*",  # Load all files
                loader_cls=TextLoader  # Default to text loader
            )
        else:
            # Default to text loader for other file types
            self.logger.warning(f"No specific loader for {file_ext}, using TextLoader")
            return TextLoader(data_path)

    def load_data_and_create_retriever(self, 
                                      data_path: str,
                                      chunk_size: int = 512, 
                                      chunk_overlap: int = 128,
                                      retrieval_k: int = 5,
                                      force_reload: bool = False,
                                      text_splitter: str = "recursive",
                                      store_name: Optional[str] = None,
                                      persist_dir: Optional[str] = None,
                                      custom_loader: Optional[BaseLoader] = None,
                                      custom_embeddings: Optional[Embeddings] = None) -> bool:
        """
        Load data and create retriever.
        
        Args:
            data_path: Path to the data file or directory
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of similar documents to retrieve
            force_reload: Whether to force reload data even if it's already loaded
            text_splitter: Type of text splitter ("recursive" or "character")
            store_name: Custom name for vector store collection
            persist_dir: Custom directory for persisting vector store
            custom_loader: Optional custom document loader
            custom_embeddings: Optional custom embeddings model
            
        Returns:
            bool: Success status
        """
        # Check if we already have a retriever for this data path
        if data_path in HyDERAG._vector_stores and not force_reload:
            self.logger.info(f"Using existing vector store for {data_path}")
            self.retriever = HyDERAG._vector_stores[data_path].as_retriever(
                search_type="similarity", 
                search_kwargs={"k": retrieval_k}
            )
            HyDERAG._current_data_path = data_path
            return True
            
        try:
            self.logger.info(f"Loading data from {data_path}")
            
            # Extract collection and directory names
            file_name = os.path.splitext(os.path.basename(data_path))[0]
            collection_name = store_name or f"{file_name}-vectorstore"
            persist_directory = persist_dir or f"{file_name}-VDB"
            
            self.logger.info(f"Collection name: {collection_name}")
            self.logger.info(f"Persist directory: {persist_directory}")
            
            # Load documents with appropriate loader
            loader = custom_loader if custom_loader else self._get_loader(data_path)
            documents = loader.load()
            self.logger.info(f"Total documents/pages loaded: {len(documents)}")
            
            # Select text splitter
            if text_splitter.lower() == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
            else:
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separator="\n",
                )
            
            # Split documents
            splitted_documents = splitter.split_documents(documents)
            self.logger.info(f"Total document chunks created: {len(splitted_documents)}")
            
            # Create vector store and retriever
            embeddings_model = custom_embeddings or self.embeddings
            vector_store = Chroma.from_documents(
                documents=splitted_documents,
                collection_name=collection_name,
                embedding=embeddings_model, 
                persist_directory=persist_directory,
            )
            
            self.retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": retrieval_k},
            )
            
            # Store for future use
            HyDERAG._vector_stores[data_path] = vector_store
            HyDERAG._current_data_path = data_path
            
            self.logger.info("Chroma Retriever created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating retriever: {str(e)}")
            return False
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents into a context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _generate_hypothetical_answer(self, question: str, custom_template: Optional[str] = None) -> str:
        """
        Generate a hypothetical answer using the HyDE approach.
        
        Args:
            question: The user's question
            custom_template: Optional custom prompt template
        """
        hyde_template = custom_template or """You are an English expert mastering in creating hypothetical answers. 
        For the given user question generate a hypothetical answer. 
        Do not generate anything else just the answer. The question that you need to answer is: 
        Question: {user_question}
        """
        
        prompt = ChatPromptTemplate.from_template(hyde_template)
        formatted_prompt = prompt.format(user_question=question)
        
        self.logger.info("Generating hypothetical answer")
        hypothetical_answer = self.hyde_llm.invoke(formatted_prompt).content
        
        if self.verbose > 0:
            self.logger.info(f"Hypothetical answer: {hypothetical_answer}")
            
        return hypothetical_answer
    
    def answer_question(self, 
                       question: str, 
                       data_path: Optional[str] = None,
                       disable_hyde: bool = False,
                       hyde_template: Optional[str] = None, 
                       answer_template: Optional[str] = None,
                       retrieval_k: Optional[int] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> str:
        """
        Answer a question using the HyDE RAG pipeline.
        
        Args:
            question: The user's question
            data_path: Path to the data (optional if already loaded)
            disable_hyde: If True, use direct retrieval instead of HyDE
            hyde_template: Optional custom template for generating hypothetical answers
            answer_template: Optional custom template for generating final answers
            retrieval_k: Optional override for number of documents to retrieve
            temperature: Optional override for LLM temperature
            max_tokens: Optional override for max tokens in answer
            
        Returns:
            str: The answer to the question
        """
        # Ensure data is loaded
        if data_path and (data_path != HyDERAG._current_data_path or not hasattr(self, 'retriever')):
            success = self.load_data_and_create_retriever(data_path)
            if not success:
                return "Failed to load data. Please check the data path."
        
        if not hasattr(self, 'retriever'):
            return "No data loaded. Please load data first using load_data_and_create_retriever()."
        
        self.logger.info(f"Processing question: {question}")
        
        # Apply temporary parameter overrides if provided
        original_temp = None
        original_max_tokens = None
        
        if temperature is not None:
            original_temp = self.answer_llm.temperature
            self.answer_llm.temperature = temperature
            
        if max_tokens is not None:
            original_max_tokens = self.answer_llm.max_tokens
            self.answer_llm.max_tokens = max_tokens
            
        # Update retrieval parameters if specified
        search_k = self.retriever.search_kwargs.get("k", 5)
        if retrieval_k is not None:
            self.retriever.search_kwargs["k"] = retrieval_k
            
        try:
            # Retrieve relevant documents
            if disable_hyde:
                # Direct retrieval without HyDE
                self.logger.info("Using direct retrieval (HyDE disabled)")
                similar_documents = self.retriever.invoke(question)
            else:
                # HyDE approach
                self.logger.info("Using HyDE approach for retrieval")
                hypothetical_answer = self._generate_hypothetical_answer(question, hyde_template)
                similar_documents = self.retriever.invoke(hypothetical_answer)
            
            if self.verbose > 1:
                for idx, doc in enumerate(similar_documents):
                    self.logger.info(f"Document {idx+1}:")
                    self.logger.info(doc.page_content)
                    self.logger.info("="*50)
            
            # Format documents and create prompt
            context = self._format_documents(similar_documents)
            
            # Use custom or default answer template
            template = answer_template or """You are an excellent assistant. 
            Answer the following question in a detailed manner based on the below provided context: 

            Context:- {context}

            Question:- {question}

            Always remember to provide a complete answer for the question asked.
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            final_query = prompt.format(context=context, question=question)
            
            self.logger.info("Generating final answer")
            response = self.answer_llm.invoke(final_query).content
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"
            
        finally:
            # Restore original parameters if they were temporarily changed
            if original_temp is not None:
                self.answer_llm.temperature = original_temp
                
            if original_max_tokens is not None:
                self.answer_llm.max_tokens = original_max_tokens
                
            # Restore original retrieval k if it was changed
            if retrieval_k is not None:
                self.retriever.search_kwargs["k"] = search_k
    
    def compare_approaches(self, 
                          question: str, 
                          data_path: Optional[str] = None,
                          hyde_template: Optional[str] = None,
                          answer_template: Optional[str] = None,
                          retrieval_k: Optional[int] = None) -> Dict[str, str]:
        """
        Compare HyDE and direct retrieval approaches.
        
        Args:
            question: The user's question
            data_path: Path to the data (optional if already loaded)
            hyde_template: Optional custom template for HyDE
            answer_template: Optional custom template for answers
            retrieval_k: Optional number of documents to retrieve
            
        Returns:
            Dict: Containing both answers
        """
        hyde_answer = self.answer_question(
            question, 
            data_path, 
            disable_hyde=False, 
            hyde_template=hyde_template,
            answer_template=answer_template,
            retrieval_k=retrieval_k
        )
        
        direct_answer = self.answer_question(
            question, 
            data_path, 
            disable_hyde=True,
            answer_template=answer_template,
            retrieval_k=retrieval_k
        )
        
        return {
            "hyde_approach": hyde_answer,
            "direct_approach": direct_answer
        }


# # Example usage
# if __name__ == "__main__":
#     # Initialize the HyDE RAG system with verbose output
#     hyde_rag = HyDERAG(verbose=1)
    
#     # Load data and create retriever
#     hyde_rag.load_data_and_create_retriever(data_path="Data\\ReAct.pdf")
    
#     # Answer a question
#     question = "What is Chain of Thought prompting and how it is related to ReAct?"
#     answer = hyde_rag.answer_question(question)
    
#     print("\nFinal Answer:")
#     print(answer)
    
#     Example of changing LLMs
#     hyde_rag.set_hyde_llm("openai")
#     hyde_rag.set_answer_llm("openai")
    
#     Example of comparing approaches
#     results = hyde_rag.compare_approaches(question)
#     print("\nHyDE Approach Answer:")
#     print(results["hyde_approach"])
#     print("\nDirect Approach Answer:")
#     print(results["direct_approach"])