import os
from dotenv import load_dotenv
from hydeRAG import hyDERAGPipeline # Import our HyDERAG class


# Load environment variables (API keys)
load_dotenv()


def main():
    """
    Example of using the HyDERAG class with custom parameters
    """
    # Initialize HyDERAG with custom parameters
    hyde_rag = hyDERAGPipeline.HyDERAG(
        verbose=1,  # Enable detailed logging
        embedding_provider="google",  # Use Google embeddings (options: "google", "openai")
        embedding_model="models/embedding-001",  # Specify embedding model
        hyde_llm_provider="groq",  # Use Groq for hypothetical document generation
        hyde_llm_model="gemma2-9b-it",  # Specify HyDE model
        answer_llm_provider="google",  # Use Google for answer generation
        answer_llm_model="gemini-1.5-flash-001",  # Specify answer model
        temperature=0.2,  # Lower temperature for more focused responses
        hyde_max_tokens=512,  # Maximum tokens for hypothetical answers
        answer_max_tokens=1024  # Maximum tokens for final answers
    )
    
    # Load data and create retriever with custom parameters
    data_path = "Data\\ReAct.pdf"  # Replace with your document path
    success = hyde_rag.load_data_and_create_retriever(
        data_path=data_path,
        chunk_size=800,  # Larger chunks for more context
        chunk_overlap=150,  # Overlap between chunks
        retrieval_k=4,  # Number of similar documents to retrieve
        text_splitter="recursive",  # Use recursive character splitter
        store_name="react-vecstore",  # Custom vector store name
        persist_dir="react-db-directory"  # Custom persistence directory
    )
    
    if not success:
        print("Failed to load data. Please check the data path.")
        return
    
    # Example 1: Basic question answering
    question = "What are the key concepts mentioned in the document?"
    answer = hyde_rag.answer_question(question)
    print("\nBasic Answer:")
    print(answer)
    
    # Example 2: Custom templates for both HyDE and answer generation
    custom_hyde_template = """You are an expert in generating hypothetical answers.
    Please create a well-informed, detailed response that would answer the following question:
    Question: {user_question}
    
    Provide an answer that covers the core concepts, key points, and any relevant details.
    """
    
    custom_answer_template = """You are a knowledgeable assistant with expertise in the subject matter.
    
    Context: {context}
    
    Question: {question}
    
    Based solely on the provided context, answer the question comprehensively. 
    Structure your response with clear sections and explain any technical concepts. 
    If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
    """
    
    # Ask with custom templates and parameters
    question = "What are the applications and limitations of this technology?"
    custom_answer = hyde_rag.answer_question(
        question=question,
        disable_hyde=False,  # Use HyDE approach (default)
        hyde_template=custom_hyde_template,
        answer_template=custom_answer_template,
        retrieval_k=6,  # Retrieve more documents for this complex question
        temperature=0.4,  # Slightly more creative
    )
    
    print("\nCustomized Answer:")
    print(custom_answer)
    
    # Example 3: Compare HyDE vs direct retrieval approaches
    question = "What are the advantages described in the document?"
    comparison_results = hyde_rag.compare_approaches(
        question=question,
        hyde_template=custom_hyde_template,
        answer_template=custom_answer_template,
        retrieval_k=5
    )
    
    print("\nHyDE Approach Answer:")
    print(comparison_results["hyde_approach"])
    print("\nDirect Approach Answer:")
    print(comparison_results["direct_approach"])
    
    # Example 4: Changing models mid-session
    print("\nChanging models...")
    hyde_rag.set_hyde_llm("openai", "gpt-4o")
    hyde_rag.set_answer_llm("openai", "gpt-4o")
    
    question = "Summarize the main points in the document"
    openai_answer = hyde_rag.answer_question(question)
    
    print("\nOpenAI Model Answer:")
    print(openai_answer)


if __name__ == "__main__":
    main()