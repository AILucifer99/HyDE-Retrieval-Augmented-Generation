# Hypothetical Document Embedding Retrieval Augemnted Generation - Multi-LLM System

Hypothetical Document Embedding Retrieval-Augmented Generation (RAG) – Multi-LLM System
A scalable and modular GenAI framework that combines semantic document embedding with retrieval-augmented generation to enable context-rich, source-grounded responses across multiple domains. This system supports orchestration of multiple Large Language Models (LLMs) to optimize for accuracy, latency, and cost. Ideal for building intelligent assistants, enterprise knowledge bots, and hybrid QA systems. Features include custom chunking, vector search, multi-LLM routing, and configurable retrieval pipelines.

### Web Application Home Page
![HyDE RAG Application Home](https://github.com/AILucifer99/HyDE-Retrieval-Augmented-Generation/blob/main/architecture/Home.png)

### Conversation and Comparison 
![HyDE RAG Application Chat Comparsion](https://github.com/AILucifer99/HyDE-Retrieval-Augmented-Generation/blob/main/architecture/Comparison.png)

### Generation History
![HyDE RAG Application](https://github.com/AILucifer99/HyDE-Retrieval-Augmented-Generation/blob/main/architecture/Chat-History.png)

## Table of Contents
- [Overview](#overview)
- [How HyDERAG Works](#how-hyde-rag-works)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
  - [API Keys](#api-keys)
- [Application Structure](#application-structure)
- [Detailed Usage Guide](#detailed-usage-guide)
  - [Configuration Panel](#configuration-panel)
  - [Document Processing](#document-processing)
  - [Question Answering](#question-answering)
  - [History and Analysis](#history-and-analysis)
- [Advanced Configuration](#advanced-configuration)
  - [Embedding Models](#embedding-models)
  - [Language Models](#language-models)
  - [RAG Parameters](#rag-parameters)
- [Performance Optimization](#performance-optimization)
- [Extending The Application](#extending-the-application)
- [Troubleshooting](#troubleshooting)
- [Technical Background](#technical-background)
- [References](#references)
- [License](#license)

## Overview

The HyDERAG Streamlit Application is a powerful tool for document question-answering that implements the Hypothetical Document Embeddings (HyDE) approach to Retrieval-Augmented Generation (RAG). This application allows users to upload documents, configure sophisticated retrieval parameters, and ask questions using both traditional and HyDE-enhanced RAG techniques.

Unlike traditional RAG systems that directly embed a user's question for retrieval, HyDERAG first generates a hypothetical answer to the question, and then uses this richer semantic representation for document retrieval. This novel approach often yields superior results, especially for complex, nuanced questions that benefit from additional context during the retrieval phase.

### Key Features:

- **Multi-format Document Support**: Process PDF, TXT, CSV and directory-based document collections
- **Multi-provider Model Support**: Leverage models from OpenAI, Google, and Groq
- **Dual-approach Question Answering**: Compare traditional and HyDE approaches side-by-side  
- **Comprehensive Configuration**: Fine-tune every aspect of the RAG pipeline
- **Performance Analysis**: Track and compare response times and quality
- **Interactive History**: Review all previous questions and their answers

## How HyDE RAG Works

HyDERAG implements a sophisticated enhancement to traditional RAG pipelines:

1. **Traditional RAG Pipeline**:
   - User question → Embedding → Vector search → Retrieved documents → LLM answer

2. **HyDE Enhanced Pipeline**:
   - User question → **Hypothetical answer generation** → Embedding of hypothetical answer → Vector search → Retrieved documents → LLM answer

The key insight is that a hypothetical answer often contains much richer semantic information than the original question, leading to more accurate document retrieval. This is especially valuable for:

- Complex questions requiring nuanced understanding
- Questions where relevant documents might use different terminology than the question
- Situations where direct keyword matching would fail

## Installation

### Prerequisites

- Python 3.9+ installed
- Basic understanding of LLMs and RAG systems
- API keys for at least one of: OpenAI, Google AI, Groq

### Setup Steps

1. **Clone/download the repository**:
   ```bash
   git clone https://your-repository-url/hyderag-app.git
   cd hyderag-app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### API Keys

The application supports multiple LLM providers. You'll need API keys for the services you intend to use:

- **OpenAI API Key**: Required for OpenAI embeddings and LLMs
  - Sign up at [https://platform.openai.com/signup](https://platform.openai.com/signup)
  - Create an API key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

- **Google AI Key**: Required for Google embeddings and LLMs
  - Create an API key at [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

- **Groq API Key**: Required for Groq LLMs
  - Sign up at [https://console.groq.com/signup](https://console.groq.com/signup)
  - Create an API key in your Groq dashboard

Add these keys to your `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

## Application Structure

The application consists of these key files:

- `app.py`: The main Streamlit application
- `hyDERAGPipeline.py`: The core HyDERAG implementation
- `requirements.txt`: List of required Python packages
- `.env`: Environment variables file (API keys)

Directory structure after running the app:
```
hyderag-app/
├── app.py                 # Main Streamlit application
├── hyDERAGPipeline.py     # Core HyDERAG implementation
├── requirements.txt       # Dependencies
├── .env                   # API keys (create from .env.example)
├── temp_uploads/          # Created automatically for uploaded files
└── *-VDB/                 # Vector database directories (created per document)
```

## Detailed Usage Guide

### Configuration Panel

The sidebar contains all configuration options organized into expandable sections:

#### API Keys
- Enter your API keys for the services you plan to use
- Click "Save API Keys" to update the environment variables

#### Model Configuration
1. **Embeddings**:
   - Select provider (Google or OpenAI)
   - Specify the embedding model name
   - Google default: `models/embedding-001`
   - OpenAI default: `text-embedding-3-large`

2. **HyDE LLM**:
   - Select provider for hypothetical answer generation
   - Specify model name (provider-specific)
   - Groq default: `gemma2-9b-it`
   - OpenAI default: `gpt-4o`
   - Google default: `gemini-1.5-flash-001`

3. **Answer LLM**:
   - Select provider for final answer generation
   - Specify model name (provider-specific)
   - Defaults follow same pattern as HyDE LLM

#### RAG Configuration
- Text Splitter: Choose between recursive or character-based splitting
- Chunk Size: Size of document chunks (128-2048)
- Chunk Overlap: Overlap between chunks (0-512)
- Retrieval K: Number of documents to retrieve (1-10)

#### LLM Parameters
- Temperature: Controls randomness (0.0-1.0)
- HyDE Max Tokens: Maximum length for hypothetical answers
- Answer Max Tokens: Maximum length for final answers

After configuring, click "Initialize HyDERAG" to apply your settings.

### Document Processing

In the "Document Processing" tab:

1. Upload a document (PDF, TXT, or CSV)
2. Click "Process Document" to chunk the document and create embeddings
3. Wait for confirmation that processing is complete

The system will:
- Save the uploaded file to a temporary directory
- Split the document into chunks based on your configuration
- Create embeddings for each chunk
- Build a vector store for similarity search

### Question Answering

In the "Question Answering" tab:

1. Enter your question in the text area
2. Choose one of three options:
   - **Ask with HyDE**: Uses the HyDE approach
   - **Ask without HyDE**: Uses traditional direct RAG
   - **Compare Approaches**: Runs both methods and displays results side-by-side

The system will display:
- The generated answer(s)
- Response generation time(s)

### History and Analysis

In the "History" tab:

1. View a table of all questions asked during the session
2. Select any entry to view the complete question and answer(s)
3. Compare performance metrics between approaches

This history tracking allows you to:
- Review previous questions and answers
- Compare the quality of HyDE vs. direct RAG responses
- Analyze performance differences across question types

## Advanced Configuration

### Embedding Models

#### Google AI Embeddings
- `models/embedding-001`: General purpose embedding model (default)
- `models/embedding-002`: Newer model with improved performance

#### OpenAI Embeddings
- `text-embedding-3-large`: High-performance, latest model (default)
- `text-embedding-3-small`: Faster, more cost-effective model
- `text-embedding-ada-002`: Legacy model (not recommended for new projects)

### Language Models

#### Groq Models
- `gemma2-9b-it`: Balanced performance and speed
- `llama3-70b-8192`: High-performance for complex tasks
- `mixtral-8x7b-32768`: Strong multi-lingual performance

#### OpenAI Models
- `gpt-4o`: Latest high-performance model 
- `gpt-4-turbo`: Good balance of performance and cost
- `gpt-3.5-turbo`: Faster, more cost-effective option

#### Google Models
- `gemini-1.5-flash-001`: Fast response time
- `gemini-1.5-pro-001`: Higher quality for complex tasks

### RAG Parameters

Fine-tuning these parameters can significantly impact performance:

#### Text Splitting
- **Recursive splitter**: Attempts to keep semantic units together (paragraphs, sentences)
- **Character splitter**: Simpler approach that splits purely by character count

#### Chunk Size
- **Smaller chunks** (128-512): More precise retrieval but less context
- **Larger chunks** (512-2048): More context but potentially less precise retrieval

#### Chunk Overlap
- **Low overlap** (0-64): Efficient storage, potential context gaps
- **High overlap** (128-256): Better context preservation, larger vector store

#### Retrieval K
- **Lower values** (1-3): Focused context, faster processing
- **Higher values** (5-10): More comprehensive context, potential noise

## Performance Optimization

For optimal performance:

1. **Document Processing**:
   - Use recursive text splitter for most documents
   - Balance chunk size with the nature of your documents (larger for conceptual content, smaller for factoid-heavy content)
   - Set chunk overlap to roughly 10-20% of chunk size

2. **Model Selection**:
   - HyDE LLM: Fast models work well here as the hypothetical answer doesn't need to be perfect
   - Answer LLM: Higher quality models typically produce better final answers

3. **Response Time vs Quality**:
   - For faster responses: Use smaller, faster models and lower retrieval K
   - For higher quality: Use more advanced models and tune retrieval parameters

## Extending The Application

The HyDERAG application can be extended in several ways:

1. **Adding Document Types**:
   - Implement new loader classes in `hyDERAGPipeline.py`
   - Update the file type handling in the Streamlit interface

2. **Supporting Additional Models**:
   - Add new model providers in the `_initialize_models` method
   - Update the UI to expose these new options

3. **Custom Prompting**:
   - Modify the application to allow user-defined prompts for both HyDE and answer generation

4. **Persistent History**:
   - Implement database storage for question/answer history
   - Add analytics dashboards for long-term performance tracking

## Troubleshooting

### Common Issues and Solutions

#### API Key Errors
- **Error**: "API key not available" or authentication errors
- **Solution**: Check the API keys in your `.env` file and ensure they are correctly formatted

#### Document Processing Errors
- **Error**: "Failed to load document" or "Error creating retriever"
- **Solution**: 
  1. Check the file format is supported
  2. Ensure the file is not corrupted or password-protected
  3. Try with a smaller or simpler document first

#### Memory Issues
- **Error**: "Memory error" or application crash with large documents
- **Solution**:
  1. Reduce chunk size
  2. Process smaller documents
  3. Run the application on a machine with more RAM

#### Model-Specific Errors
- **Error**: Errors mentioning specific models
- **Solution**:
  1. Verify the model name is correct
  2. Check if your API key has access to the specified model
  3. Try a different model from the same provider

#### Vector Store Errors
- **Error**: "Error with vector store" or ChromaDB errors
- **Solution**:
  1. Delete the generated vector store directories and try again
  2. Check disk space availability
  3. Ensure you have permission to write to the directories

## Technical Background

### The Science Behind HyDE

Hypothetical Document Embeddings (HyDE) was introduced in the paper ["Precise Zero-Shot Dense Retrieval without Relevance Labels"](https://arxiv.org/abs/2212.10496) by Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Nicola De Cao.

The key insight of HyDE is that a hypothetical answer contains richer semantic information than the original question. This approach addresses several limitations of traditional RAG:

1. **Vocabulary Mismatch**: Questions often use different terminology than documents
2. **Intent Expansion**: A hypothetical answer expands the query with related concepts
3. **Context Enhancement**: The answer provides additional semantic context

The implementation in this application follows the paper's approach with practical optimizations for real-world use.

### Vector Search Methodology

The application uses ChromaDB as the vector database which implements:

1. **Approximate Nearest Neighbor (ANN) Search**: Finding similar vectors efficiently
2. **Cosine Similarity**: Measuring the angle between vectors to determine semantic similarity
3. **Dimension Reduction**: Managing high-dimensional embedding spaces effectively

## References

- Izacard, G., et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Langchain Documentation: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ by AILucifer**
