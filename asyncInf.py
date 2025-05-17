import asyncio
from asynchronous import asyncHyDERAGPipeline 


async def main():
    # Step 1: Initialize the AsyncHyDERAG system with verbose output enabled
    hyde_rag = asyncHyDERAGPipeline.AsyncHyDERAG(verbose=1)

    # Step 2: Load data from a PDF file and build the document retriever
    # Note: This step is synchronous because the underlying Chroma library is not async-compatible yet
    success = hyde_rag.load_data_and_create_retriever("Data\\ReAct.pdf")

    # Check if documents were loaded successfully; if not, exit early
    if not success:
        print("Failed to load documents.")
        return

    # Step 3: Define a question to ask asynchronously
    question = "What is Chain of Thought prompting and how it is related to ReAct?"

    # Await the asynchronous answer generation method of the HyDERAG system
    answer = await hyde_rag.answer_question(question)

    # Step 4: Output the final answer to the console
    print("\nFinal Answer:")
    print(answer)


# Entry point of the script to run the async event loop
if __name__ == "__main__":
    import sys

    # On Windows, set the appropriate event loop policy for asyncio to avoid runtime errors
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main coroutine within the asyncio event loop
    asyncio.run(main())
