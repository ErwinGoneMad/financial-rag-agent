"""
Complete RAG pipeline for the Financial RAG Agent.

This module combines retrieval, reranking, and generation
to create an end-to-end RAG system.
"""

from src.generation import create_financial_qa_prompt, create_llm, create_qa_chain
from src.retrieval import create_reranked_retriever, create_retriever


class FinancialRAGPipeline:
    """
    Complete RAG pipeline for financial questions.
    """

    def __init__(
        self,
        vector_store_path: str = "./chroma_db",
        model_path: str = "./models/meta-llama-3-8b-instruct.Q4_K_M.gguf",
        use_reranking: bool = True,
        top_k_retrieve: int = 20,
        top_n_final: int = 5,
    ):
        """
        Initializes the RAG pipeline.

        Args:
            vector_store_path: Path to the ChromaDB vector store
            model_path: Path to the .gguf model
            use_reranking: Use FlashRank reranking
            top_k_retrieve: Number of documents to retrieve initially
            top_n_final: Final number of documents after reranking
        """
        self.retriever = create_retriever(vector_store_path, top_k_retrieve)
        if use_reranking:
            self.retriever = create_reranked_retriever(self.retriever, top_n_final)
        self.llm = create_llm(model_path)
        self.prompt_template = create_financial_qa_prompt()
        self.chain = create_qa_chain(self.llm, self.prompt_template)

    def query(self, question: str) -> dict:
        """
        Processes a question and returns an answer.

        Args:
            question: User question

        Returns:
            Dictionary with 'answer' and 'source_documents'
        """
        documents = self.retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in documents)

        answer = self.chain.invoke({"context": context, "query": question})

        return {
            "answer": answer,
            "source_documents": documents,
        }

    def query_with_sources(self, question: str) -> dict:
        """
        Processes a question and returns answer with formatted sources.

        Args:
            question: User question

        Returns:
            Dictionary with answer and formatted sources
        """
        result = self.query(question)
        return {
            "answer": result["answer"],
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result["source_documents"]
            ],
        }


def create_pipeline(
    vector_store_path: str = "./chroma_db",
    model_path: str = "./models/meta-llama-3-8b-instruct.Q4_K_M.gguf",
) -> FinancialRAGPipeline:
    """
    Factory function to create a configured RAG pipeline.

    Args:
        vector_store_path: Path to the vector store
        model_path: Path to the model

    Returns:
        Configured FinancialRAGPipeline instance
    """
    return FinancialRAGPipeline(vector_store_path, model_path)


if __name__ == "__main__":
    pipeline = create_pipeline()
    result = pipeline.query("What is the Asset Turnover Ratio of the company?")
    print(result)
    result = pipeline.query_with_sources("What is the Asset Turnover Ratio of the company?")
    print(result)
