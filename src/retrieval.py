"""
Retrieval and reranking module for the Financial RAG Agent.

This module implements the retrieval chain with reranking:
1. Vector retrieval (top 20)
2. Reranking with FlashRank (top 5)
"""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_retriever(vector_store_path: str = "./chroma_db", top_k: int = 20):
    """
    Creates a vector retriever from ChromaDB.

    Args:
        vector_store_path: Path to the ChromaDB vector store
        top_k: Number of documents to retrieve initially

    Returns:
        Configured retriever
    """

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return retriever


def create_reranked_retriever(base_retriever, top_n: int = 5):
    """
    Adds a FlashRank reranker to the base retriever.

    Args:
        base_retriever: Base vector retriever
        top_n: Final number of documents after reranking

    Returns:
        ContextualCompressionRetriever with reranking
    """
    compressor = FlashrankRerank(top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return compression_retriever


def retrieve_documents(query: str, retriever, top_n: int = 5) -> list:
    """
    Retrieves and reranks relevant documents for a query.

    Args:
        query: User query
        retriever: Configured retriever (with or without reranking)
        top_n: Number of documents to return

    Returns:
        List of relevant documents
    """
    return retriever.invoke(query)


if __name__ == "__main__":
    retriever = create_retriever()
    reranked_retriever = create_reranked_retriever(retriever)
    documents = retrieve_documents(
        "What is the Asset Turnover Ratio of the company?", reranked_retriever
    )
    print(documents)
