"""
Data ingestion module for the Financial RAG Agent.

This module loads the FinanceQA dataset from Hugging Face,
converts it to LangChain Document format, and indexes it into ChromaDB.
"""

from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_finance_dataset():
    """
    Loads the sweatSmile/FinanceQA dataset from Hugging Face.

    {
        "COMPANY_ID": "ICICIBANK_2023_converted.txt_2",
        "QUERY": "What is the Asset Turnover Ratio of the company?",
        "ANSWER": "Asset Turnover Ratio is 11.04%.",
        "CONTEXT": "Net Profit Margin (%): 3,729,625,427.97 ... Asset Turnover Ratio (%): 11.04 ..."
    }

    Returns:
        List[Document]: List of LangChain documents with metadata
    """
    dataset = load_dataset("sweatSmile/FinanceQA", split="train")
    documents = []
    for row in dataset:
        document = Document(
            page_content=row["CONTEXT"],
            metadata={
                "company_id": row["COMPANY_ID"],
                "query": row["QUERY"],
                "answer": row["ANSWER"],
            },
        )
        documents.append(document)
    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Splits documents into chunks with overlap.

    Args:
        documents: List of LangChain documents
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        List[Document]: Documents split into chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_vector_store(documents: list[Document], persist_directory: str = "./chroma_db"):
    """
    Creates a ChromaDB vector store with embedded documents.

    Args:
        documents: List of documents to index
        persist_directory: ChromaDB persistence directory
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectorstore.add_documents(documents)
    return vectorstore


if __name__ == "__main__":
    documents = load_finance_dataset()
    chunked_documents = chunk_documents(documents)
    vectorstore = create_vector_store(chunked_documents)
    print(vectorstore)
