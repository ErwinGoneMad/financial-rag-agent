"""
Streamlit interface for the Financial RAG Agent.

This application allows interaction with the RAG system via a web interface.
"""

import streamlit as st

from src.rag_pipeline import create_pipeline

st.set_page_config(
    page_title="Financial Insight RAG Agent",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Financial Insight RAG Agent")
st.markdown(
    """
    Ask questions about financial data extracted from the FinanceQA dataset.
    The system uses RAG (Retrieval-Augmented Generation) with Llama-3-8B-Instruct.
    """
)


@st.cache_resource
def load_pipeline():
    """Loads the RAG pipeline (cached by Streamlit)."""
    try:
        pipeline = create_pipeline()
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None


with st.sidebar:
    st.header("Configuration")
    st.markdown("### RAG Pipeline")
    st.info(
        """
        **Composants:**
        - LLM: Llama-3-8B-Instruct (4-bit quantized)
        - Embeddings: bge-small-en-v1.5
        - Reranker: FlashRank
        - Vector DB: ChromaDB
        """
    )

pipeline = load_pipeline()

if pipeline is None:
    st.error("Failed to load the RAG pipeline. Please check the configuration and try again.")
    st.stop()

st.markdown("---")
query = st.text_input(
    "Ask a question about financial data:",
    placeholder="e.g., What is the Asset Turnover Ratio of the company?",
    key="user_query",
)

if query:
    with st.spinner("Processing your question..."):
        try:
            result = pipeline.query_with_sources(query)

            st.markdown("### Answer")
            st.write(result["answer"])

            if result.get("sources"):
                st.markdown("---")
                st.markdown("### Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i}"):
                        st.markdown("**Content:**")
                        st.text(source["content"])
                        if source.get("metadata"):
                            st.markdown("**Metadata:**")
                            st.json(source["metadata"])
            else:
                st.info("No sources found for this query.")

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Enter a question above to get started!")
