"""
Generation module for the Financial RAG Agent.

This module configures the Llama-3-8B-Instruct LLM via llama-cpp-python
with parameters optimized for RTX 2080 Super (8GB VRAM).
"""

import os
from pathlib import Path

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


def create_llm(
    model_path: str = None,
    n_ctx: int = 2048,
    temperature: float = 0.1,
    n_gpu_layers: int = -1,
):
    """
    Creates a LlamaCpp instance optimized for RTX 2080 Super.

    Args:
        model_path: Path to the .gguf model file (if None, uses default from project root)
        n_ctx: Context size (limited to 2048-4096 to save VRAM)
        temperature: Generation temperature (0.1 for precise answers)
        n_gpu_layers: Number of layers to load on GPU (-1 = all)

    Returns:
        Configured LlamaCpp instance
    """
    if model_path is None:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "meta-llama-3-8b-instruct.Q4_K_M.gguf"
        model_path = str(model_path)

    if not os.path.isabs(model_path):
        project_root = Path(__file__).parent.parent
        model_path = str(project_root / model_path.lstrip("./"))

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Please download the model and place it in the models/ directory."
        )

    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        temperature=temperature,
        n_gpu_layers=n_gpu_layers,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        verbose=False,
    )


def create_financial_qa_prompt():
    """
    Creates a prompt template for financial questions.

    Returns:
        Configured PromptTemplate
    """
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial assistant. Answer questions based on the provided context. Be precise and only use information from the context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "query"],
    )


def create_qa_chain(llm, prompt_template):
    """
    Creates an LLM chain for Q&A.

    Args:
        llm: Configured LLM instance
        prompt_template: Prompt template

    Returns:
        Runnable chain (LLM + prompt)
    """
    return prompt_template | llm


if __name__ == "__main__":
    llm = None
    try:
        llm = create_llm()
        prompt_template = create_financial_qa_prompt()
        chain = create_qa_chain(llm, prompt_template)
        result = chain.invoke(
            {
                "context": "Net Profit Margin (%): 3.97 ... Asset Turnover Ratio (%): 11.04 ...",
                "query": "What is the Asset Turnover Ratio of the company?",
            }
        )
        print(result)
    finally:
        if llm is not None:
            try:
                if hasattr(llm, "client") and llm.client is not None:
                    llm.client.close()
            except Exception:
                pass
