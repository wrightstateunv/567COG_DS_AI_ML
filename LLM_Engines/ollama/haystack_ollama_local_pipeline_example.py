# -----------------------------------------------------------------------------
# haystack_ollama_local_pipeline_example.py
#
# Example: Local semantic search and LLM QA pipeline using Haystack 2.x
# with Ollama for both embeddings and LLM completions. No cloud APIs required.
#
# Features:
# - Embeddings: Uses Ollama's embedding model (e.g., nomic-embed-text)
# - LLM: Uses Ollama's LLM (e.g., llama2) for prompt completion
# - Document store: In-memory (for demo; use FAISS or other for persistence)
# - All inference is local; no data leaves your machine
# - Supports ingestion of both hardcoded text and PDF files
#
# Usage:
#   1. Start Ollama: `ollama serve` (and pull required models)
#   2. Run this script: `python haystack_ollama_local_pipeline_example.py`
#   3. Edit the docs, query, or models as needed for your use case
#
# Requirements:
#   - haystack-ai >=2.x
#   - pypdf (for PDF ingestion)
#   - requests
#   - Ollama running locally with required models pulled
#
# Example PDF usage:
#   run_ollama_pipeline_with_pdfs(["/path/to/file1.pdf", "/path/to/file2.pdf"])
# -----------------------------------------------------------------------------
from haystack.components.converters import PyPDFToDocument
try:
    from haystack.utils import TextSplitter
except ImportError:
    TextSplitter = None
# Haystack 2.x (haystack-ai) minimal test for LLM connection (Ollama integration)
from haystack import Pipeline
#from haystack.pipelines import Pipeline
from haystack.components.builders import PromptBuilder
import requests
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument
import os

# --- Core embedding and LLM call utilities ---

def get_ollama_embedding(text, model="nomic-embed-text"):
    """
    Get an embedding vector for the given text using Ollama's embedding model.
    Args:
        text (str): The input text to embed.
        model (str): The Ollama embedding model to use (default: "nomic-embed-text").
    Returns:
        list[float]: The embedding vector.
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]


def call_ollama_llm(prompt, model="llama2"):
    """
    Call Ollama's LLM to generate a response for the given prompt.
    Args:
        prompt (str): The prompt/question to send to the LLM.
        model (str): The Ollama LLM model to use (default: "llama2").
    Returns:
        str: The LLM's response.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json().get("response", "[No response]")

# --- Example 1: Minimal pipeline with hardcoded docs ---

def run_minimal_ollama_pipeline():
    """
    Minimal example pipeline: Ingests two hardcoded text documents, embeds them with Ollama,
    retrieves the most relevant doc for a sample query, and gets an answer from Ollama LLM.
    """
    docs = [
        Document(content="Sachin Tendulkar is a famous Indian cricketer, he has 2 kids: arjun and sara. "
        "His best friend is Vinod Kambli. "
        "His Sharajah knock desert storm is still etched in memory."),
        Document(content="Llama2 is a large language model available in Ollama.")
    ]
    embedding_model = "nomic-embed-text"
    for doc in docs:
        doc.embedding = get_ollama_embedding(doc.content, model=embedding_model)
    store = InMemoryDocumentStore()
    store.write_documents(docs)
    retriever = InMemoryEmbeddingRetriever(document_store=store)
    prompt_builder = PromptBuilder(template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}")
    llm_model = "llama2"
    query = "Who is Sachin Tendulkar?"
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=1)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)

# --- Example 2: Pipeline with PDF input ---

def run_ollama_pipeline_with_pdfs(pdf_paths):
    """
    Example pipeline: Ingests a list of PDF files, converts them to Documents, embeds with Ollama,
    retrieves the most relevant docs for a sample query, and gets an answer from Ollama LLM.
    Args:
        pdf_paths (list[str]): List of file paths to PDF files.
    """
    converter = PyPDFToDocument()
    docs = converter.run(sources=pdf_paths)["documents"]
    embedding_model = "nomic-embed-text"
    for doc in docs:
        doc.embedding = get_ollama_embedding(doc.content, model=embedding_model)
    store = InMemoryDocumentStore()
    store.write_documents(docs)
    retriever = InMemoryEmbeddingRetriever(document_store=store)
    llm_model = "llama2"
    query = "Summarize the main topic of the provided PDFs."
    prompt_builder = PromptBuilder(template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}")
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=3)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)


# --- Example 3:
def run_ollama_pipeline_with_pdfs_and_persist(pdf_paths, persist_path):
    """
    Ingests a list of PDF files, embeds them with Ollama, stores them in a persistent document store,
    and saves the store to disk as a JSON file.
    Args:
        pdf_paths (list[str]): List of file paths to PDF files.
        persist_path (str): Path to save the persistent document store (e.g., './store/embedding/my_store.json').
    """
    converter = PyPDFToDocument()
    docs = converter.run(sources=pdf_paths)["documents"]
    embedding_model = "nomic-embed-text"
    for doc in docs:
        doc.embedding = get_ollama_embedding(doc.content, model=embedding_model)
    store = InMemoryDocumentStore()
    store.write_documents(docs)
    # Persist the document store to disk
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    store.save_to_disk(persist_path)
    print(f"Document store with embeddings saved to: {persist_path}")

    # Reload the store and run a retrieval + LLM prompt
    loaded_store = InMemoryDocumentStore.load_from_disk(persist_path)
    retriever = InMemoryEmbeddingRetriever(document_store=loaded_store)
    llm_model = "llama2"
    query = "What is Virat's last name."
    prompt_builder = PromptBuilder(template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}")
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=3)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)


def run_ollama_pipeline_with_pdfs_chunked_and_persist(pdf_paths, persist_path, chunk_size=512, chunk_overlap=50):
    """
    Ingests a list of PDF files, splits them into smaller chunks, embeds them with Ollama, stores them in a persistent document store,
    and saves the store to disk as a JSON file. Also runs a retrieval and LLM prompt from the persisted store.
    Args:
        pdf_paths (list[str]): List of file paths to PDF files.
        persist_path (str): Path to save the persistent document store (e.g., './store/embedding/my_store.json').
        chunk_size (int): Number of characters per chunk (default: 512).
        chunk_overlap (int): Number of overlapping characters between chunks (default: 50).
    """
    converter = PyPDFToDocument()
    docs = converter.run(sources=pdf_paths)["documents"]
    # Split documents into smaller chunks and save each chunk as a file
    chunk_dir = os.path.join(os.path.dirname(persist_path), "../chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunked_docs = []
    chunk_files = []
    if TextSplitter is not None:
        splitter = TextSplitter(
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
            split_respect_sentence_boundary=True
        )
        chunked_docs = splitter.split_documents(docs)
    else:
        # Fallback: simple manual chunking by words
        for doc in docs:
            words = doc.content.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk = " ".join(words[i:i+chunk_size])
                chunked_docs.append(Document(content=chunk, meta=doc.meta))

    # Save each chunk as a file
    for idx, doc in enumerate(chunked_docs):
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx+1}.txt")
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(doc.content)
        chunk_files.append(chunk_path)

    # Load chunks from files as Documents
    loaded_chunk_docs = []
    for chunk_path in chunk_files:
        with open(chunk_path, "r", encoding="utf-8") as f:
            content = f.read()
        loaded_chunk_docs.append(Document(content=content))
    embedding_model = "nomic-embed-text"
    for doc in loaded_chunk_docs:
        doc.embedding = get_ollama_embedding(doc.content, model=embedding_model)
    store = InMemoryDocumentStore()
    store.write_documents(loaded_chunk_docs)
    # Persist the document store to disk
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    store.save_to_disk(persist_path)
    print(f"Chunked document store with embeddings saved to: {persist_path}")

    # Reload the store and run a retrieval + LLM prompt
    loaded_store = InMemoryDocumentStore.load_from_disk(persist_path)
    retriever = InMemoryEmbeddingRetriever(document_store=loaded_store)
    llm_model = "llama2"
    query = "Fastest player."
    prompt_builder = PromptBuilder(template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}")
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=3)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)


# --- Example usage ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Haystack+Ollama local pipeline example.")
    parser.add_argument(
        "--mode",
        choices=["minimal", "pdf", "pdf_persist", "pdf_chunked_persist"],
        default="pdf_chunked_persist",
        help="Which pipeline function to run: minimal (hardcoded docs), pdf (PDF input), pdf_persist (PDF input with persistent embedding store), or pdf_chunked_persist (PDF input, chunked, with persistent embedding store)."
    )
    parser.add_argument(
        "--persist_path",
        type=str,
        default="/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/store/embedding/indian_cricketers_store.json",
        help="Path to save persistent embedding store (used only with --mode pdf_persist)."
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=[
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/Indian_Cricketers_Statistics.pdf",
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/Indian_Cricketers_Records.pdf",
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/Indian_Cricketers_Family.pdf",
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/Indian_Cricketers_Achievements.pdf"
        ],
        help="List of PDF file paths to use as input (used with --mode pdf or pdf_persist)."
    )
    args = parser.parse_args()

    if args.mode == "minimal":
        print("--- Running minimal Ollama pipeline example (hardcoded docs) ---")
        run_minimal_ollama_pipeline()
    elif args.mode == "pdf":
        print("--- Running Ollama pipeline with PDF input ---")
        run_ollama_pipeline_with_pdfs(args.pdfs)
    elif args.mode == "pdf_persist":
        print("--- Running Ollama pipeline with PDF input and persistent embedding store ---")
        run_ollama_pipeline_with_pdfs_and_persist(args.pdfs, args.persist_path)
    elif args.mode == "pdf_chunked_persist":
        print("--- Running Ollama pipeline with PDF input, chunked, and persistent embedding store ---")
        run_ollama_pipeline_with_pdfs_chunked_and_persist(args.pdfs, args.persist_path)
