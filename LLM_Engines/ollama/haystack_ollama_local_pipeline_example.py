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
#   2. Pull required models: `ollama pull llama2` and `ollama pull nomic-embed-text`
#   3. Choose your mode and run the script:
#
# BASIC MODES:
#   python haystack_ollama_local_pipeline_example.py --mode minimal
#   python haystack_ollama_local_pipeline_example.py --mode pdf --pdfs path/to/file.pdf
#
# INTEL DATA PROCESSING:
#   # First, process intel data (creates embeddings - run once):
#   python haystack_ollama_local_pipeline_example.py --mode intel_chunked_persist
#
#   # Then, ask questions using processed data (fast):
#   python haystack_ollama_local_pipeline_example.py --mode intel_qa --question "What are the main APT groups?"
#
#   # Or start interactive session:
#   python haystack_ollama_local_pipeline_example.py --mode intel_interactive
#
# CRICKET DATA PROCESSING:
#   python haystack_ollama_local_pipeline_example.py --mode pdf_chunked_persist
#
# Requirements:
#   - haystack-ai >=2.x
#   - pypdf (for PDF ingestion)
#   - requests
#   - reportlab (for PDF generation)
#   - Ollama running locally with required models pulled
#
# Models needed:
#   - ollama pull llama2 (for text generation)
#   - ollama pull nomic-embed-text (for embeddings)
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
import glob
import csv
import time

# --- Core embedding and LLM call utilities ---

def estimate_tokens(text):
    """
    Estimate token count for a given text.
    This is a rough approximation: tokens ≈ words * 1.3 for English text.
    Args:
        text (str): The input text.
    Returns:
        int: Estimated number of tokens.
    """
    if not text:
        return 0
    # Simple estimation: split by whitespace and multiply by 1.3
    words = len(text.split())
    return int(words * 1.3)

def get_ollama_embedding(text, model="nomic-embed-text", max_retries=3, show_tokens=False):
    """
    Get an embedding vector for the given text using Ollama's embedding model.
    Args:
        text (str): The input text to embed.
        model (str): The Ollama embedding model to use (default: "nomic-embed-text").
        max_retries (int): Maximum number of retry attempts.
        show_tokens (bool): Whether to display token count information.
    Returns:
        list[float]: The embedding vector.
    """
    
    # Count tokens before processing
    original_tokens = estimate_tokens(text)
    
    # Truncate very long texts to avoid server errors
    max_length = 1500  # Reduced from 2000 to be more conservative
    if len(text) > max_length:
        text = text[:max_length]
        truncated_tokens = estimate_tokens(text)
        if show_tokens:
            print(f"Warning: Text truncated from {original_tokens} to {truncated_tokens} tokens for embedding")
    elif show_tokens:
        print(f"Embedding tokens: {original_tokens}")
    
    # Skip empty or very short texts
    if not text.strip() or len(text.strip()) < 10:
        print("Warning: Skipping empty or very short text")
        return [0.0] * 768  # Return zero vector with typical embedding dimension
    
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text.strip()}
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
                print(f"Text length: {len(text)}")
                print(f"First 200 chars: {text[:200]}")
                raise
        except Exception as e:
            print(f"Embedding error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print(f"Text length: {len(text)}")
                print(f"First 200 chars: {text[:200]}")
                raise


def call_ollama_llm(prompt, model="llama2", show_tokens=False):
    """
    Call Ollama's LLM to generate a response for the given prompt with token counting.
    Args:
        prompt (str): The prompt/question to send to the LLM.
        model (str): The Ollama LLM model to use (default: "llama2").
        show_tokens (bool): Whether to display token counts and timing info.
    Returns:
        str: The LLM's response.
    """
    # Count tokens in prompt and track timing
    prompt_tokens = estimate_tokens(prompt)
    start_time = time.time()
    
    if show_tokens:
        print(f"LLM Prompt tokens: {prompt_tokens}")
    
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    
    llm_response = response.json().get("response", "[No response]")
    
    # Count response tokens and show timing
    response_tokens = estimate_tokens(llm_response)
    elapsed_time = time.time() - start_time
    
    if show_tokens:
        total_tokens = prompt_tokens + response_tokens
        print(f"LLM Response tokens: {response_tokens}")
        print(f"Total LLM tokens: {total_tokens}")
        print(f"LLM processing time: {elapsed_time:.2f}s")
    
    return llm_response

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
    prompt_builder = PromptBuilder(
        template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}",
        required_variables=["context", "query"]
    )
    llm_model = "llama2"
    query = "Who is Sachin Tendulkar?"
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=1)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)

# --- Example 2: Pipeline with PDF files as input ---

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
    prompt_builder = PromptBuilder(
        template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}",
        required_variables=["context", "query"]
    )
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=3)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)


# --- Example 3 with pdfs as files and saving the embeddings locally as files:
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
    prompt_builder = PromptBuilder(
        template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}",
        required_variables=["context", "query"]
    )
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
    # Use cricket folder name for cricket data
    chunk_dir = os.path.join(os.path.dirname(persist_path), "../chunks/cricket")
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
        chunk_path = os.path.join(chunk_dir, f"cricket_chunk_{idx+1}.txt")
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
    print("Generating embeddings for chunks...")
    for idx, doc in enumerate(loaded_chunk_docs, 1):
        print(f"Processing chunk {idx}/{len(loaded_chunk_docs)}")
        try:
            doc.embedding = get_ollama_embedding(doc.content, model=embedding_model)
        except Exception as e:
            print(f"Failed to generate embedding for chunk {idx}: {e}")
            print(f"Chunk content preview: {doc.content[:200]}...")
            raise
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
    prompt_builder = PromptBuilder(
        template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}",
        required_variables=["context", "query"]
    )
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=3)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)


def run_ollama_pipeline_with_intel_data_chunked_and_persist(persist_path, chunk_size=512, chunk_overlap=50):
    """
    Ingests all supported files (PDF, TXT, CSV) from the intel_simulated_data folder, splits them into smaller chunks, 
    embeds them with Ollama, stores them in a persistent document store, and saves the store to disk as a JSON file. 
    Also runs a retrieval and LLM prompt from the persisted store.
    Args:
        persist_path (str): Path to save the persistent document store (e.g., './store/embedding/intel_store.json').
        chunk_size (int): Number of characters per chunk (default: 512).
        chunk_overlap (int): Number of overlapping characters between chunks (default: 50).
    """
    intel_data_dir = "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/intel_simulated_data"
    
    # Find all supported file types in the intel_simulated_data folder
    pdf_paths = glob.glob(os.path.join(intel_data_dir, "*.pdf"))
    txt_paths = glob.glob(os.path.join(intel_data_dir, "*.txt"))
    csv_paths = glob.glob(os.path.join(intel_data_dir, "*.csv"))
    
    all_files = pdf_paths + txt_paths + csv_paths
    
    if not all_files:
        print("No supported files found in intel_simulated_data folder")
        return
    
    print(f"Found {len(all_files)} files in intel_simulated_data folder:")
    print(f"  - {len(pdf_paths)} PDF files")
    print(f"  - {len(txt_paths)} TXT files") 
    print(f"  - {len(csv_paths)} CSV files")
    
    docs = []
    
    # Process PDF files
    if pdf_paths:
        print("Processing PDF files...")
        converter = PyPDFToDocument()
        pdf_docs = converter.run(sources=pdf_paths)["documents"]
        docs.extend(pdf_docs)
    
    # Process TXT files
    if txt_paths:
        print("Processing TXT files...")
        for txt_path in txt_paths:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = Document(content=content, meta={"source": txt_path, "file_type": "txt"})
            docs.append(doc)
    
    # Process CSV files
    if csv_paths:
        print("Processing CSV files...")
        for csv_path in csv_paths:
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                rows = list(csv_reader)
                
                # Convert CSV to text format for better chunking
                if rows:
                    headers = rows[0] if rows else []
                    content_lines = [f"Headers: {', '.join(headers)}"]
                    
                    for i, row in enumerate(rows[1:], 1):
                        if len(row) == len(headers):
                            row_text = "; ".join([f"{headers[j]}: {row[j]}" for j in range(len(headers))])
                            content_lines.append(f"Row {i}: {row_text}")
                    
                    content = "\n".join(content_lines)
                    doc = Document(content=content, meta={"source": csv_path, "file_type": "csv"})
                    docs.append(doc)
    
    print(f"Loaded {len(docs)} documents total")
    
    # Split documents into smaller chunks and save each chunk as a file
    # Use intel_simulated_data folder name for intel data
    chunk_dir = os.path.join(os.path.dirname(persist_path), "../chunks/intel_simulated_data")
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
        chunk_path = os.path.join(chunk_dir, f"intel_simulated_data_chunk_{idx+1}.txt")
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(doc.content)
        chunk_files.append(chunk_path)

    print(f"Created {len(chunked_docs)} chunks from intel data files")

    # Load chunks from files as Documents
    loaded_chunk_docs = []
    for chunk_path in chunk_files:
        with open(chunk_path, "r", encoding="utf-8") as f:
            content = f.read()
        loaded_chunk_docs.append(Document(content=content))
    
    embedding_model = "nomic-embed-text"
    print("Generating embeddings for intel chunks...")
    for idx, doc in enumerate(loaded_chunk_docs, 1):
        print(f"Processing chunk {idx}/{len(loaded_chunk_docs)}")
        try:
            doc.embedding = get_ollama_embedding(doc.content, model=embedding_model)
        except Exception as e:
            print(f"Failed to generate embedding for chunk {idx}: {e}")
            print(f"Chunk content preview: {doc.content[:200]}...")
            raise
    
    store = InMemoryDocumentStore()
    store.write_documents(loaded_chunk_docs)
    
    # Persist the document store to disk
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    store.save_to_disk(persist_path)
    print(f"Intel document store with embeddings saved to: {persist_path}")

    # Reload the store and run a retrieval + LLM prompt
    loaded_store = InMemoryDocumentStore.load_from_disk(persist_path)
    retriever = InMemoryEmbeddingRetriever(document_store=loaded_store)
    llm_model = "llama2"
    query = "What can you tell me about Quantum"#"What are the main cybersecurity threats mentioned in the intelligence reports?"
    prompt_builder = PromptBuilder(
        template="Answer the user's question using the provided context. Context: {{context}} Question: {{query}}",
        required_variables=["context", "query"]
    )
    query_emb = get_ollama_embedding(query, model=embedding_model)
    retrieved = retriever.run(query_embedding=query_emb, top_k=5)
    context = " ".join([doc.content for doc in retrieved["documents"]])
    prompt = prompt_builder.run(template_variables={"context": context, "query": query})["prompt"]
    answer = call_ollama_llm(prompt, model=llm_model)
    print("Prompt sent to LLM:", prompt)
    print("LLM answer:", answer)


def ask_intel_question(question, persist_path=None, top_k=5, show_tokens=True):
    """
    Interactive Q&A function that loads the existing intel document store and answers questions
    without reprocessing files. Much faster for interactive sessions.
    
    USAGE:
        # Single question mode:
        python haystack_ollama_local_pipeline_example.py --mode intel_qa --question "What are the main threat actors?"
        
        # Or call directly in code:
        ask_intel_question("What cybersecurity threats are mentioned?")
        
    PREREQUISITE:
        Must run --mode intel_chunked_persist first to create the document store.
    
    Args:
        question (str): The question to ask about the intel data.
        persist_path (str): Path to the intel document store (defaults to standard location).
        top_k (int): Number of relevant chunks to retrieve (default: 5).
        show_tokens (bool): Whether to display token counts and timing info (default: True).
    Returns:
        str: The LLM's answer to the question.
    """
    if persist_path is None:
        persist_path = "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/store/embedding/intel_store.json"
    
    # Check if the store exists
    if not os.path.exists(persist_path):
        print(f"Error: Intel document store not found at {persist_path}")
        print("Please run --mode intel_chunked_persist first to create the store.")
        return None
    
    print(f"Loading intel document store from {persist_path}...")
    
    try:
        # Load the existing document store
        loaded_store = InMemoryDocumentStore.load_from_disk(persist_path)
        retriever = InMemoryEmbeddingRetriever(document_store=loaded_store)
        
        # Set up models and prompt template
        embedding_model = "nomic-embed-text"
        llm_model = "llama2"
        prompt_builder = PromptBuilder(
            template="You are a cybersecurity intelligence analyst. Answer the user's question using the provided context from intelligence reports. "
                    "Provide detailed, accurate information based on the context. If the context doesn't contain relevant information, "
                    "state that clearly.\n\nContext: {{context}}\n\nQuestion: {{query}}\n\nAnswer:",
            required_variables=["context", "query"]
        )
        
        print(f"Processing question: {question}")
        if show_tokens:
            print(f"\n{'='*40} TOKEN ANALYSIS {'='*40}")
        
        # Generate embedding for the question
        query_emb = get_ollama_embedding(question, model=embedding_model, show_tokens=show_tokens)
        
        # Retrieve relevant chunks
        retrieved = retriever.run(query_embedding=query_emb, top_k=top_k)
        context = " ".join([doc.content for doc in retrieved["documents"]])
        
        if show_tokens:
            context_tokens = estimate_tokens(context)
            print(f"Retrieved context tokens: {context_tokens}")
        
        # Generate response
        prompt = prompt_builder.run(template_variables={"context": context, "query": question})["prompt"]
        answer = call_ollama_llm(prompt, model=llm_model, show_tokens=show_tokens)
        
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        print(f"ANSWER: {answer}")
        print(f"{'='*60}\n")
        
        return answer
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return None


def intel_interactive_session(persist_path=None, show_tokens=True):
    """
    Start an interactive Q&A session with the intel document store with token tracking.
    
    USAGE:
        # Start interactive session:
        python haystack_ollama_local_pipeline_example.py --mode intel_interactive
        
        # Then ask questions interactively:
        Enter your question: What are the main APT groups?
        Enter your question: Tell me about quantum threats
        Enter your question: quit
        
    PREREQUISITE:
        Must run --mode intel_chunked_persist first to create the document store.
        
    SAMPLE QUESTIONS:
        - "What are the main cybersecurity threats mentioned?"
        - "Tell me about APT-GLACIER's tactics and targets"
        - "What quantum cryptography threats are discussed?"
        - "What is the economic impact of the cyberattacks?"
        - "Which countries are most affected by cyber incidents?"
        - "What are the recommended mitigation strategies?"
        
    Args:
        persist_path (str): Path to the intel document store (defaults to standard location).
        show_tokens (bool): Whether to display token counts and session statistics (default: True).
    """
    if persist_path is None:
        persist_path = "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/store/embedding/intel_store.json"
    
    print("=" * 60)
    print("INTEL INTERACTIVE Q&A SESSION")
    print("=" * 60)
    print("Ask questions about the cybersecurity intelligence data.")
    print("Type 'quit', 'exit', or 'q' to end the session.")
    if show_tokens:
        print("Token usage statistics will be displayed for each query.")
    print()
    
    # Check if store exists
    if not os.path.exists(persist_path):
        print(f"Error: Intel document store not found at {persist_path}")
        print("Please run --mode intel_chunked_persist first to create the document store.")
        return
    
    print(f"Loaded intel store: {persist_path}\n")
    
    # Initialize session tracking
    session_start = time.time()
    question_count = 0
    total_session_tokens = 0
    
    while True:
        try:
            question = input("Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                # Display session summary
                session_duration = time.time() - session_start
                if show_tokens and question_count > 0:
                    print(f"\n{'='*60}")
                    print("SESSION SUMMARY")
                    print(f"{'='*60}")
                    print(f"Questions asked: {question_count}")
                    print(f"Total session tokens: {total_session_tokens}")
                    print(f"Average tokens per question: {total_session_tokens/question_count:.0f}")
                    print(f"Session duration: {session_duration:.1f}s")
                    print(f"{'='*60}")
                print("Ending session. Goodbye!")
                break
            
            if not question:
                print("Please enter a question.\n")
                continue
            
            # Track question start for token counting
            question_start_tokens = total_session_tokens
            
            answer = ask_intel_question(question, persist_path, show_tokens=show_tokens)
            if answer is None:
                print("Failed to process question. Please try again.\n")
            else:
                question_count += 1
                if show_tokens:
                    # Estimate tokens used in this question (rough approximation)
                    question_tokens = estimate_tokens(question) + estimate_tokens(answer) + 500  # +500 for context/prompt overhead
                    total_session_tokens += question_tokens
                    print(f"Question {question_count} tokens: ~{question_tokens}")
                    print(f"Session total tokens: {total_session_tokens}\n")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except EOFError:
            print("\nSession ended. Goodbye!")
            break


# --- Example usage ---
"""
COMPLETE USAGE GUIDE:

1. SETUP (One-time):
   ollama serve                    # Start Ollama service
   ollama pull llama2             # Pull LLM model
   ollama pull nomic-embed-text   # Pull embedding model

2. INTEL DATA WORKFLOW:
   # Step 1: Process intel data (creates embeddings - run once)
   python haystack_ollama_local_pipeline_example.py --mode intel_chunked_persist
   
   # Step 2: Ask questions (fast, uses existing embeddings)
   
   # Option A: Single question
   python haystack_ollama_local_pipeline_example.py --mode intel_qa --question "What are the main APT groups?"
   
   # Option B: Interactive session
   python haystack_ollama_local_pipeline_example.py --mode intel_interactive
   # Then type questions and press Enter, type 'quit' to exit

3. CRICKET DATA WORKFLOW:
   python haystack_ollama_local_pipeline_example.py --mode pdf_chunked_persist

4. OTHER MODES:
   python haystack_ollama_local_pipeline_example.py --mode minimal
   python haystack_ollama_local_pipeline_example.py --mode pdf --pdfs path/to/file.pdf

SAMPLE INTEL QUESTIONS:
- "What are the main cybersecurity threats mentioned in the intelligence reports?"
- "Tell me about APT-GLACIER's tactics, techniques, and procedures"
- "What quantum cryptography threats are discussed?"
- "What is the economic impact of the cyberattacks mentioned?"
- "Which countries or regions are most affected by cyber incidents?"
- "What are the recommended mitigation strategies?"
- "Tell me about Operation NORDWIND"
- "What vulnerabilities are mentioned in the reports?"

FOLDER STRUCTURE CREATED:
LLM_Engines/ollama/store/
├── chunks/
│   ├── cricket/                 # Cricket data chunks
│   └── intel_simulated_data/    # Intel data chunks
└── embedding/
    ├── indian_cricketers_store.json  # Cricket embeddings
    └── intel_store.json              # Intel embeddings
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Haystack+Ollama local pipeline example.")
    parser.add_argument(
        "--mode",
        choices=["minimal", "pdf", "pdf_persist", "pdf_chunked_persist", "intel_chunked_persist", "intel_qa", "intel_interactive"],
        default="pdf_chunked_persist",
        help="Which pipeline function to run: minimal (hardcoded docs), pdf (PDF input), pdf_persist (PDF input with persistent embedding store), pdf_chunked_persist (PDF input, chunked, with persistent embedding store), intel_chunked_persist (all files from intel_simulated_data folder, chunked, with persistent embedding store), intel_qa (ask a single question to intel data), or intel_interactive (start interactive Q&A session with intel data)."
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
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/cricket/Indian_Cricketers_Statistics.pdf",
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/cricket/Indian_Cricketers_Records.pdf",
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/cricket/Indian_Cricketers_Family.pdf",
            "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/data/cricket/Indian_Cricketers_Achievements.pdf"
        ],
        help="List of PDF file paths to use as input (used with --mode pdf or pdf_persist)."
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask the intel system (used with --mode intel_qa)."
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
    elif args.mode == "intel_chunked_persist":
        print("--- Running Ollama pipeline with all intel_simulated_data files, chunked, and persistent embedding store ---")
        intel_persist_path = "/Users/mathewthomas/Documents/hobby_projects/AI_ML_Work/567COG_DS_AI_ML/LLM_Engines/ollama/store/embedding/intel_store.json"
        run_ollama_pipeline_with_intel_data_chunked_and_persist(intel_persist_path)
    elif args.mode == "intel_qa":
        print("--- Intel Q&A Mode ---")
        if not args.question:
            print("Error: --question argument is required for intel_qa mode")
            print('Example: python script.py --mode intel_qa --question "What are the main threat actors?"')
        else:
            ask_intel_question(args.question)
    elif args.mode == "intel_interactive":
        print("--- Starting Intel Interactive Q&A Session ---")
        intel_interactive_session()
