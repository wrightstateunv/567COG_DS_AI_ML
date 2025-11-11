# Ollama Engine Folder

This folder documents the use of Ollama as a local engine for embeddings and LLM inference in this project.

## Purpose
- Serve as a reference for local LLM and embedding model usage (instead of cloud APIs like OpenAI)
- Track which models are used for semantic search and QA
- Provide setup, usage, and management instructions for Ollama

## What is Installed
- **Ollama** (installed via Homebrew)
- **Models pulled:**
  - `nomic-embed-text` (for embeddings)
  - `llama2` (for LLM completions)

## How to Run Ollama
1. **Start the Ollama service:**
   ```sh
   brew services start ollama
   # or run manually:
   ollama serve
   ```
2. **List available models:**
   ```sh
   ollama list
   ```
3. **Pull additional models:**
   ```sh
   ollama pull <model-name>
   # Example:
   ollama pull llama2
   ollama pull nomic-embed-text
   ```
4. **Ollama runs a local API at** `http://localhost:11434` by default.

## How to Delete Ollama and Models
- **Stop the service:**
  ```sh
  brew services stop ollama
  ```
- **Uninstall Ollama:**
  ```sh
  brew uninstall ollama
  ```
- **Remove all models/data:**
  ```sh
  rm -rf ~/.ollama
  ```


## Using Haystack with Ollama (Local Pipeline)

See `haystack_ollama_local_pipeline_example.py` for a fully local semantic search and QA pipeline using Haystack 2.x and Ollama.

### Features
- Local embeddings and LLM completions (no cloud APIs)
- Ingests both hardcoded text and PDF files
- All inference is local; no data leaves your machine

### Requirements
- `haystack-ai >=2.x`
- `pypdf` (for PDF ingestion)
- `requests`
- Ollama running locally with required models pulled

### Example Usage
1. **Start Ollama:**
  ```sh
  ollama serve
  # or (if using Homebrew):
  brew services start ollama
  ```
2. **Pull required models:**
  ```sh
  ollama pull llama2
  ollama pull nomic-embed-text
  ```
3. **Install Python dependencies:**
  ```sh
  pip install haystack-ai pypdf requests
  ```
4. **Run the script:**
  ```sh
  python haystack_ollama_local_pipeline_example.py
  ```
  - Edit the script to change the PDF file paths or queries as needed.

### Troubleshooting
- If you see `ImportError: No module named 'pypdf'`, run `pip install pypdf`.
- If Ollama is not running, start it with `ollama serve`.
- Ensure the required models are pulled with `ollama pull <model-name>`.

---
- Ollama stores models in `~/.ollama` (not in this folder).
- Use this folder for scripts, configs, or documentation related to Ollama usage in this project.
