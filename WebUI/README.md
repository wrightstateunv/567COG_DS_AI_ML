# WebUI for Ollama Chat

This folder contains a Streamlit-based web user interface for chatting with your local Ollama LLM models.

## Purpose
- Provide a simple browser-based chat interface to interact with Ollama models running locally.
- Select from available models, ask questions, and view conversation history.

## How to Start the Streamlit App
1. Make sure your Python environment is activated and Streamlit is installed:
   ```sh
   pip install streamlit requests
   ```
2. Make sure the Ollama service is running:
   ```sh
   brew services start ollama
   # or
   ollama serve
   ```
3. Start the Streamlit app:
   ```sh
   streamlit run WebUI/ollama_chat_ui.py
   ```
4. Open the provided local URL in your browser (usually http://localhost:8501).

## How to Stop the Streamlit App
- In the terminal where Streamlit is running, press `Ctrl+C` to stop the app.

## Notes
- The app communicates with Ollama's local API at `http://localhost:11434`.
- You can add more features or customize the UI as needed.
