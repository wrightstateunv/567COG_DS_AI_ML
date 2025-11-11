
import streamlit as st
from huggingface_hub import snapshot_download, list_models
import openai
import os

st.title("Haystack Model Downloader & Data Uploader")

st.header("1. Select Model Provider")
provider = st.selectbox("Choose a provider", ["HuggingFace", "OpenAI"])

if provider == "HuggingFace":
    st.subheader("HuggingFace Model Selection")
    search_query = st.text_input("Search for a model (e.g. bert, gpt2, llama)", "bert")
    if search_query:
        models = list_models(filter=search_query, limit=10)
        model_names = [m.modelId for m in models]
        model_choice = st.selectbox("Select a model to download", model_names)
        if st.button("Download Model"):
            with st.spinner("Downloading model from HuggingFace..."):
                model_dir = snapshot_download(repo_id=model_choice)
            st.success(f"Model '{model_choice}' downloaded to {model_dir}")

elif provider == "OpenAI":
    st.subheader("OpenAI Model Selection")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        try:
            models = openai.models.list()
            model_names = [m.id for m in models['data']]
            model_choice = st.selectbox("Select a model to use", model_names)
            st.info(f"Selected OpenAI model: {model_choice}")
        except Exception as e:
            st.error(f"Error fetching OpenAI models: {e}")

st.header("2. Upload Your Data")
uploaded_file = st.file_uploader("Upload a file for processing (CSV, TXT, JSON, etc.)")
if uploaded_file is not None:
    file_path = os.path.join("uploaded_data", uploaded_file.name)
    os.makedirs("uploaded_data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded and saved to {file_path}")

st.markdown("---")
st.markdown("This is a demo UI for model selection and data upload. Integrate with Haystack pipelines as needed.")
