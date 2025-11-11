import streamlit as st
import requests

st.set_page_config(page_title="Ollama Chat UI", page_icon="🤖")
st.title("Ollama Local LLM Chat")

# Model selection
def get_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            return [tag["name"] for tag in tags]
    except Exception:
        pass
    return ["llama2", "llama3.2", "nomic-embed-text"]

models = get_ollama_models()
model = st.selectbox("Select Ollama Model", models)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Ask a question:")
if st.button("Send") and user_input:
    st.session_state["chat_history"].append(("user", user_input))
    # Call Ollama API
    payload = {
        "model": model,
        "prompt": user_input,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        if response.status_code == 200:
            answer = response.json().get("response", "[No response]")
        else:
            answer = f"[Error: {response.status_code}]"
    except Exception as e:
        answer = f"[Error: {e}]"
    st.session_state["chat_history"].append(("ollama", answer))

# Display chat history
st.markdown("---")
st.subheader("Conversation")
for role, msg in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Ollama:** {msg}")
