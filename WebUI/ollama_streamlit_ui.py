import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Haystack+Ollama Interactive LLM UI", layout="wide")

st.title("Haystack+Ollama Interactive LLM UI")


# LLM Q&A modes available in the pipeline for Streamlit
MODES = {
    "Tennis Q&A": "tennis_qa",
    "Intel Q&A": "intel_qa",
}

st.sidebar.header("Select LLM Interactive Mode")
mode_label = st.sidebar.selectbox("Choose a mode:", list(MODES.keys()))
mode = MODES[mode_label]

st.sidebar.markdown("---")

# Optionally allow user to set persist_path or other args
persist_path = st.sidebar.text_input(
    "Persist Path (optional)", value=""
)

st.sidebar.markdown("---")

st.markdown(f"**Selected Mode:** `{mode_label}`")

# Session state for chat
if "history" not in st.session_state:
    st.session_state.history = []


# Show chat input for Q&A modes
user_input = st.text_input("Ask a question:", "")
if st.button("Send") and user_input.strip():
    # Call the pipeline script with the selected mode and user input as --question
    cmd = [sys.executable, "haystack_ollama_local_pipeline_example.py", "--mode", mode, "--question", user_input]
    if persist_path.strip():
        cmd += ["--persist_path", persist_path.strip()]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=120)
        if err.strip():
            st.session_state.history.append((user_input, f"[Error] {err.strip()}"))
        else:
            st.session_state.history.append((user_input, out.strip()))
    except Exception as e:
        st.session_state.history.append((user_input, f"[Error] {e}"))
# Display chat history
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**LLM:** {a}")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Haystack+Ollama local LLM workflows.")
