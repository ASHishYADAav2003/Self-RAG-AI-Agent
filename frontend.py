import streamlit as st
import os
import tempfile
from backend import SelfRAG
from langchain_core.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")
    
    def on_llm_end(self, response, **kwargs):
        self.container.markdown(self.text)


st.set_page_config(page_title="Self-RAG AI Agent", layout="wide")
st.title("Self-RAG AI Agent")
st.write("Upload PDF files and chat with the Self-RAG agent.")

with st.sidebar:
    st.header("📄 Document Info")
    if "pdf_paths" in st.session_state and st.session_state.pdf_paths:
        st.write(f"Loaded {len(st.session_state.pdf_paths)} document(s):")
        for path in st.session_state.pdf_paths:
            st.write(f"- {os.path.basename(path)}")
    else:
        st.write("No documents loaded yet.")
    
    st.divider()
    
    st.header("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        for msg in st.session_state.messages:
            role = "🧑 User" if msg["role"] == "user" else "🤖 Agent"
            content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            st.markdown(f"**{role}:** {content}")
    else:
        st.write("No chat history.")

uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if "rag_agent" not in st.session_state:
        with st.spinner("Processing documents and setting up RAG agent..."):
            # Save uploaded files to temp directory
            temp_dir = tempfile.mkdtemp()
            pdf_paths = []
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(file_path)
            
            try:
                st.session_state.rag_agent = SelfRAG(pdf_paths=pdf_paths)
                st.session_state.pdf_paths = pdf_paths
                st.success("RAG agent setup complete!")
            except Exception as e:
                st.error(f"Error initializing RAG: {e}")

    if "rag_agent" in st.session_state:
        # Chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question based on the documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer_container = st.empty()
                stream_handler = StreamHandler(answer_container)
                
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_agent.query(prompt, stream_handler=stream_handler)
                    answer = result.get("answer", "I couldn't find an answer.")
                    
                    if not stream_handler.text:
                        answer_container.markdown(answer)
                    else:
                        answer_container.markdown(stream_handler.text)
                    
                    with st.expander("Show Execution Details"):
                        st.write("**Need Retrieval:**", result.get("need_retrieval"))
                        st.write("**Rewrite Tries:**", result.get("rewrite_tries", 0))
                        st.write("**IsSUP (Verification):**", result.get("issup", "N/A"))
                        st.write("**IsUSE (Usefulness):**", result.get("isuse", "N/A"))
                        st.write("**Context Provided:**", result.get("context", "N/A"))

            st.session_state.messages.append({"role": "assistant", "content": answer})
