import streamlit as st
import os
import sys

# Add project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import process_uploaded_file_rag, reset_knowledge_base
from src.engine import get_rag_response

st.set_page_config(page_title="Enterprise Doc Q&A Agent", layout="wide")

st.title("🤖 Enterprise Document Q&A Agent (Custom RAG)")
st.markdown("Upload research papers (PDF) and ask questions. Uses **Custom RAG** with Google Gemini & ChromaDB.")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Uploads
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Chunking, Embedding & Indexing..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    num_chunks = process_uploaded_file_rag(uploaded_file)
                    total_chunks += num_chunks
                
                st.success(f"Processed {len(uploaded_files)} files into {total_chunks} chunks!")
        else:
            st.info("Please select files to upload.")

    st.divider()
    if st.button("Reset Knowledge Base"):
        reset_knowledge_base()
        st.session_state.messages = []
        st.success("Knowledge Base Reset!")

    if st.button("Clear Chat History"):
        st.session_state.messages = []

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving & Generating..."):
            try:
                response_text = get_rag_response(prompt)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                import traceback
                st.error(f"Error generating response: {e}")
                st.code(traceback.format_exc())

# Initial check
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("⚠️ GOOGLE_API_KEY not found in environment.")
