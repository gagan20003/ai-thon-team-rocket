import hashlib
import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    answer_question,
    build_conversational_chain,
    build_vector_store,
    generate_doctor_questions,
    get_llm,
    load_pdf_from_bytes,
    split_documents,
    summarize_report,
)


load_dotenv()

st.set_page_config(page_title="Medical Report Assistant", layout="wide")

st.title("Medical Report Assistant")
st.markdown("Upload a medical report PDF to get a clear summary and ask questions.")

provider_label = st.sidebar.selectbox("LLM Provider", ["Groq", "OpenAI"], index=0)
provider = "groq" if provider_label == "Groq" else "openai"
low_token_mode = st.sidebar.checkbox("Low token mode (free tier)", value=True)
embeddings_label = st.sidebar.selectbox(
    "Embeddings Provider", ["Jina (free)", "OpenAI"], index=0
)
embedding_provider = "jina" if embeddings_label.startswith("Jina") else "openai"

if embedding_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OPENAI_API_KEY for embeddings. Add it to your environment.")
    st.stop()
if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OPENAI_API_KEY for the OpenAI LLM provider.")
    st.stop()
if embedding_provider == "jina" and not os.getenv("JINA_API_KEY"):
    st.error("Missing JINA_API_KEY. Add it to your environment before proceeding.")
    st.stop()

if provider == "groq" and not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY. Add it to your environment before proceeding.")
    st.stop()


@st.cache_resource(show_spinner=False)
def build_index(file_bytes: bytes, embedding_provider: str):
    docs = load_pdf_from_bytes(file_bytes)
    if not docs or all(not d.page_content.strip() for d in docs):
        raise ValueError("The PDF appears to be empty or unreadable.")
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")
    return build_vector_store(chunks, embedding_provider=embedding_provider)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = ""
    if "summary_provider" not in st.session_state:
        st.session_state.summary_provider = ""
    if "doctor_questions" not in st.session_state:
        st.session_state.doctor_questions = ""
    if "doctor_questions_provider" not in st.session_state:
        st.session_state.doctor_questions_provider = ""


init_session_state()

uploaded_file = st.file_uploader("Upload a medical report (PDF only)", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.file_hash:
        st.session_state.file_hash = file_hash
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.summary_provider = ""
        st.session_state.doctor_questions = ""
        st.session_state.doctor_questions_provider = ""

    try:
        with st.spinner("Processing report..."):
            vector_store = build_index(file_bytes, embedding_provider=embedding_provider)
            llm = get_llm(provider=provider)
            if not st.session_state.summary or st.session_state.summary_provider != provider:
                k = 3 if low_token_mode else 6
                max_chars = 3500 if low_token_mode else 8000
                st.session_state.summary = summarize_report(
                    vector_store, llm, k=k, max_chars=max_chars
                )
                st.session_state.summary_provider = provider
        st.success("Report indexed successfully.")
    except Exception as exc:
        st.error(f"Failed to process the PDF: {exc}")
        st.stop()

    st.subheader("Summary")
    st.info(st.session_state.summary)

    if st.button("Generate Doctor Visit Questions"):
        with st.spinner("Generating questions..."):
            k = 3 if low_token_mode else 6
            max_chars = 3500 if low_token_mode else 8000
            st.session_state.doctor_questions = generate_doctor_questions(
                vector_store, llm, k=k, max_chars=max_chars
            )
            st.session_state.doctor_questions_provider = provider

    if st.session_state.doctor_questions:
        st.subheader("Doctor Visit Questions")
        st.markdown(st.session_state.doctor_questions)

    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about the report")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = build_conversational_chain(vector_store, llm)
                answer = answer_question(
                    chain=chain,
                    question=user_input,
                    chat_history=st.session_state.chat_history,
                )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append((user_input, answer))

else:
    st.info("Upload a PDF to begin.")
