import os
import tempfile
from typing import List, Tuple

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import DOCTOR_QUESTIONS_PROMPT, QA_PROMPT, SUMMARY_PROMPT

# ---------- LiteLLM gateway configuration ----------
# NOTE: env vars are read inside functions (not at module level)
# because this module is imported before load_dotenv() runs.

def _get_gateway_config():
    """Read LiteLLM gateway settings from environment."""
    return {
        "base_url": os.getenv("LITELLM_BASE_URL", "https://litellm.ai-coe-test.aws.evernorthcloud.com"),
        "api_key": os.getenv("LITELLM_API_KEY", ""),
    }


def get_llm(model: str | None = None):
    """Return a ChatOpenAI instance pointed at the LiteLLM gateway."""
    cfg = _get_gateway_config()
    return ChatOpenAI(
        model=model or os.getenv("LITELLM_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )


def get_embeddings():
    """Return an OpenAIEmbeddings instance pointed at the LiteLLM gateway."""
    cfg = _get_gateway_config()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )


def load_pdf_from_bytes(file_bytes: bytes) -> List[Document]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        return docs
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def build_vector_store(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def summarize_report(
    vector_store: FAISS,
    llm,
    k: int = 6,
    max_chars: int = 8000,
) -> str:
    docs = vector_store.similarity_search("Summarize the medical report", k=k)
    if not docs:
        raise ValueError("No content found to summarize.")
    context_parts = []
    total_chars = 0
    for doc in docs:
        chunk = doc.page_content.strip()
        if not chunk:
            continue
        if total_chars + len(chunk) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                context_parts.append(chunk[:remaining])
            break
        context_parts.append(chunk)
        total_chars += len(chunk)
    context = "\n\n".join(context_parts)
    prompt = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_doctor_questions(
    vector_store: FAISS,
    llm,
    k: int = 6,
    max_chars: int = 8000,
) -> str:
    docs = vector_store.similarity_search(
        "Generate questions for a doctor visit based on abnormal findings",
        k=k,
    )
    if not docs:
        raise ValueError("No content found to generate questions.")
    context_parts = []
    total_chars = 0
    for doc in docs:
        chunk = doc.page_content.strip()
        if not chunk:
            continue
        if total_chars + len(chunk) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                context_parts.append(chunk[:remaining])
            break
        context_parts.append(chunk)
        total_chars += len(chunk)
    context = "\n\n".join(context_parts)
    prompt = DOCTOR_QUESTIONS_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content.strip()


def build_conversational_chain(vector_store: FAISS, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )


def answer_question(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: List[Tuple[str, str]],
) -> str:
    result = chain({"question": question, "chat_history": chat_history})
    return result["answer"].strip()
