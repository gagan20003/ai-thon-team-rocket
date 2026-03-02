import os
import tempfile
from typing import List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import DOCTOR_QUESTIONS_PROMPT, QA_PROMPT, SUMMARY_PROMPT


def get_llm(provider: str = "groq", model: str | None = None):
    if provider == "openai":
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    return ChatGroq(model_name=model or "llama3-70b-8192", temperature=0)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


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


def summarize_report(vector_store: FAISS, llm) -> str:
    docs = vector_store.similarity_search("Summarize the medical report", k=6)
    if not docs:
        raise ValueError("No content found to summarize.")
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_doctor_questions(vector_store: FAISS, llm) -> str:
    docs = vector_store.similarity_search(
        "Generate questions for a doctor visit based on abnormal findings",
        k=6,
    )
    if not docs:
        raise ValueError("No content found to generate questions.")
    context = "\n\n".join(doc.page_content for doc in docs)
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
