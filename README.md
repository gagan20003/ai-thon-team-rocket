# Medical Report Assistant

A Streamlit RAG chatbot that summarizes medical report PDFs, highlights abnormal findings, and answers questions grounded only in the uploaded report.

## Features
- PDF upload and parsing with PyPDFLoader
- RAG pipeline using LangChain + FAISS
- Medical summary with abnormal findings and plain-language explanations
- Conversational Q&A grounded strictly in the report

## Tech Stack
- Python + Streamlit
- LangChain
- GROQ (llama3-70b-8192) with optional OpenAI switch
- Embeddings: Jina (free) or OpenAI (text-embedding-3-small)
- FAISS in-memory vector store

## Setup
1. Use Python 3.10 or 3.11 (recommended for LangChain compatibility).
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `.env.example` and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   JINA_API_KEY=your_jina_api_key
   ```

## Run
```
streamlit run app.py
```

## Usage
1. Open the app in your browser.
2. Upload a medical report PDF.
3. Review the auto-generated summary.
4. Ask questions in the chat. The assistant will answer using only the report.

## Notes
- The default LLM provider is Groq. You can switch to OpenAI from the sidebar.
- The default embeddings provider is Jina (free). You can switch to OpenAI from the sidebar.
- The assistant will not diagnose or prescribe and will encourage consulting a clinician.
