# Medical Report Assistant

A Streamlit RAG chatbot that summarizes medical report PDFs, highlights abnormal findings, and answers questions grounded only in the uploaded report.

## Features

- PDF upload and parsing with PyPDFLoader
- RAG pipeline using LangChain + FAISS
- Medical summary with abnormal findings and plain-language explanations
- Conversational Q&A grounded strictly in the report
- Nearby hospitals finder sorted by distance using Google Places API

## Tech Stack

- Python + Streamlit
- LangChain
- LiteLLM gateway (OpenAI-compatible chat + embeddings)
- Google Maps APIs (Geolocation + Geocoding + Places Nearby Search)
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
   LITELLM_BASE_URL=https://litellm.ai-coe-test.aws.evernorthcloud.com
   LITELLM_API_KEY=your_litellm_gateway_key
   LITELLM_CHAT_MODEL=gpt-4o-mini
   GOOGLE_MAPS_API_KEY=your_google_maps_key
   GOOGLE_PLACES_RADIUS_METERS=5000
   GOOGLE_PLACES_MAX_RESULTS=10
   ```

## Run

```
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. Upload a medical report PDF.
3. Review the auto-generated summary.
4. Use Nearby Hospitals to auto-detect your approximate location (or enter a location manually) and list nearest hospitals.
5. Ask questions in the chat. The assistant will answer using only the report.

## Notes

- Enable these APIs in Google Cloud for hospital search: Geolocation API, Geocoding API, and Places API.
- Nearby-hospital location auto-detection uses IP-based approximate location from Google Geolocation API.
- The assistant will not diagnose or prescribe and will encourage consulting a clinician.
