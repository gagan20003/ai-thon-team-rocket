import os
import math
import tempfile
from typing import Any, Dict, List, Tuple

import httpx
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


def _get_google_maps_config() -> Dict[str, Any]:
    return {
        "api_key": os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        "radius_meters": int(os.getenv("GOOGLE_PLACES_RADIUS_METERS", "5000")),
        "max_results": int(os.getenv("GOOGLE_PLACES_MAX_RESULTS", "10")),
        "timeout_seconds": float(os.getenv("GOOGLE_API_TIMEOUT_SECONDS", "12")),
    }


def _detect_location_from_public_ip(timeout_seconds: float) -> Tuple[float, float]:
    """Fallback IP-based location using a public endpoint (no API key required)."""
    url = "https://ipapi.co/json/"
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(url)
    response.raise_for_status()
    payload = response.json()
    lat = payload.get("latitude")
    lng = payload.get("longitude")
    if lat is None or lng is None:
        raise ValueError("Unable to detect location from IP fallback service.")
    return float(lat), float(lng)


def detect_location_from_ip() -> Tuple[float, float]:
    """Detect approximate user location using Google Geolocation API with public fallback."""
    cfg = _get_google_maps_config()
    if not cfg["api_key"]:
        return _detect_location_from_public_ip(cfg["timeout_seconds"])

    url = "https://www.googleapis.com/geolocation/v1/geolocate"
    try:
        with httpx.Client(timeout=cfg["timeout_seconds"]) as client:
            response = client.post(
                url,
                params={"key": cfg["api_key"]},
                json={"considerIp": True},
            )
        response.raise_for_status()
        payload = response.json()
        location = payload.get("location") or {}
        lat = location.get("lat")
        lng = location.get("lng")
        if lat is None or lng is None:
            raise ValueError("Unable to detect location automatically.")
        return float(lat), float(lng)
    except Exception:
        return _detect_location_from_public_ip(cfg["timeout_seconds"])


def geocode_location(location_query: str) -> Tuple[float, float]:
    cfg = _get_google_maps_config()
    if not cfg["api_key"]:
        raise ValueError("Missing GOOGLE_MAPS_API_KEY.")
    if not location_query.strip():
        raise ValueError("Please provide a location query.")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    with httpx.Client(timeout=cfg["timeout_seconds"]) as client:
        response = client.get(
            url,
            params={"address": location_query, "key": cfg["api_key"]},
        )
    response.raise_for_status()
    payload = response.json()

    status = payload.get("status")
    if status == "ZERO_RESULTS":
        raise ValueError("Location not found. Try a city, ZIP code, or full address.")
    if status != "OK":
        error_message = payload.get("error_message", "Geocoding failed.")
        raise ValueError(f"Geocoding error ({status}): {error_message}")

    first = payload["results"][0]["geometry"]["location"]
    return float(first["lat"]), float(first["lng"])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def fetch_nearby_hospitals(
    lat: float,
    lng: float,
    radius_meters: int | None = None,
    max_results: int | None = None,
) -> List[Dict[str, Any]]:
    cfg = _get_google_maps_config()
    if not cfg["api_key"]:
        raise ValueError("Missing GOOGLE_MAPS_API_KEY.")

    radius = radius_meters or cfg["radius_meters"]
    limit = max_results or cfg["max_results"]

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    with httpx.Client(timeout=cfg["timeout_seconds"]) as client:
        response = client.get(
            url,
            params={
                "location": f"{lat},{lng}",
                "radius": radius,
                "type": "hospital",
                "key": cfg["api_key"],
            },
        )
    response.raise_for_status()
    payload = response.json()

    status = payload.get("status")
    if status == "ZERO_RESULTS":
        return []
    if status != "OK":
        error_message = payload.get("error_message", "Nearby hospital lookup failed.")
        raise ValueError(f"Google Places error ({status}): {error_message}")

    hospitals: List[Dict[str, Any]] = []
    for item in payload.get("results", []):
        place_loc = (item.get("geometry") or {}).get("location") or {}
        place_lat = place_loc.get("lat")
        place_lng = place_loc.get("lng")
        if place_lat is None or place_lng is None:
            continue

        distance_km = _haversine_km(lat, lng, float(place_lat), float(place_lng))
        hospitals.append(
            {
                "name": item.get("name", "Unknown hospital"),
                "address": item.get("vicinity") or item.get("formatted_address", "N/A"),
                "rating": item.get("rating"),
                "user_ratings_total": item.get("user_ratings_total"),
                "distance_km": distance_km,
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{item.get('place_id', '')}",
            }
        )

    hospitals.sort(key=lambda x: x["distance_km"])
    return hospitals[:limit]


def find_nearby_hospitals(
    *,
    location_query: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
    radius_meters: int | None = None,
    max_results: int | None = None,
) -> Tuple[Tuple[float, float], List[Dict[str, Any]]]:
    if lat is None or lng is None:
        if location_query and location_query.strip():
            lat, lng = geocode_location(location_query.strip())
        else:
            lat, lng = detect_location_from_ip()

    hospitals = fetch_nearby_hospitals(
        lat=float(lat),
        lng=float(lng),
        radius_meters=radius_meters,
        max_results=max_results,
    )
    return (float(lat), float(lng)), hospitals


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
