import hashlib
import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    answer_question,
    build_conversational_chain,
    build_vector_store,
    find_nearby_hospitals,
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

if not os.getenv("LITELLM_API_KEY"):
    st.error("Missing LITELLM_API_KEY. Add it to your .env file.")
    st.stop()

if not os.getenv("GOOGLE_MAPS_API_KEY"):
    st.warning("GOOGLE_MAPS_API_KEY is missing. Nearby Hospitals feature will be unavailable.")


@st.cache_resource(show_spinner=False)
def build_index(file_bytes: bytes):
    docs = load_pdf_from_bytes(file_bytes)
    if not docs or all(not d.page_content.strip() for d in docs):
        raise ValueError("The PDF appears to be empty or unreadable.")
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")
    return build_vector_store(chunks)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = ""
    if "doctor_questions" not in st.session_state:
        st.session_state.doctor_questions = ""
    if "nearby_hospitals" not in st.session_state:
        st.session_state.nearby_hospitals = []
    if "origin_location" not in st.session_state:
        st.session_state.origin_location = None


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
        st.session_state.doctor_questions = ""
        st.session_state.nearby_hospitals = []
        st.session_state.origin_location = None

    try:
        with st.spinner("Processing report..."):
            vector_store = build_index(file_bytes)
            llm = get_llm()
            if not st.session_state.summary:
                st.session_state.summary = summarize_report(vector_store, llm)
        st.success("Report indexed successfully.")
    except Exception as exc:
        st.error(f"Failed to process the PDF: {exc}")
        st.stop()

    st.subheader("Summary")
    st.info(st.session_state.summary)

    if st.button("Generate Doctor Visit Questions"):
        with st.spinner("Generating questions..."):
            st.session_state.doctor_questions = generate_doctor_questions(
                vector_store, llm
            )

    if st.session_state.doctor_questions:
        st.subheader("Doctor Visit Questions")
        st.markdown(st.session_state.doctor_questions)

    # --- Nearby Hospitals (auto-fetch on first load) ---
    st.subheader("Nearby Hospitals")

    if os.getenv("GOOGLE_MAPS_API_KEY") and not st.session_state.origin_location:
        try:
            with st.spinner("Detecting your location and finding nearby hospitals..."):
                origin, hospitals = find_nearby_hospitals()
                st.session_state.origin_location = origin
                st.session_state.nearby_hospitals = hospitals
        except Exception as exc:
            st.warning(f"Auto-detection failed: {exc}. You can search manually below.")

    if st.session_state.origin_location:
        lat, lng = st.session_state.origin_location
        st.caption(f"Detected location: {lat:.5f}, {lng:.5f}")

    with st.expander("Refine hospital search", expanded=False):
        location_query = st.text_input(
            "Override location (ZIP, city, or full address)",
            value="",
            placeholder="e.g., Nashville, TN or 37203",
        )
        radius_meters = st.slider(
            "Search radius (meters)",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=500,
        )
        max_results = st.slider(
            "Max hospitals", min_value=3, max_value=20, value=10, step=1
        )
        if st.button("Search Again"):
            if not os.getenv("GOOGLE_MAPS_API_KEY"):
                st.error("Missing GOOGLE_MAPS_API_KEY in .env")
            else:
                try:
                    with st.spinner("Finding nearby hospitals..."):
                        origin, hospitals = find_nearby_hospitals(
                            location_query=location_query if location_query.strip() else None,
                            radius_meters=radius_meters,
                            max_results=max_results,
                        )
                        st.session_state.origin_location = origin
                        st.session_state.nearby_hospitals = hospitals
                except Exception as exc:
                    st.error(f"Unable to fetch nearby hospitals: {exc}")

    if st.session_state.nearby_hospitals:
        for idx, hospital in enumerate(st.session_state.nearby_hospitals, start=1):
            rating = hospital.get("rating")
            ratings_total = hospital.get("user_ratings_total")
            rating_text = (
                f"{rating} ({ratings_total} reviews)"
                if rating is not None and ratings_total is not None
                else "N/A"
            )
            st.markdown(
                f"**{idx}. {hospital['name']}**  \n"
                f"📍 {hospital['distance_km']:.2f} km away  \n"
                f"📫 {hospital['address']}  \n"
                f"⭐ {rating_text}  \n"
                f"[Open in Google Maps]({hospital['maps_url']})"
            )
            st.divider()
    elif st.session_state.origin_location:
        st.info("No hospitals found in the selected radius. Try increasing the search radius.")

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
