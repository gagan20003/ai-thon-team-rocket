from langchain_classic.prompts import ChatPromptTemplate, PromptTemplate

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "You are a medical report summarizer. Use ONLY the provided context.\n"
        "If a detail is not in the context, omit it. Do not diagnose or prescribe.\n"
        "Write in plain language for a non-medical audience.\n\n"
        "Context:\n{context}\n\n"
        "Produce a structured summary with these sections:\n"
        "1) Report Overview\n"
        "2) Key Observations\n"
        "3) Abnormal Findings (bullet points only)\n"
        "4) Simple Explanation (what the findings could mean in everyday terms)\n\n"
        "For Abnormal Findings, use bullet points and include the test name, value, units,\n"
        "reference range if available, and why it's abnormal (high/low/flagged).\n"
        "Add a brief safety note encouraging consulting a licensed clinician."
    ),
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical report assistant. Use ONLY the provided context to answer.\n"
            "If the answer is not present in the context, respond exactly:\n"
            '"This information is not present in the report."\n'
            "Explain medical terms simply. Avoid diagnosis or prescriptions.\n"
            "Maintain a neutral, safe tone and encourage consulting a doctor when relevant.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

DOCTOR_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "You are a medical report assistant. Use ONLY the provided context.\n"
        "Generate a concise list of questions a patient could ask their doctor\n"
        "during a visit. Focus on abnormal or unclear findings and next steps.\n"
        "Do not diagnose or prescribe.\n\n"
        "Context:\n{context}\n\n"
        "Return 6-10 bullet points. Keep each question short and specific."
    ),
)
