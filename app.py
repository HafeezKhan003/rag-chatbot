import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ---------------- LOAD API KEY ----------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If running on Streamlit Cloud, use Secrets
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("❌ GROQ_API_KEY not found.")
        st.stop()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (Groq + FAISS)")

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- LOAD VECTOR DB ----------------
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------------- GROQ LLM ----------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)

# ---------------- RAG FUNCTION ----------------
def ask_rag(question):
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found in the document."

    context = "\n\n".join(
        d.page_content.replace("\n", " ")
        for d in docs
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

If answer is not in context, say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer briefly (2–5 sentences):
"""

    response = llm.invoke(prompt)
    return response.content

# ---------------- CHAT UI ----------------
user_question = st.text_input("Ask a question from your PDF:")

if user_question:
    with st.spinner("Thinking..."):
        start = time.time()
        answer = ask_rag(user_question)
        end = time.time()

    st.success(f"Done in {round(end - start, 2)}s")
    st.markdown("### Answer")
    st.write(answer)