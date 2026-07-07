import streamlit as st
import tempfile
import os
import sys

sys.path.append("src")

from parser import extract_text_from_resume
from chunker import build_corpus
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="ResumeRAG", page_icon="📄")
st.title("📄 ResumeRAG")
st.write("Upload your resume and paste a job description to get tailored, accurate bullet points.")

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

def build_session_vectorstore(corpus):
    """Builds a fresh, temporary in-memory vector store for this session only."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection("session_resume")

    model = load_embed_model()
    texts = [c["text"] for c in corpus]
    ids = [c["id"] for c in corpus]
    metadatas = [{"title": c["title"] or "", "category": c["category"]} for c in corpus]
    embeddings = model.encode(texts).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection

def retrieve_chunks(collection, job_description, top_k=4):
    model = load_embed_model()
    jd_embedding = model.encode([job_description]).tolist()
    results = collection.query(query_embeddings=jd_embedding, n_results=top_k)

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "title": results["metadatas"][0][i]["title"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    return matches

def generate_bullets(job_description, chunks):
    context = "\n\n".join([f"{c['title']}:\n{c['text']}" for c in chunks])

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a resume-writing assistant. Write tailored resume bullets based only on the experience provided. Keep all facts and metrics accurate — do not invent anything."
            },
            {
                "role": "user",
                "content": f"""The candidate has the following relevant experience:

{context}

Here is the job description they're applying to:
{job_description}

Write 3-4 tailored resume bullet points that highlight the most relevant parts of this candidate's background for this specific job. Use strong action verbs and quantify impact where the original text already does."""
            }
        ]
    )
    return response.choices[0].message.content

# --- UI ---

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description", height=200)

if st.button("Generate Tailored Bullets"):
    if not uploaded_file or not job_description.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Parsing resume..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            resume_text = extract_text_from_resume(tmp_path)
            corpus = build_corpus(resume_text)
            os.unlink(tmp_path)

        if not corpus:
            st.error("Couldn't extract job/project sections from this resume. Try a different formatting.")
        else:
            with st.spinner("Embedding and retrieving relevant experience..."):
                collection = build_session_vectorstore(corpus)
                matches = retrieve_chunks(collection, job_description)

            with st.expander("🔍 See which experience was matched"):
                for m in matches:
                    st.markdown(f"**{m['title']}** _(similarity score: {m['distance']:.3f})_")
                    st.write(m["text"][:200] + "...")

            with st.spinner("Generating tailored bullets..."):
                result = generate_bullets(job_description, matches)

            st.subheader("✨ Tailored Resume Bullets")
            st.write(result)