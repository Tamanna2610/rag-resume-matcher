import requests
from retrieve import retrieve_relevant_chunks

import os
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
MODEL_NAME = "llama3.2:1b"  

def generate_tailored_bullets(job_description, top_k=4):
    chunks = retrieve_relevant_chunks(job_description, top_k=top_k)

    context = "\n\n".join([f"{c['title']}:\n{c['text']}" for c in chunks])

    prompt = f"""You are a resume-writing assistant. A candidate has the following relevant experience:

{context}

Here is the job description they're applying to:

{job_description}

Based ONLY on the experience provided above, write 3-4 tailored resume bullet points that highlight the most relevant parts of this candidate's background for this specific job. Keep the original facts/metrics accurate -- do not invent numbers or skills not mentioned. Use strong action verbs and quantify impact where the original text already does."""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    return response.json()["response"]

if __name__ == "__main__":
    sample_jd = """
    We are looking for a Machine Learning Engineer with experience in Python,
    PyTorch, model evaluation, and credit risk or financial data. Experience
    with adversarial robustness or model security is a plus.
    """

    result = generate_tailored_bullets(sample_jd)
    print(result)