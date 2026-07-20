# ResumeRAG 📄

A RAG-powered tool that analyzes your resume against any job description and generates tailored, accurate bullet points — plus adversarial stress-testing of the retrieval pipeline.

Live Demo: https://rag-resume-matcher-fjuhehklpo4qflk33lqgxc.streamlit.app

---

## What it does

Upload your resume (PDF) + paste a job description → the tool retrieves your most semantically relevant experience and generates tailored resume bullets using an LLM — without hallucinating or inventing credentials.

---

## Architecture

```
Resume PDF → Parser (pypdf) → Chunker → Embedder (all-MiniLM-L6-v2)
                                              ↓
                                        ChromaDB (in-memory, per session)
                                              ↓
Job Description → Embedding → Vector Similarity Search → Top-k Chunks
                                              ↓
                              Groq LLaMA 3.3 70B → Tailored Bullets
```

---

## Adversarial Testing

Beyond building the RAG pipeline, I stress-tested it against three attack types inspired by my prior research in adversarial ML (BATT backdoor attacks on CIFAR-10/ResNet-18):

| Attack | Description | Pre-Defense | Post-Defense |
|--------|-------------|-------------|--------------|
| Prompt Injection | Hidden instructions in JD hijack LLM output | ⚠️ Succeeded | ✅ Neutralized |
| Retrieval Poisoning | Malicious chunk ranked #1 in vector search | ⚠️ Succeeded | ✅ Neutralized |
| Context Manipulation | Crafted JD forces irrelevant chunks to surface | ⚠️ Succeeded | ⚠️ Partially mitigated |

### Defenses implemented

- **Input sanitization:** regex-based detection and redaction of 12 injection patterns before JD reaches the pipeline
- **Hardened system prompt:** explicit LLM instructions to ignore embedded commands and never fabricate credentials
- **Output safety checker:** post-generation scan for fabricated credentials before returning results to user
- **Content-based chunk filtering:** pattern matching on retrieved chunks to remove adversarial injections before LLM context assembly

### Key finding

Context manipulation remains partially unmitigated — when a crafted JD semantically matches an irrelevant chunk with high confidence (distance 0.75 vs next chunk at 1.74), similarity thresholding alone cannot distinguish intent. This is a known open problem in RAG security and reflects real-world limitations of embedding-based retrieval.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | pypdf |
| Chunking | Custom regex-based section parser |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (ephemeral, per-session) |
| LLM | Groq API (LLaMA 3.3 70B Versatile) |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## Run locally

```bash
git clone https://github.com/Tamanna2610/rag-resume-matcher.git
cd rag-resume-matcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

### Run adversarial tests

```bash
# Attack suite
python tests/adversarial_tests.py

# Defense suite
python tests/test_mitigations.py
```

---

## Project structure

```
rag-resume-matcher/
├── app.py                        # Streamlit UI
├── src/
│   ├── parser.py                 # PDF text extraction
│   ├── chunker.py                # Resume section chunking
│   ├── vectorstore.py            # Embedding + ChromaDB storage
│   ├── retrieve.py               # JD-to-chunk similarity search
│   ├── generate.py               # LLM bullet generation (local)
│   └── mitigations.py            # Adversarial defenses
├── tests/
│   ├── adversarial_tests.py      # Attack suite
│   └── test_mitigations.py       # Defense suite
├── data/
│   ├── adversarial_results.json  # Attack results
│   └── defense_results.json      # Defense results
└── requirements.txt
```