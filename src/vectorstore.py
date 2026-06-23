import json
import chromadb
from sentence_transformers import SentenceTransformer

# Small, fast, good-quality embedding model — runs locally, no API key needed
MODEL_NAME = "all-MiniLM-L6-v2"

def load_corpus(path="data/corpus.json"):
    with open(path, "r") as f:
        return json.load(f)

def build_vector_store(corpus, collection_name="resume_chunks"):
    model = SentenceTransformer(MODEL_NAME)

    # Persistent client -- saves the DB to disk in chroma_db/
    client = chromadb.PersistentClient(path="chroma_db")

    # Wipe any old collection with the same name so re-runs don't duplicate data
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    texts = [chunk["text"] for chunk in corpus]
    ids = [chunk["id"] for chunk in corpus]
    metadatas = [{"title": chunk["title"] or "", "category": chunk["category"]} for chunk in corpus]

    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(texts)} chunks in vector DB collection '{collection_name}'")
    return collection

if __name__ == "__main__":
    corpus = load_corpus()
    build_vector_store(corpus)