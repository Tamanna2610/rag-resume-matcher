import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def get_collection(collection_name="resume_chunks"):
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection(collection_name)

def retrieve_relevant_chunks(job_description, top_k=4):
    """Embeds a job description and retrieves the most relevant resume chunks."""
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()

    jd_embedding = model.encode([job_description]).tolist()

    results = collection.query(
        query_embeddings=jd_embedding,
        n_results=top_k
    )

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "category": results["metadatas"][0][i]["category"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]  # lower = more similar
        })
    return matches

if __name__ == "__main__":
    sample_jd = """
    We are looking for a Machine Learning Engineer with experience in Python,
    PyTorch, model evaluation, and credit risk or financial data. Experience
    with adversarial robustness or model security is a plus.
    """

    matches = retrieve_relevant_chunks(sample_jd, top_k=4)

    print("Top matches for sample JD:\n")
    for m in matches:
        print(f"[{m['distance']:.4f}] {m['title']}")
        print(m["text"][:150], "...\n")