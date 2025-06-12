import numpy as np
import faiss, logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


def retrieve_transcripts(query, file_path, transcripts, transcript_index, top_k=3):
    logging.info("Starting retrieval process...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Loaded embedding model for query.")

    index = faiss.read_index(transcript_index)
    logging.info(f"Loaded FAISS index from {transcript_index}.")

    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    logging.info("Encoded query to embedding.")

    distances, indices = index.search(np.array(query_embedding), top_k)
    logging.info(f"Retrieved top {top_k} results from index.")

    results = []
    for idx in indices[0]:
        results.append(transcripts[idx])
        logging.info(f"Retrieved transcript from: {file_path[idx]}")

    logging.info("Retrieval process completed.\n")
    return results
