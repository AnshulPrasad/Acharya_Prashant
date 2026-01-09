import faiss
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import TRANSCRIPT_INDEX

logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(TRANSCRIPT_INDEX)


def retrieve_transcripts(query: str, file_paths: list[Path], transcripts: list[str], top_k: int = 3) -> list[str]:
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(transcripts[idx])
            logger.info(f"Retrieved transcript from: {file_paths[idx]}")

    logger.info("Retrieval process completed.")
    return results
