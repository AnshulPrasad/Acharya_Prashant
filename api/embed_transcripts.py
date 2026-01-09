import faiss
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def embedding(transcripts: list[str], transcript_index: Path) -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(transcripts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    transcript_index.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(transcript_index))
    logger.info("Embedding completed.\n")
