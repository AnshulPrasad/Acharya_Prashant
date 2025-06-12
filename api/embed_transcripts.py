import faiss, logging, os
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


def embedding(transcripts, transcript_index):
    logging.info("Starting embedding of transcripts...")
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    logging.info("Loaded embedding model.")

    transcripts_embeddings = embedding_model.encode(
        transcripts, convert_to_tensor=False, show_progress_bar=True
    )
    logging.info(f"Generated embeddings for {len(transcripts)} transcripts.")

    dimension = transcripts_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    logging.info("Initialized FAISS index.")

    index.add(transcripts_embeddings)
    logging.info("Added embeddings to FAISS index.")


    os.makedirs(os.path.dirname(transcript_index), exist_ok=True)
    faiss.write_index(index, transcript_index)
    logging.info(f"FAISS index written to {transcript_index}.\n")
