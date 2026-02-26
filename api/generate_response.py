import logging
import os
from llama_cpp import Llama

from utils.token import count_tokens
from config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

llm = None

def load_model_at_startup():
    global llm
    try:
        logger.info("Loading Qwen2.5-1.5B model into RAM...")

        llm = Llama.from_pretrained(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            n_threads=4,
            n_gpu_layers=0,  # CPU only (safe for HF Spaces)
            verbose=True,
            n_ctx=16384,  # plenty for your RAG context
        )
        logger.info("Qwen2.5-1.5B model loaded into RAM successfully.")

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        llm = None

def generate_response(query: str, context: str) -> str:

    if llm is None:
        return "Error: Model not loaded.."

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    logging.info("Total number of tokens in prompt: %s", count_tokens(prompt))

    try:
        answer = llm(
            f"[SYSTEM]{SYSTEM_PROMPT}[/SYSTEM]\n{prompt}",
            max_tokens=2000,
            temperature=1.0,
            top_p=1.0,
            stop=["<|end|>", "Question:", "<|user|>"],
            echo=False
        )
        answer = answer["choices"][0]["text"].strip()

        if not answer:
            logger.warning("Failed to generate response. Returning empty response.")
            return "I couldn't generate response. Please try again."

        logging.info('Answer Generation Succeeded.')
        return answer

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return "Sorry, there was an error generating the response."