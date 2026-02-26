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
        logger.info("Loading Phi-3-mini model into RAM...")

        llm = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
            verbose=True,
            n_gpu_layers=0,  # CPU only (safe for HF Spaces)
            n_ctx=4096,
        )
        logger.info("Phi-3-mini model loaded into RAM successfully.")

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