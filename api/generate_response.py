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
        logger.info("Loading model into RAM...")

        llm = Llama.from_pretrained(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            verbose=False,
            n_gpu_layers=0,  # CPU only (safe for HF Spaces)
            n_ctx=2048,
        )
        logger.info("Model loaded into RAM successfully.")

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
            max_tokens=7000,
            temperature=1.0,
            top_p=1.0,
            stop=["Question:", "Context:"]
        )
        answer = answer["choices"][0]["text"].strip()
        logging.info('Answer Generation Succeeded.')
        return answer

    except Exception as e:
        logging.error("Error during inference",)
        return "Sorry, there was an error generating the response."