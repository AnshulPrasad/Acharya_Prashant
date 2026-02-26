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
            n_ctx=4096,  # plenty for your RAG context
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
        full_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        answer = llm(
            full_prompt,
            max_tokens=512,
            temperature=0.25,
            top_p=0.95,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )
        answer = answer["choices"][0]["text"].strip()

        if not answer:
            logger.warning("Failed to generate response. Returning empty response.")
            return "I couldn't generate response. Please try again."

        logging.info('Answer Generation Succeeded.')
        return answer

    except Exception as e:
        logger.error("Failed to generate response: %s", e)
        return "Sorry, there was an error generating the response."