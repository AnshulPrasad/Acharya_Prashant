from openai import OpenAI
import logging, tiktoken, random, threading, time
from config import API_URL, MODEL, GH_API_TOKEN

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

try:
    encoder = tiktoken.encoding_for_model(MODEL)
except KeyError:
    # fallback for custom or unrecognized model names
    encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in a string, using your model's tokenizer."""
    return len(encoder.encode(text))


try:
    client = OpenAI(base_url=API_URL, api_key=GH_API_TOKEN, timeout=60)
    logging.info("OpenAI client initialized.")
except Exception as e:
    logging.critical(f"Failed to initialize OpenAI client: {e}")
    client = None

# --- Minimal concurrency limiter (per-process) ---
# Tune this to 1 or 2 depending on your provider/account limits.
SEMAPHORE_LIMIT = 1
_semaphore = threading.Semaphore(SEMAPHORE_LIMIT)


def generate_response(query, context, max_retries: int = 4, base_backoff: float = 0.5):

    if client is None:
        return "Error: AI client not configured."

    logging.info("Starting answer generation...")

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    logging.info("Prepared prompt for generation.")
    logging.info(f"Total number of tokens in prompt: {count_tokens(prompt)}")

    for attempt in range(max_retries):
        try:
            # Acquire semaphore (blocks this thread until allowed)
            with _semaphore:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=1,
                    top_p=1,
                    model=MODEL,
                    stream=False,
                )

            # Extract text defensively (depends on SDK return shape)
            try:

                response = response.choices[0].message.content
            except Exception:
                response = getattr(response, "text", None) or str(response)
                logging.warning("Fallback used for response parsing.")

            logging.info("Answer generation succeeded.")
            return response

        except Exception as e:
            msg = str(e)
            # Heuristic detection for rate-limit / 429
            is_rate_limit = (
                "429" in msg
                or "RateLimit" in msg
                or "Rate limit" in msg
                or "RateLimitReached" in msg
            )

            if is_rate_limit and attempt < max_retries - 1:
                wait = base_backoff * (2**attempt) + random.random() * 0.1
                logging.warning(
                    f"Rate limited by API (attempt {attempt+1}/{max_retries}). "
                    f"Sleeping {wait:.2f}s before retry. Error: {msg}"
                )
                time.sleep(wait)
                continue
            # Non-retryable error or retries exhausted
            logging.error(f"Error during API call: {e}")
            return "Sorry, there was an error generating the response."
    return "Sorry, there was an error generating the response."
