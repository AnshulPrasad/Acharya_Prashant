from openai import OpenAI
import logging
from config import API_URL, MODEL, GH_API_TOKEN

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


try:
    client = OpenAI(
        base_url=API_URL,
        api_key=GH_API_TOKEN,
        timeout=10
    )
    logging.info("OpenAI client initialized.")
except Exception as e:
    logging.critical(f"Failed to initialize OpenAI client: {e}")
    client = None


def generate_response(query, context):

    if client is None:
        return "Error: AI client not configured."

    logging.info("Starting answer generation...")

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    logging.info("Prepared prompt for generation.")

    try:

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

        response = response.choices[0].message.content
        logging.info("Answer generation succeeded.")
        return response

    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return "Sorry, there was an error generating the response."
