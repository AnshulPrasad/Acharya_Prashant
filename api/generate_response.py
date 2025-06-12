from transformers import pipeline, AutoTokenizer
import logging, requests
from config import HF_API_TOKEN, API_URL, MODEL

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


def generate_response(query, context):
    logging.info("Starting answer generation...")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    logging.info("Prepared prompt for generation.")

    payload = {
        "model": MODEL,  # or your preferred model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        try:
            result = response.json()
        except Exception as json_err:
            logging.error(f"Non-JSON response from HF API: {response.text}")
            return "Sorry, the model server returned an invalid response. Please try again later."
        # Parse OpenAI-style response
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"Error: {result['error']}"
        else:
            return str(result)
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return "Sorry, there was an error generating the response."
