import os, sys, logging, pickle, tiktoken
from utils.download_vtt import download_vtt
from utils.vtt_to_txt import vtt_to_txt
from utils.preprocess import clean_txt, preprocess
from api.retrieve_context import retrieve_transcripts
from api.generate_response import generate_response
from api.embed_transcripts import embedding
from config import (
    CHANNEL_URLS,
    VTT_DIR,
    TXT_DIR,
    COOKIES_FILE,
    TRANSCRIPT_INDEX,
    RETRIEVED_TRANSCRIPTS_FILE,
    RESPONSE_FILE,
    FILE_PATHS,
    TRANSCRIPTS,
    MAX_CONTEXT_TOKENS,
    MODEL
)

import pytz
from datetime import datetime

# 1) Define IST timezone and converter
IST = pytz.timezone("Asia/Kolkata")
def ist_time(*args):
    return datetime.now(IST).timetuple()

# 2) Create a handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)
formatter.converter = ist_time
handler.setFormatter(formatter)

# 3) Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any old handlers (so default UTC handler is gone)
root_logger.handlers = []
# Add our new handler
root_logger.addHandler(handler)



def download_all_vtts():
    for channel_url in CHANNEL_URLS:
        try:
            download_vtt(channel_url, VTT_DIR, language="en", cookies_file=COOKIES_FILE)
        except Exception as e:
            logging.error(f"Failed to download vtt for channel {channel_url}: {e}")


def preprocess_transcripts():
    vtt_to_txt(VTT_DIR, TXT_DIR)
    clean_txt(TXT_DIR)
    file_paths, transcripts = preprocess(TXT_DIR)

    with open(FILE_PATHS, "wb") as f:
        pickle.dump(file_paths, f)
    with open(TRANSCRIPTS, "wb") as f:
        pickle.dump(transcripts, f)


def build_embedding(transcripts):
    embedding(transcripts, TRANSCRIPT_INDEX)


def get_user_query():
    query = input("Enter query:\n").strip()
    if not query:
        logging.error("Query cannot be empty!")
        return None
    return query


def retrieve_contexts(query, file_paths, transcripts):
    retrieved_transcripts = retrieve_transcripts(
        query, file_paths, transcripts, 20
    )
    if not retrieved_transcripts:
        logging.warning("No transcripts retrieved.")
        return None
    return retrieved_transcripts


try:
    encoder = tiktoken.encoding_for_model(MODEL)
except KeyError:
    # fallback for custom or unrecognized model names
    encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in a string, using your model's tokenizer."""
    return len(encoder.encode(text))


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    """
    If text exceeds max_tokens, cut it down to the first max_tokens tokens.
    """
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    # decode only the first max_tokens tokens back into a string
    return encoder.decode(tokens[:max_tokens])


def write_response(response):
    try:
        with open(RESPONSE_FILE, "w") as resp_file:
            resp_file.write(response)
        logging.info(f"Response written to {RESPONSE_FILE}")
    except Exception as e:
        logging.error(f"Failed to write response: {e}")


def write_retrieved_transcripts(retrieved_transcripts, file_paths):
    try:
        with open(RETRIEVED_TRANSCRIPTS_FILE, "w") as f:
            for i, transcript in enumerate(retrieved_transcripts):
                video_id = os.path.splitext(os.path.basename(file_paths[i]))[0]
                f.write(f"Video id: {video_id}\nTranscript {i+1}:\n{transcript}\n")
    except Exception as e:
        logging.error(f"Failed to write transcripts: {e}")


if __name__ == "__main__":

    query = get_user_query()
    if not query:
        sys.exit()

    download_all_vtts()
    preprocess_transcripts()

    with open(FILE_PATHS, "rb") as f:
        file_paths = pickle.load(f)
    with open(TRANSCRIPTS, "rb") as f:
        transcripts = pickle.load(f)

    build_embedding(transcripts)

    retrieved_transcripts = retrieve_contexts(query, file_paths, transcripts)
    write_retrieved_transcripts(retrieved_transcripts, file_paths)
    if not retrieved_transcripts:
        sys.exit()

    full_context = " ".join(retrieved_transcripts)
    logging.info(
        f"Total number of tokens in full_context: {count_tokens(full_context)}"
    )
    logging.info(
        f"Total number of words in full_context: {len(full_context.split(' '))}"
    )
    limit_context = trim_to_token_limit(full_context, MAX_CONTEXT_TOKENS)
    logging.info(
        f"Total number of tokens in limit_context: {count_tokens(limit_context)}"
    )
    logging.info(
        f"Total number of words in limit_context: {len(limit_context.split(' '))}"
    )

    response = generate_response(query, limit_context)

    logging.info(f"Received query: {query}")
    context_str = ' '.join(limit_context.split('\n'))
    logging.info(f"Context: {context_str}")
    logging.info(f"Response: {response}")

    write_response(response)