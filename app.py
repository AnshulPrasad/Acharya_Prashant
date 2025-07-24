from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.generate_response import generate_response
from api.retrieve_context import retrieve_transcripts
from config import FILE_PATHS, TRANSCRIPTS, MAX_CONTEXT_TOKENS, MODEL
from fastapi.middleware.cors import CORSMiddleware
import traceback, logging, pickle, pytz, tiktoken
from datetime import datetime

# 1) Define IST timezone and converter
IST = pytz.timezone("Asia/Kolkata")


def ist_time(*args):
    return datetime.now(IST).timetuple()


# 2) Create a handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's URL for more security
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


with open(FILE_PATHS, "rb") as f:
    file_paths = pickle.load(f)
with open(TRANSCRIPTS, "rb") as f:
    transcripts = pickle.load(f)


@app.post("/ask")
async def ask_question(request: Request):
    try:
        logging.info("\n")
        data = await request.json()

        query = data.get("query")
        logging.info(f"Received query:\n{query}")
        if not query:
            return JSONResponse({"error": "Query cannot be empty"}, status_code=400)
        
        retrieved_transcripts = retrieve_transcripts(query, file_paths, transcripts, 15)
        if not retrieved_transcripts:
            return JSONResponse(
                {"error": "No relevant transcripts found"}, status_code=404
            )
        
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
        logging.info(f"Context:\n{limit_context}")

        response = generate_response(query, limit_context)
        logging.info(f"Response:\n{response}")

        return JSONResponse({"answer": response})

    except Exception as e:
        logging.error(f"Internal error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            {"error": "Internal server error. Please try again later."}, status_code=500
        )


from fastapi.staticfiles import StaticFiles
import os

# Serve frontend from 'frontend/' directory
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
