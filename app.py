from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.generate_response import generate_response
from api.retrieve_context import retrieve_transcripts
from config import FILE_PATHS, TRANSCRIPTS, MAX_CONTEXT_WORDS
from fastapi.middleware.cors import CORSMiddleware
import traceback, logging, pickle, pytz
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
        data = await request.json()
        query = data.get("query")
        if not query:
            return JSONResponse({"error": "Query cannot be empty"}, status_code=400)
        retrieved_transcripts = retrieve_transcripts(
            query, file_paths, transcripts, 10
        )
        if not retrieved_transcripts:
            return JSONResponse(
                {"error": "No relevant transcripts found"}, status_code=404
            )

        full_context = " ".join(retrieved_transcripts)
        logging.info(
            f"Number of characters in retrieved_transcripts separated by space: {len(full_context.split(' '))}"
        )
        if len(full_context.split(" ")) >= MAX_CONTEXT_WORDS:
            limit_context = " ".join(
                full_context.split(" ")[:MAX_CONTEXT_WORDS]
            )
        else:
            limit_context = full_context

        response = generate_response(query, limit_context)
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
