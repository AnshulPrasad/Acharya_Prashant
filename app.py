import logging
import os
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.generate_response import generate_response
from api.retrieve_context import retrieve_transcripts
from utils.token import count_tokens, trim_to_token_limit
from config import FILE_PATHS, TRANSCRIPTS, MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["POST"], allow_headers=["*"])

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

        retrieved_transcripts = retrieve_transcripts(query, file_paths, transcripts, 15)
        if not retrieved_transcripts:
            return JSONResponse({"error": "No relevant transcripts found"}, status_code=404)

        full_context = " ".join(retrieved_transcripts)
        limit_context = trim_to_token_limit(full_context, MAX_CONTEXT_TOKENS)
        context_str = " ".join(limit_context.split("\n"))
        response = generate_response(query, limit_context)

        logger.info("Full_context: Tokens=%d, Words=%d", count_tokens(full_context), len(full_context.split(" ")))
        logger.info("Limit_context: Tokens=%d, Words=%d", count_tokens(limit_context), len(limit_context.split(" ")))

        return JSONResponse({"answer": response})

    except Exception as e:
        logger.exception("Internal error: %s",e)
        return JSONResponse({"error": "Internal server error. Please try again later."}, status_code=500)

# Serve frontend from 'frontend/' directory
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
