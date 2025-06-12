from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.generate_response import generate_response
from api.retrieve_context import retrieve_transcripts
from config import TRANSCRIPT_INDEX, FILE_PATHS, TRANSCRIPTS
from fastapi.middleware.cors import CORSMiddleware
import traceback, logging, pickle

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
        query, file_paths, transcripts, TRANSCRIPT_INDEX, 5
    )
        if not retrieved_transcripts:
            return JSONResponse(
                {"error": "No relevant transcripts found"}, status_code=404
            )
        context = " ".join(retrieved_transcripts)
        response = generate_response(query, context)
        return JSONResponse({"answer": response})
    except Exception as e:
        logging.error(f"Internal error: {e}\n{traceback.format_exc()}")
        return JSONResponse({"error": "Internal server error. Please try again later."}, status_code=500)


from fastapi.staticfiles import StaticFiles
import os

# Serve frontend from 'frontend/' directory
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
