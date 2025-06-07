from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from generate_response import generate_response
from retrieve_context import retrieve_transcripts
from config import TXT_DIR, TRANSCRIPT_INDEX
from preprocess import preprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


file_paths, transcripts = preprocess(TXT_DIR)


@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            return JSONResponse({"error": "Query cannot be empty"}, status_code=400)
        retrieved_transcripts = retrieve_transcripts(
        query, file_paths, TRANSCRIPT_INDEX, 10
    )
        if not retrieved_transcripts:
            return JSONResponse(
                {"error": "No relevant transcripts found"}, status_code=404
            )
        context = " ".join(retrieved_transcripts)
        response = generate_response(query, context)
        return JSONResponse({"answer": response})
    except Exception as e:
        return JSONResponse({"error": f"Internal error: {str(e)}"}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Acharya Prashant API is running."}