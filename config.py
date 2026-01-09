import os
from pathlib import Path

CHANNEL_URLS = [
    "https://www.youtube.com/@AcharyaPrashant",
    "https://www.youtube.com/@ShriPrashant",
]

VTT_DIR = Path("data/subtitles_vtt")
TXT_DIR = Path("data/transcripts_txt")
FILE_PATHS = Path("data/file_paths.pkl")
TRANSCRIPTS = Path("data/transcripts.pkl")
TRANSCRIPT_INDEX = "data/transcript_index.faiss"
RETRIEVED_TRANSCRIPTS_FILE = Path("outputs/retrieved_transcripts.txt")
RESPONSE_FILE = Path("outputs/generated_response.txt")
COOKIES_FILE = Path("utils/youtube_cookies.txt")
API_URL = "https://models.github.ai/inference"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
GH_API_TOKEN = os.environ.get("GITHUB_TOKEN")
MODEL = "openai/gpt-4.1"
MAX_CONTEXT_TOKENS = 7000
