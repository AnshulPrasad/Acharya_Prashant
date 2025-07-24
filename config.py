import os

CHANNEL_URLS = [
    "https://www.youtube.com/@AcharyaPrashant",
    # "https://www.youtube.com/@ShriPrashant",
    # "https://www.youtube.com/@AP_Shastragyaan_Hindi",
    # "https://www.youtube.com/@AP_lalkaar",
    # "https://www.youtube.com/@AP_RashtraDharma_Hindi",
    # "https://www.youtube.com/@AP_Shakti_Hindi",
    # "https://www.youtube.com/@AP_Prakrati_Hindi",
]

VTT_DIR = "data/subtitles_vtt"
TXT_DIR = "data/transcripts_txt"
FILE_PATHS = "data/file_paths.pkl"
TRANSCRIPTS = "data/transcripts.pkl"
TRANSCRIPT_INDEX = "data/transcript_index.faiss"
RETRIEVED_TRANSCRIPTS_FILE = "outputs/retrieved_transcripts.txt"
RESPONSE_FILE = "outputs/generated_response.txt"
COOKIES_FILE = "utils/youtube_cookies.txt"
API_URL = "https://models.github.ai/inference"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
GH_API_TOKEN = os.environ.get("GITHUB_TOKEN")
MODEL = "openai/gpt-4.1"
MAX_CONTEXT_TOKENS = 7000