import os

CHANNEL_URLS = [
    "https://www.youtube.com/@AcharyaPrashant",
    "https://www.youtube.com/@ShriPrashant",
    "https://www.youtube.com/@AP_Shastragyaan_Hindi",
    "https://www.youtube.com/@AP_lalkaar",
    "https://www.youtube.com/@AP_RashtraDharma_Hindi",
    "https://www.youtube.com/@AP_Shakti_Hindi",
    "https://www.youtube.com/@AP_Prakrati_Hindi",
]

VTT_DIR = "data/subtitles_vtt"
TXT_DIR = "data/transcripts_txt"
COOKIES_FILE = "data/youtube_cookies.txt"
TRANSCRIPT_INDEX = "models/transcript_index_en.faiss"
RETRIEVED_TRANSCRIPTS_FILE = "outputs/retrieved_transcripts_en.txt"
RESPONSE_FILE= "outputs/generated_response.txt"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
MODEL = "deepseek/deepseek-v3-0324"
