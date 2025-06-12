import os
import logging
from utils.preprocess import extract_transcript

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


def vtt_to_txt(vtt_dir: str, txt_dir: str):
    """
    Convert Youtube file format .vtt (WebVTT or Web Video Text to Track) to text file format .txt

    Args:
        vtt_dir (str): directory which contain all the .vtt files
        txt_dir (str): directory in which the converted .txt files will be saved
    """

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    for file_name in os.listdir(vtt_dir):
        if file_name.endswith(".vtt"):

            vtt_file = os.path.join(vtt_dir, file_name)
            txt_file = os.path.join(txt_dir, os.path.splitext(file_name)[0] + ".txt")

            if os.path.exists(txt_file):
                continue

            extract_transcript(vtt_file, txt_file)

    logging.info(
        f"Completed {os.path.basename(vtt_dir)} to {os.path.basename(txt_dir)} conversion."
    )
