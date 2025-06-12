import os
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


def preprocess(txt_dir):
    """
    Store path and content of each .txt file in the directory in two separate lists and
    return them
    Args:
        txt_dir (str): directory in which the .txt files are be present
    """
    logging.info("preprocessing starts!")
    transcripts = []
    file_paths = []

    for file_name in os.listdir(txt_dir):
        file_path = os.path.join(txt_dir, file_name)

        with open(file_path, "r") as f:
            text = f.read()
            transcripts.append(text)
            file_paths.append(file_path)
    logging.info("file_paths and transcripts are ready to be returned!")
    return file_paths, transcripts


def extract_transcript(vtt_file, txt_file):
    keywords = (
        "WEBVTT",
        "Kind",
        "Language",
        "-->",
        "<",
        "[Music]",
        "[music]",
        "[Applause]",
        "[Laughter]",
        "[Cheering]",
        "[Clapping]",
        "[Audience Laughing]",
        "[Audience Applause]",
        "[Background Noise]",
        "[प्रशंसा]",
        "[तालियाँ]",
        "[संगीत]",
        "[हँसी]",
        "[जयकारे]",
        "[शोर]",
        "[पृष्ठभूमि संगीत]",
    )

    with open(vtt_file, "r") as vtt, open(txt_file, "w") as txt:
        for line in vtt:
            if any(keyword in line for keyword in keywords) or line.strip() == "":
                continue
            else:
                txt.write(line.strip() + "\n")
    logging.info(f"Conversion completed for file: {vtt_file}")


def clean_txt(txt_dir):
    """
    Remove consequtive duplicate lines and
    rewrite them to the same file

    Args:
        txt_dir (str): directory in which the .txt files are saved
    """

    for file_name in os.listdir(txt_dir):

        if file_name.endswith(".txt"):
            file_path = os.path.join(txt_dir, file_name)

            with open(file_path, "r") as f:
                lines = f.readlines()
                cleaned_lines = []
                flag = 1

                for idx, line in enumerate(lines):
                    if (
                        idx != len(lines) - 1
                        and lines[idx].strip() == lines[idx + 1].strip()
                    ):
                        continue
                    else:
                        cleaned_lines.append(line)

            with open(file_path, "w") as f:
                f.seek(0)
                f.writelines(cleaned_lines)
    logging.info(f"Cleaned {txt_dir.split('/')[-1]}\n")
