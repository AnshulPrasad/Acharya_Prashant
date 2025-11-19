import subprocess, os, logging, time, json

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


def download_vtt(
    channel_url: str,
    output_dir: str,
    language="en",
    cookies_file="www.youtube.com_cookies.txt",
):
    """
    Download subtitles of all the videos from a Youtube channel

    Args:
        channel_url (str): url of the youtube channel
        output_dir (str): output directory where the subtitles will be stored
        language (str): laguage of the subtile to be retrieved
        cookies_file (str): the cookie file to bipass the authentication in age restricted content
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    command_list = ["yt-dlp", "--flat-playlist", "--get-id", channel_url]

    try:
        result = subprocess.run(
            command_list, capture_output=True, text=True, check=True
        )
        video_ids = result.stdout.strip().split("\n")
        logging.info(f"Found video IDs: {video_ids}\n")

    except Exception as e:
        logging.error(f"Error listing videos: {e}")
        return

    for video_id in video_ids:
        subtitle_path = os.path.join(output_dir, f"{video_id}.{language}.vtt")
        if os.path.exists(subtitle_path):
            logging.info(f"Skipping already downloaded video: {video_id}")
            continue

        # 1. Get metadata for the video
        meta_cmd = ["yt-dlp", "-J", f"https://www.youtube.com/watch?v={video_id}"]
        try:
            meta_result = subprocess.run(
                meta_cmd, capture_output=True, text=True, check=True
            )
            meta = meta_result.stdout

            info = json.loads(meta)
        except Exception as e:
            logging.warning(f"Could not fetch metadata for {video_id}: {e}")
            continue

        # 2. Check for manual or auto subtitles
        has_manual = language in (info.get("subtitles") or {})
        has_auto = language in (info.get("automatic_captions") or {})

        if has_manual:
            logging.info(
                f"Manual {language} subtitles available for {video_id}, downloading..."
            )
            command = [
                "yt-dlp",
                "--write-sub",
                "--sub-lang",
                language,
                "--skip-download",
                "--cookies",
                cookies_file,
                "-o",
                f"{output_dir}/{video_id}.%(ext)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
        elif has_auto:
            logging.info(
                f"Only auto {language} subtitles available for {video_id}, downloading..."
            )
            command = [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang",
                language,
                "--skip-download",
                "--cookies",
                cookies_file,
                "-o",
                f"{output_dir}/{video_id}.%(ext)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
        else:
            logging.warning(
                f"No {language} subtitles (manual or auto) for video: {video_id}\n"
            )
            continue

        try:
            subprocess.run(command, check=True)
            if os.path.exists(subtitle_path):
                logging.info(f"Subtitle downloaded for video: {video_id}\n")
            else:
                logging.warning(
                    f"Subtitle download command ran but file not found for video: {video_id}\n"
                )
        except Exception as e:
            logging.warning(f"Subtitle download failed for video: {video_id}: {e}\n")

        time.sleep(2)
