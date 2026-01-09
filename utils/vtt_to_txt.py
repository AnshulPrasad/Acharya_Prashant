import logging
from pathlib import Path
from utils.preprocess import vtt_to_clean_text

logger = logging.getLogger(__name__)


def vtt_to_txt(vtt_dir: Path, txt_dir: Path) -> None:
    """
    Convert Youtube file format .vtt (WebVTT or Web Video Text to Track) to text file format .txt

    Args:
        vtt_dir (Path): directory which contain all the .vtt files
        txt_dir (Path): directory in which the converted .txt files will be saved
    """

    txt_dir.mkdir(parents=True, exist_ok=True)

    for vtt_path in vtt_dir.glob("*.vtt"):
        txt_path = txt_dir / vtt_path.with_suffix(".txt").name

        if txt_path.exists():
            logger.info("Skipping %s (already exists)", txt_path.name)
            continue

        vtt_to_clean_text(vtt_path, txt_path)

    logger.info("Completed %s → %s conversion", vtt_dir.name, txt_dir.name)
