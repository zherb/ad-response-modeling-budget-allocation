import os
import tarfile
from pathlib import Path
from typing import Tuple, Optional

import gdown
import pandas as pd

from .config import Paths, ensure_dirs

def download_tarball(gdrive_url: str, output_path: Path, quiet: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(gdrive_url, str(output_path), quiet=quiet, fuzzy=True)
    if not output_path.exists():
        raise FileNotFoundError(f"Download failed: {output_path}")
    return output_path

def safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as t:
        for member in t.getmembers():
            member_path = extract_dir / member.name
            # block path traversal
            if not str(member_path.resolve()).startswith(str(extract_dir.resolve())):
                raise RuntimeError(f"Blocked suspicious path in tar: {member.name}")
        t.extractall(extract_dir)

def load_parquet_tables(extract_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events_path = extract_dir / "events"
    impressions_path = extract_dir / "impressions"
    if not events_path.exists() or not impressions_path.exists():
        raise FileNotFoundError("Expected parquet directories 'events' and 'impressions' inside extracted tar.")
    events_df = pd.read_parquet(events_path)
    impressions_df = pd.read_parquet(impressions_path)
    return events_df, impressions_df

def ingest(
    gdrive_url: Optional[str],
    tar_filename: str = "hw_data.tar.gz",
    remove_tar_after: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = Paths()
    ensure_dirs(paths)

    tar_path = paths.raw_dir / tar_filename

    if gdrive_url:
        download_tarball(gdrive_url, tar_path, quiet=False)
    else:
        if not tar_path.exists():
            raise FileNotFoundError(
                f"No tarball found at {tar_path}. Provide --gdrive-url or place tarball there."
            )

    safe_extract_tar(tar_path, paths.extracted_dir)
    events_df, impressions_df = load_parquet_tables(paths.extracted_dir)

    if remove_tar_after and tar_path.exists():
        os.remove(tar_path)

    return events_df, impressions_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download/extract tarball and load parquet tables.")
    parser.add_argument("--gdrive-url", type=str, default=None, help="Google Drive share link for the tarball.")
    parser.add_argument("--remove-tar", action="store_true", help="Delete tarball after extraction.")
    args = parser.parse_args()

    events, impressions = ingest(args.gdrive_url, remove_tar_after=args.remove_tar)
    print("Loaded:")
    print("events:", events.shape)
    print("impressions:", impressions.shape)
