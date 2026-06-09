import os
import requests
from pathlib import Path
from src.config import DATA_DIR, RESEARCH_PAPERS

def download_papers(force=False):
    """
    Downloads the default research papers from arXiv if they are not already cached.
    """
    print("Initializing research papers download...")
    success_count = 0
    for title, info in RESEARCH_PAPERS.items():
        dest_path = DATA_DIR / info["filename"]
        if dest_path.exists() and not force:
            print(f"Already cached: {info['display_name']} -> {dest_path.name}")
            success_count += 1
            continue
            
        url = info["url"]
        print(f"Downloading {info['display_name']} from {url}...")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Write to a temp file first, then rename to avoid incomplete files
            temp_path = dest_path.with_suffix(".tmp")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            temp_path.rename(dest_path)
            print(f"Successfully downloaded to {dest_path.name}")
            success_count += 1
        except Exception as e:
            print(f"Failed to download {info['display_name']}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            
    return success_count == len(RESEARCH_PAPERS)

if __name__ == "__main__":
    download_papers()
