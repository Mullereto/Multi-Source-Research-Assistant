import os
import json
from typing import List

def save_chunks(chunks: List[str], output_path: str):
    """
    Save chunks to a JSON file for later use.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to {output_path}")

