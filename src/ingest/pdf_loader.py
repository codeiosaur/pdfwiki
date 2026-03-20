# src/ingest/pdf_loader.py
from dataclasses import dataclass
from typing import List
from pathlib import Path
import uuid

import pypdf  # pip install pypdf


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chapter: str | None


def load_pdf(path: str, chunk_size: int = 1000) -> List[Chunk]:
    """
    Load a PDF and split it into text chunks (~chunk_size words each).
    Return a list of Chunk objects with unique IDs.
    """

    # 1. Read PDF
    pdf = pypdf.PdfReader(path)
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

    # 2. Split into words
    words = full_text.split()
    chunks = []

    # 3. Create chunks
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                source=str(Path(path).name),
                chapter=None,
            )
        )

    return chunks