import re
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from src.models import ParsedDocument


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    path = Path(pdf_path)
    doc_id = path.stem

    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()

    full_text = "\n\n".join(pages)
    full_text = _clean_text(full_text)

    logger.info(f"Parsed {path.name}: {len(pages)} pages, {len(full_text):,} chars")

    return ParsedDocument(
        doc_id=doc_id,
        source_file=path.name,
        text=full_text,
        page_count=len(pages),
    )


def _clean_text(text: str) -> str:
    # Scientific PDFs often break words across lines with a hyphen: "extra-\nchromosomal"
    # This joins them back into one word.
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse runs of spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Normalize 3+ newlines down to 2 (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
