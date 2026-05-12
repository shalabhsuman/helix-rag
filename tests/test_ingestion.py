from pathlib import Path

import fitz
import pytest

from src.ingestion.parser import _clean_text, parse_pdf
from src.models import ParsedDocument


def test_clean_text_fixes_hyphenated_line_breaks():
    assert "extrachromosomal" in _clean_text("extra-\nchromosomal")


def test_clean_text_collapses_multiple_newlines():
    result = _clean_text("paragraph one\n\n\n\nparagraph two")
    assert "\n\n\n" not in result


def test_clean_text_collapses_extra_spaces():
    result = _clean_text("word   another   word")
    assert "word another word" in result


def test_parse_pdf_returns_parsed_document(tmp_path: Path):
    pdf_path = tmp_path / "test_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a test paper about extrachromosomal DNA research.")
    doc.save(str(pdf_path))
    doc.close()

    result = parse_pdf(pdf_path)

    assert isinstance(result, ParsedDocument)
    assert result.doc_id == "test_paper"
    assert result.source_file == "test_paper.pdf"
    assert result.page_count == 1
    assert "extrachromosomal" in result.text


def test_parse_pdf_doc_id_strips_extension(tmp_path: Path):
    pdf_path = tmp_path / "kim_2020_ecdna.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Some content.")
    doc.save(str(pdf_path))
    doc.close()

    result = parse_pdf(pdf_path)
    assert result.doc_id == "kim_2020_ecdna"
