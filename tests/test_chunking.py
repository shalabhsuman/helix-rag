from src.chunking.splitter import ChunkSet, split_document
from src.models import ParsedDocument


def make_doc(text: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id="test_paper",
        source_file="test_paper.pdf",
        text=text,
        page_count=1,
    )


LONG_TEXT = "extrachromosomal DNA amplification oncogene tumor " * 300  # ~15000 chars


def test_split_returns_chunk_set():
    result = split_document(make_doc(LONG_TEXT))
    assert isinstance(result, ChunkSet)


def test_produces_parents_and_children():
    result = split_document(make_doc(LONG_TEXT))
    assert len(result.parents) > 0
    assert len(result.children) > 0


def test_more_children_than_parents():
    result = split_document(make_doc(LONG_TEXT))
    assert len(result.children) > len(result.parents)


def test_every_child_references_a_valid_parent():
    result = split_document(make_doc(LONG_TEXT))
    parent_ids = {p.chunk_id for p in result.parents}
    for child in result.children:
        assert child.parent_chunk_id in parent_ids


def test_children_are_shorter_than_parents():
    result = split_document(make_doc(LONG_TEXT))
    for child in result.children:
        parent = next(p for p in result.parents if p.chunk_id == child.parent_chunk_id)
        assert len(child.text) <= len(parent.text)


def test_doc_id_propagates_to_all_chunks():
    result = split_document(make_doc(LONG_TEXT))
    for parent in result.parents:
        assert parent.doc_id == "test_paper"
    for child in result.children:
        assert child.doc_id == "test_paper"


def test_chunk_ids_are_unique():
    result = split_document(make_doc(LONG_TEXT))
    all_ids = [p.chunk_id for p in result.parents] + [c.chunk_id for c in result.children]
    assert len(all_ids) == len(set(all_ids))
