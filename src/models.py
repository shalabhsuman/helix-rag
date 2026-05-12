from pydantic import BaseModel


class ParsedDocument(BaseModel):
    doc_id: str        # filename without extension, e.g. kim_2020_ecdna_oncogene_amplification
    source_file: str   # original filename with extension
    text: str          # full extracted and cleaned text
    page_count: int


class ParentChunk(BaseModel):
    chunk_id: str      # e.g. kim_2020_parent_0
    doc_id: str
    source_file: str
    text: str
    chunk_index: int


class ChildChunk(BaseModel):
    chunk_id: str          # e.g. kim_2020_child_3
    parent_chunk_id: str   # points back to the parent this came from
    doc_id: str
    source_file: str
    text: str
    chunk_index: int       # global index across all children in the document
