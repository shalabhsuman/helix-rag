from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.models import ChildChunk, ParentChunk, ParsedDocument

# ~1200 tokens at 4 chars/token. This is what the LLM receives when answering.
PARENT_CHUNK_SIZE = 4800
PARENT_CHUNK_OVERLAP = 400

# ~300 tokens at 4 chars/token. This is what gets searched in Qdrant.
CHILD_CHUNK_SIZE = 1200
CHILD_CHUNK_OVERLAP = 100


@dataclass
class ChunkSet:
    parents: list[ParentChunk]
    children: list[ChildChunk]


def split_document(doc: ParsedDocument) -> ChunkSet:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    parent_texts = parent_splitter.split_text(doc.text)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    child_index = 0

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = f"{doc.doc_id}_parent_{p_idx}"
        parents.append(
            ParentChunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                source_file=doc.source_file,
                text=p_text,
                chunk_index=p_idx,
            )
        )

        for c_text in child_splitter.split_text(p_text):
            children.append(
                ChildChunk(
                    chunk_id=f"{doc.doc_id}_child_{child_index}",
                    parent_chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    source_file=doc.source_file,
                    text=c_text,
                    chunk_index=child_index,
                )
            )
            child_index += 1

    logger.info(f"{doc.doc_id}: {len(parents)} parents -> {len(children)} children")
    return ChunkSet(parents=parents, children=children)
