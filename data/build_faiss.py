from pathlib import Path
import re
from typing import List, Dict, Any

import faiss

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)

from hierarchical.postprocessor import ResultPostprocessor

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from marshmallow import pprint

FILE_DIR = Path("data/raw/pdf")

pdf_file: List[Dict[str, str]] = [
    {
        "floder": "food",
        "name": "dietary_guidelines_for_americans-2020-2025.pdf",
        "language": "en",
        "domain": "dietary",
        "year": "2020",
    },
    # {
    #     "floder": "food",
    #     "name": "Dietary Reference Intakes for Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids (2005).pdf",
    #     "language": "en",
    #     "domain": "dietary",
    #     "year": "2005",
    # },
    # {
    #     "floder": "food",
    #     "name": "Dietary Reference Intakes for Sodium and Potassium (2019).pdf",
    #     "language": "en",
    #     "domain": "dietary",
    #     "year": "2019",
    # },
    # {
    #     "floder": "workout",
    #     "name": "Open-Textbook-of-Exercise-Physiology-1756071395.pdf",
    #     "language": "en",
    #     "domain": "fitness",
    #     "year": "2024",
    # },
    # {
    #     "floder": "workout",
    #     "name": "Physical_Activity_Guidelines_2nd_edition.pdf",
    #     "language": "en",
    #     "domain": "fitness",
    #     "year": "2018",
    # },
]

OUTPUT_DIR = Path("data/processed")
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
FAISS_DIR = OUTPUT_DIR / "faiss_store"
FAISS_INDEX_PATH = FAISS_DIR / "faiss.index"

MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA
)

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    },
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cuda",
)

Settings.embed_model = embed_model

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBED_DIM = 1024
faiss_index = faiss.IndexFlatL2(EMBED_DIM)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:-]+(\|[\s:-]+)+\|?\s*$")


def build_metadata(file: Dict[str, str], markdown_text: str) -> Dict[str, Any]:
    path = FILE_DIR / file["floder"] / file["name"]

    return {
        "source_path": str(path),
        "file_name": path.name,
        "title": path.stem,
        "export_format": "markdown",
        "language": file.get("language"),
        "domain": file.get("domain"),
        "year": file.get("year"),
        "source_type": "official_or_reference",
        "text_length": len(markdown_text),
        "heading_processor": "docling-hierarchical-pdf",
    }


def convert_pdf_to_markdown(file: Dict[str, str]) -> str:
    path = FILE_DIR / file["floder"] / file["name"]

    result = converter.convert(path)
    ResultPostprocessor(result, source=path).process()

    return result.document.export_to_markdown()


def save_markdown(file: Dict[str, str], markdown_text: str) -> Path:
    path = FILE_DIR / file["floder"] / file["name"]
    out_path = MARKDOWN_DIR / f"{path.stem}.md"
    out_path.write_text(markdown_text, encoding="utf-8")
    return out_path


def get_markdown_path(file: Dict[str, str]) -> Path:
    path = FILE_DIR / file["floder"] / file["name"]
    return MARKDOWN_DIR / f"{path.stem}.md"


def get_or_create_markdown(file: Dict[str, str], force_rebuild: bool = False) -> tuple[str, Path]:
    md_path = get_markdown_path(file)

    if md_path.exists() and not force_rebuild:
        print(f"[INFO] Reusing markdown: {md_path}")
        return md_path.read_text(encoding="utf-8"), md_path

    path = FILE_DIR / file["floder"] / file["name"]
    print(f"[INFO] Building markdown with hierarchical heading recovery: {path}")
    markdown_text = convert_pdf_to_markdown(file)
    save_markdown(file, markdown_text)
    return markdown_text, md_path


def normalize_markdown_line(line: str) -> str:
    return line.rstrip()


def is_heading_line(line: str) -> bool:
    return HEADING_RE.match(line.strip()) is not None


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return stripped.count("|") >= 2 or TABLE_SEPARATOR_RE.match(stripped) is not None


def clean_block_lines(lines: List[str]) -> str:
    return "\n".join(normalize_markdown_line(line) for line in lines).strip()


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    parts: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            split_at = text.rfind("\n\n", start, end)
            if split_at <= start:
                split_at = text.rfind(". ", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at + (0 if text[split_at:split_at + 2] == "\n\n" else 1)

        piece = text[start:end].strip()
        if piece:
            parts.append(piece)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return parts


def parse_markdown_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[Dict[str, Any]] = []
    lines = markdown_text.splitlines()
    i = 0

    while i < len(lines):
        raw_line = normalize_markdown_line(lines[i])
        stripped = raw_line.strip()

        if not stripped or stripped == "<!-- image -->":
            i += 1
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = [item for item in heading_stack if item["level"] < level]
            heading_stack.append({"level": level, "title": title})
            i += 1
            continue

        if is_table_line(raw_line):
            table_lines = [raw_line]
            i += 1
            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(normalize_markdown_line(lines[i]))
                i += 1
            blocks.append(
                {
                    "type": "table",
                    "text": clean_block_lines(table_lines),
                    "headings": [item.copy() for item in heading_stack],
                }
            )
            continue

        paragraph_lines = [raw_line]
        i += 1
        while i < len(lines):
            next_line = normalize_markdown_line(lines[i])
            next_stripped = next_line.strip()
            if not next_stripped or next_stripped == "<!-- image -->":
                i += 1
                break
            if is_heading_line(next_line) or is_table_line(next_line):
                break
            paragraph_lines.append(next_line)
            i += 1

        blocks.append(
            {
                "type": "text",
                "text": clean_block_lines(paragraph_lines),
                "headings": [item.copy() for item in heading_stack],
            }
        )

    return blocks


def merge_heading_path(headings: List[Dict[str, Any]]) -> str:
    return " > ".join(item["title"] for item in headings)


def build_chunk_metadata(
    base_metadata: Dict[str, Any],
    headings: List[Dict[str, Any]],
    chunk_type: str,
) -> Dict[str, Any]:
    metadata = dict(base_metadata)
    section_path = merge_heading_path(headings)
    metadata["chunk_type"] = chunk_type
    metadata["section_path"] = section_path
    metadata["section_title"] = headings[-1]["title"] if headings else base_metadata["title"]
    metadata["heading_level"] = headings[-1]["level"] if headings else 0
    metadata["heading_path"] = [item["title"] for item in headings]
    return metadata


def build_chunk_text(headings: List[Dict[str, Any]], body: str) -> str:
    section_path = merge_heading_path(headings)
    if section_path:
        return f"{section_path}\n\n{body}"
    return body


def chunk_markdown_document(
    file: Dict[str, str],
    markdown_text: str,
    md_path: Path,
) -> List[Document]:
    base_metadata = build_metadata(file, markdown_text)
    base_metadata["markdown_path"] = str(md_path)

    docs: List[Document] = []
    pending_text_blocks: List[str] = []
    pending_headings: List[Dict[str, Any]] = []

    def flush_text_blocks() -> None:
        nonlocal pending_text_blocks, pending_headings
        if not pending_text_blocks:
            return

        merged_text = "\n\n".join(block for block in pending_text_blocks if block).strip()
        if not merged_text:
            pending_text_blocks = []
            return

        parts = split_text_with_overlap(merged_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for part in parts:
            docs.append(
                Document(
                    text=build_chunk_text(pending_headings, part),
                    metadata=build_chunk_metadata(base_metadata, pending_headings, "text"),
                )
            )

        pending_text_blocks = []

    for block in parse_markdown_blocks(markdown_text):
        headings = block["headings"]
        block_text = block["text"]
        block_type = block["type"]

        if not block_text:
            continue

        if block_type == "table":
            flush_text_blocks()
            docs.append(
                Document(
                    text=build_chunk_text(headings, block_text),
                    metadata=build_chunk_metadata(base_metadata, headings, "table"),
                )
            )
            continue

        if pending_text_blocks and pending_headings != headings:
            flush_text_blocks()

        pending_headings = headings
        candidate_blocks = pending_text_blocks + [block_text]
        candidate_text = "\n\n".join(candidate_blocks)

        if pending_text_blocks and len(candidate_text) > CHUNK_SIZE:
            flush_text_blocks()
            pending_headings = headings

        pending_text_blocks.append(block_text)

    flush_text_blocks()
    return docs


def build_documents(files: List[Dict[str, str]], force_rebuild_markdown: bool = False) -> List[Document]:
    docs: List[Document] = []

    for file in files:
        path = FILE_DIR / file["floder"] / file["name"]
        markdown_text, md_path = get_or_create_markdown(file, force_rebuild=force_rebuild_markdown)
        print(f"[INFO] Building structured chunks from: {path}")
        docs.extend(chunk_markdown_document(file, markdown_text, md_path))

    return docs


def build_and_persist_index(force_rebuild_markdown: bool = False) -> None:
    documents = build_documents(pdf_file, force_rebuild_markdown=force_rebuild_markdown)
    print(f"[INFO] Documents: {len(documents)}")

    nodes: List[TextNode] = []
    for idx, doc in enumerate(documents):
        metadata = dict(doc.metadata)
        metadata["chunk_id"] = idx
        metadata["chunk_char_count"] = len(doc.text)
        nodes.append(
            TextNode(
                text=doc.text,
                metadata=metadata,
            )
        )

    print(f"[INFO] Nodes: {len(nodes)}")

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=str(FAISS_DIR))
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

    print(f"[INFO] Saved markdown files to: {MARKDOWN_DIR}")
    print(f"[INFO] Saved llama-index storage to: {FAISS_DIR}")
    print(f"[INFO] Saved faiss index to: {FAISS_INDEX_PATH}")


def load_index_from_disk() -> VectorStoreIndex:
    loaded_faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    loaded_vector_store = FaissVectorStore(faiss_index=loaded_faiss_index)

    loaded_storage_context = StorageContext.from_defaults(
        persist_dir=str(FAISS_DIR),
        vector_store=loaded_vector_store,
    )

    loaded_index = load_index_from_storage(loaded_storage_context)
    return loaded_index


def test_similarity_from_disk(query: str, top_k: int = 3) -> None:
    loaded_index = load_index_from_disk()
    retriever = loaded_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    print("\n[TEST RETRIEVE RESULT - LOADED FROM DISK]")
    for i, node in enumerate(nodes, 1):
        print(f"\n{'-' * 50} Result {i} {'-' * 50}")
        print(node.text[:500])
        pprint(node.metadata)


def main() -> None:
    force_rebuild_markdown = False

    if FAISS_INDEX_PATH.exists() and not force_rebuild_markdown:
        print(f"[INFO] Reusing existing index: {FAISS_INDEX_PATH}")
    else:
        print("[INFO] Building / rebuilding index from markdown/PDF sources.")
        build_and_persist_index(force_rebuild_markdown=force_rebuild_markdown)

    test_similarity_from_disk(
        "How much protein should I take daily?",
        top_k=3,
    )


if __name__ == "__main__":
    main()