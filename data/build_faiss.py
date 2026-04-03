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
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
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
    {
        "floder": "food",
        "name": "Dietary Reference Intakes for Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids (2005).pdf",
        "language": "en",
        "domain": "dietary",
        "year": "2005",
    },
    {
        "floder": "food",
        "name": "Dietary Reference Intakes Applications in Dietary Planning (2003).pdf",
        "language": "en",
        "domain": "dietary",
        "year": "2003",
    },
    {
        "floder": "workout",
        "name": "Open-Textbook-of-Exercise-Physiology-1756071395.pdf",
        "language": "en",
        "domain": "fitness",
        "year": "2024",
    },
    {
        "floder": "workout",
        "name": "Physical_Activity_Guidelines_2nd_edition.pdf",
        "language": "en",
        "domain": "fitness",
        "year": "2018",
    },
]

OUTPUT_DIR = Path("data/processed")
MARKDOWN_DIR = OUTPUT_DIR / "markdown"
FAISS_DIR = OUTPUT_DIR / "faiss_store"
INDEX_CONFIG = {
    "text": {
        "persist_dir": FAISS_DIR / "text",
        "index_path": FAISS_DIR / "text" / "faiss.index",
    },
    "table": {
        "persist_dir": FAISS_DIR / "table",
        "index_path": FAISS_DIR / "table" / "faiss.index",
    },
}

MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
for config in INDEX_CONFIG.values():
    config["persist_dir"].mkdir(parents=True, exist_ok=True)

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
MARKDOWN_NODE_PARSER = MarkdownNodeParser()
TEXT_SPLITTER = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

EMBED_DIM = 1024

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:-]+(\|[\s:-]+)+\|?\s*$")


def build_metadata(file: Dict[str, str], markdown_text: str) -> Dict[str, Any]:
    path = FILE_DIR / file["floder"] / file["name"]

    return {
        "file_name": path.name,
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


def is_heading_line(line: str) -> bool:
    return HEADING_RE.match(line.strip()) is not None


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return stripped.count("|") >= 2 or TABLE_SEPARATOR_RE.match(stripped) is not None


def split_text_block(text: str) -> List[str]:
    return [part.strip() for part in TEXT_SPLITTER.split_text(text) if part.strip()]


def parse_markdown_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[Dict[str, Any]] = []
    lines = [line.rstrip() for line in markdown_text.splitlines()]
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped == "<!-- image -->":
            i += 1
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            heading_stack = [item for item in heading_stack if item["level"] < level]
            heading_stack.append({"level": level, "title": title})
            i += 1
            continue

        if is_table_line(line):
            table_lines = [line]
            i += 1
            while i < len(lines) and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(
                {
                    "type": "table",
                    "text": "\n".join(table_lines),
                    "headings": [item.copy() for item in heading_stack],
                }
            )
            continue

        paragraph_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
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
                "text": "\n".join(paragraph_lines),
                "headings": [item.copy() for item in heading_stack],
            }
        )

    return blocks


def flatten_parser_header_path(header_path: Any) -> str:
    if not isinstance(header_path, str):
        return ""

    parts = [part.strip() for part in header_path.split("/") if part.strip()]
    return " > ".join(parts)


def render_heading_markdown(headings: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{'#' * item['level']} {item['title']}" for item in headings)


def build_section_markdown(headings: List[Dict[str, Any]], body: str) -> str:
    heading_markdown = render_heading_markdown(headings)
    if heading_markdown:
        return f"{heading_markdown}\n\n{body}"
    return body



def build_chunk_metadata(
    base_metadata: Dict[str, Any],
    chunk_type: str,
    header_path: str,
) -> Dict[str, Any]:
    metadata = dict(base_metadata)
    metadata["chunk_type"] = chunk_type
    metadata["header_path"] = header_path
    return metadata


def chunk_markdown_document(
    file: Dict[str, str],
    markdown_text: str,
) -> List[TextNode]:
    base_metadata = build_metadata(file, markdown_text)

    parser_docs: List[Document] = []
    nodes: List[TextNode] = []
    pending_text_blocks: List[str] = []
    pending_headings: List[Dict[str, Any]] = []

    def flush_text_blocks() -> None:
        nonlocal pending_text_blocks, pending_headings
        if not pending_text_blocks:
            return

        merged_text = "\n\n".join(block for block in pending_text_blocks if block)
        if not merged_text:
            pending_text_blocks = []
            return

        parser_docs.append(
            Document(
                text=build_section_markdown(pending_headings, merged_text),
                metadata=dict(base_metadata),
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
            nodes.append(
                TextNode(
                    text=block_text,
                    metadata=build_chunk_metadata(
                        base_metadata,
                        "table",
                        " > ".join(item["title"] for item in headings),
                    ),
                )
            )
            continue

        if pending_text_blocks and pending_headings != headings:
            flush_text_blocks()

        pending_headings = headings
        pending_text_blocks.append(block_text)

    flush_text_blocks()

    parsed_nodes = MARKDOWN_NODE_PARSER.get_nodes_from_documents(parser_docs)
    for parsed_node in parsed_nodes:
        text = parsed_node.text.strip()
        if not text:
            continue

        metadata = build_chunk_metadata(
            base_metadata,
            "text",
            flatten_parser_header_path(parsed_node.metadata.get("header_path")),
        )

        for part in split_text_block(text):
            nodes.append(
                TextNode(
                    text=part,
                    metadata=dict(metadata),
                )
            )

    return nodes


def build_nodes(files: List[Dict[str, str]], force_rebuild_markdown: bool = False) -> List[TextNode]:
    nodes: List[TextNode] = []

    for file in files:
        path = FILE_DIR / file["floder"] / file["name"]
        markdown_text, _ = get_or_create_markdown(file, force_rebuild=force_rebuild_markdown)
        print(f"[INFO] Building structured chunks from: {path}")
        nodes.extend(chunk_markdown_document(file, markdown_text))

    return nodes


def filter_nodes_by_chunk_type(nodes: List[TextNode], chunk_type: str) -> List[TextNode]:
    return [node for node in nodes if node.metadata.get("chunk_type") == chunk_type]


def build_storage_context() -> tuple[faiss.IndexFlatL2, StorageContext]:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return faiss_index, storage_context


def build_and_persist_single_index(nodes: List[TextNode], chunk_type: str) -> None:
    config = INDEX_CONFIG[chunk_type]
    faiss_index, storage_context = build_storage_context()

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=str(config["persist_dir"]))
    faiss.write_index(faiss_index, str(config["index_path"]))

    print(f"[INFO] Saved {chunk_type} llama-index storage to: {config['persist_dir']}")
    print(f"[INFO] Saved {chunk_type} faiss index to: {config['index_path']}")


def build_and_persist_index(force_rebuild_markdown: bool = False) -> None:
    nodes = build_nodes(pdf_file, force_rebuild_markdown=force_rebuild_markdown)
    print(f"[INFO] Nodes: {len(nodes)}")

    indexed_nodes_by_type: Dict[str, List[TextNode]] = {"text": [], "table": []}
    for idx, node in enumerate(nodes):
        metadata = dict(node.metadata)
        metadata["chunk_id"] = idx
        metadata["chunk_len"] = len(node.text)
        chunk_type = metadata.get("chunk_type")
        if chunk_type not in indexed_nodes_by_type:
            continue

        indexed_nodes_by_type[chunk_type].append(
            TextNode(text=node.text, metadata=metadata)
        )

    for chunk_type, chunk_nodes in indexed_nodes_by_type.items():
        print(f"[INFO] Indexed {chunk_type} nodes: {len(chunk_nodes)}")
        if not chunk_nodes:
            print(f"[WARN] No {chunk_type} nodes found. Skipping index build.")
            continue
        build_and_persist_single_index(chunk_nodes, chunk_type)

    print(f"[INFO] Saved markdown files to: {MARKDOWN_DIR}")
    print(f"[INFO] Saved split indexes under: {FAISS_DIR}")


def load_index_from_disk(chunk_type: str = "text") -> VectorStoreIndex:
    config = INDEX_CONFIG[chunk_type]
    loaded_faiss_index = faiss.read_index(str(config["index_path"]))
    loaded_vector_store = FaissVectorStore(faiss_index=loaded_faiss_index)

    loaded_storage_context = StorageContext.from_defaults(
        persist_dir=str(config["persist_dir"]),
        vector_store=loaded_vector_store,
    )

    loaded_index = load_index_from_storage(loaded_storage_context)
    return loaded_index


def test_similarity_from_disk(query: str, top_k: int = 3, chunk_type: str = "text") -> None:
    loaded_index = load_index_from_disk(chunk_type=chunk_type)
    retriever = loaded_index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    print(f"\n[TEST RETRIEVE RESULT - LOADED FROM DISK - {chunk_type.upper()}]")
    for i, node in enumerate(nodes, 1):
        print(f"\n{'-' * 50} Result {i} {'-' * 50}")
        print(node.text)
        pprint(node.metadata)


def main() -> None:
    force_rebuild_markdown = True

    text_index_path = INDEX_CONFIG["text"]["index_path"]
    table_index_path = INDEX_CONFIG["table"]["index_path"]

    if text_index_path.exists() and table_index_path.exists() and not force_rebuild_markdown:
        print(f"[INFO] Reusing existing indexes: {FAISS_DIR}")
    else:
        print("[INFO] Building / rebuilding split indexes from markdown/PDF sources.")
        build_and_persist_index(force_rebuild_markdown=force_rebuild_markdown)

    test_similarity_from_disk(
        "How much protein should I take daily?",
        top_k=2,
        chunk_type="table",
    )


if __name__ == "__main__":
    main()
