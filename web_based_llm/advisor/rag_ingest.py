import json
import logging
import importlib
from pathlib import Path

from django.db import transaction
from django.utils import timezone

from .models import RAGChunk, RAGSource
from retrieval import embed_texts

logger = logging.getLogger('advisor.rag_ingest')


def _chunk_text(text, max_chars=1000, overlap=120):
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return chunks


def _read_source_texts(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        texts = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or obj.get("body") or obj.get("content") or "").strip()
                if text:
                    texts.append(text)
        return texts, "jsonl"

    if suffix == ".pdf":
        pypdf = importlib.util.find_spec("pypdf")
        if pypdf is None:
            raise RuntimeError("PDF support requires pypdf. Install it and retry ingestion.")

        pypdf_module = importlib.import_module("pypdf")
        pdf_reader = pypdf_module.PdfReader

        reader = pdf_reader(str(file_path))
        pages = []
        for page in reader.pages:
            pages.append((page.extract_text() or "").strip())
        texts = [page for page in pages if page]
        return texts, "pdf"

    text = file_path.read_text(encoding="utf-8")
    return [text], suffix.replace(".", "") or "unknown"


def schedule_ingest(source_id):
    """Ingest synchronously and log all steps."""
    logger.info(f"Starting ingestion for RAGSource id={source_id}")
    try:
        ingest_source(source_id)
        logger.info(f"Completed ingestion for RAGSource id={source_id}")
    except Exception as exc:
        logger.exception(f"Failed to ingest RAGSource id={source_id}: {exc}")
        raise


def ingest_source(source_id):
    source = RAGSource.objects.get(pk=source_id)
    logger.info(f"Ingesting: {source.title} (id={source.id}, type={source.source_type})")

    source.chunks.all().delete()
    source.ingestion_status = RAGSource.IngestionStatus.PROCESSING
    source.error_message = ""
    source.chunk_count = 0
    source.processed_at = None
    source.save(update_fields=["ingestion_status", "error_message", "chunk_count", "processed_at"])

    try:
        file_path = Path(source.uploaded_file.path)
        logger.debug(f"File path: {file_path}")
        source.source_size_bytes = file_path.stat().st_size if file_path.exists() else 0
        source.save(update_fields=["source_size_bytes"])
        logger.debug(f"Reading source texts from {file_path.suffix} file")
        source_texts, source_type = _read_source_texts(file_path)
        logger.debug(f"Read {len(source_texts)} text segment(s) from source")

        chunk_texts = []
        for text in source_texts:
            chunk_texts.extend(_chunk_text(text))
        logger.info(f"Created {len(chunk_texts)} chunks from source")

        if not chunk_texts:
            logger.warning(f"No valid text content found in uploaded file for source id={source_id}")
            source.ingestion_status = RAGSource.IngestionStatus.FAILED
            source.error_message = "No valid text content found in uploaded file."
            source.source_type = source_type
            source.processed_at = timezone.now()
            source.save(update_fields=["ingestion_status", "error_message", "source_type", "processed_at"])
            return

        logger.debug(f"Embedding {len(chunk_texts)} chunks...")
        vectors = embed_texts(chunk_texts)
        logger.debug(f"Generated embeddings with shape {vectors.shape}")

        chunk_rows = []
        for idx, (chunk_text, vector) in enumerate(zip(chunk_texts, vectors)):
            chunk_rows.append(
                RAGChunk(
                    source=source,
                    chunk_index=idx,
                    content=chunk_text,
                    embedding=vector.astype("float32").tobytes(),
                    embedding_dim=int(vector.shape[0]),
                    chunk_size_bytes=len(chunk_text.encode("utf-8")),
                )
            )

        with transaction.atomic():
            logger.debug(f"Bulk creating {len(chunk_rows)} RAGChunk records")
            RAGChunk.objects.bulk_create(chunk_rows, batch_size=500)
            source.source_type = source_type
            source.ingestion_status = RAGSource.IngestionStatus.COMPLETED
            source.chunk_count = len(chunk_rows)
            source.processed_at = timezone.now()
            source.error_message = ""
            source.save(update_fields=["source_type", "ingestion_status", "chunk_count", "processed_at", "error_message"])
        logger.info(f"Successfully completed ingestion: {source.chunk_count} chunks created")

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Ingestion failed for source id={source_id}: {error_msg}")
        source.ingestion_status = RAGSource.IngestionStatus.FAILED
        source.error_message = error_msg
        source.processed_at = timezone.now()
        source.save(update_fields=["ingestion_status", "error_message", "processed_at"])
        raise


def _warmup_embedding_model():
    """Pre-download and cache the embedding model to avoid timeout on first ingest."""
    try:
        logger.info("Pre-warming embedding model...")
        from retrieval import _get_model
        _get_model()  # Lazy-loads and caches the model
        logger.info("Embedding model pre-warmed successfully")
    except Exception as exc:
        logger.warning(f"Failed to pre-warm embedding model: {exc}. First ingestion may be slow.")
