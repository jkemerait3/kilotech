import json
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path

from django.db import close_old_connections
from django.db import transaction
from django.utils import timezone

from .models import RAGChunk, RAGSource
from retrieval import embed_texts


_INGEST_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-ingest")


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


def _run_ingest_job(source_id):
    close_old_connections()
    try:
        ingest_source(source_id)
    finally:
        close_old_connections()


def schedule_ingest(source_id):
    RAGSource.objects.filter(pk=source_id).update(
        ingestion_status=RAGSource.IngestionStatus.QUEUED,
        error_message="",
    )
    _INGEST_EXECUTOR.submit(_run_ingest_job, source_id)


def ingest_source(source_id):
    source = RAGSource.objects.get(pk=source_id)

    source.chunks.all().delete()
    source.ingestion_status = RAGSource.IngestionStatus.PROCESSING
    source.error_message = ""
    source.chunk_count = 0
    source.processed_at = None
    source.save(update_fields=["ingestion_status", "error_message", "chunk_count", "processed_at"])

    try:
        file_path = Path(source.uploaded_file.path)
        source.source_size_bytes = file_path.stat().st_size if file_path.exists() else 0
        source.save(update_fields=["source_size_bytes"])
        source_texts, source_type = _read_source_texts(file_path)

        chunk_texts = []
        for text in source_texts:
            chunk_texts.extend(_chunk_text(text))

        if not chunk_texts:
            source.ingestion_status = RAGSource.IngestionStatus.FAILED
            source.error_message = "No valid text content found in uploaded file."
            source.source_type = source_type
            source.processed_at = timezone.now()
            source.save(update_fields=["ingestion_status", "error_message", "source_type", "processed_at"])
            return

        vectors = embed_texts(chunk_texts)

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
            RAGChunk.objects.bulk_create(chunk_rows, batch_size=500)
            source.source_type = source_type
            source.ingestion_status = RAGSource.IngestionStatus.COMPLETED
            source.chunk_count = len(chunk_rows)
            source.processed_at = timezone.now()
            source.error_message = ""
            source.save(update_fields=["source_type", "ingestion_status", "chunk_count", "processed_at", "error_message"])

    except Exception as exc:
        source.ingestion_status = RAGSource.IngestionStatus.FAILED
        source.error_message = str(exc)
        source.processed_at = timezone.now()
        source.save(update_fields=["ingestion_status", "error_message", "processed_at"])
        raise
