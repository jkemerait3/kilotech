import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

try:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model(model_name=EMBED_MODEL):
    return SentenceTransformer(model_name)


def embed_texts(texts, model_name=EMBED_MODEL):
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model(model_name)
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


def _decode_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32)


class SemanticRetriever:
    def __init__(self, embed_model=EMBED_MODEL):
        self.embed_model = embed_model

    @staticmethod
    def _build_citation(first_author, has_multiple_authors, title):
        author = (first_author or "").strip()
        if author:
            if has_multiple_authors:
                return f"{author} et. al"
            return author
        return (title or "Unknown Author").strip()

    def _load_chunks(self):
        from advisor.models import RAGChunk

        rows = list(
            RAGChunk.objects.filter(
                source__is_active=True,
                source__ingestion_status="completed",
            )
            .select_related("source")
            .values(
                "id",
                "content",
                "embedding",
                "chunk_index",
                "source__title",
                "source__first_author",
                "source__has_multiple_authors",
            )
        )
        if not rows:
            return [], np.empty((0, 384), dtype=np.float32)

        chunks = []
        vectors = []
        for row in rows:
            chunk_text = row["content"]
            vector = _decode_embedding(row["embedding"])
            if chunk_text and vector.size > 0:
                source_title = row["source__title"]
                citation = self._build_citation(
                    first_author=row["source__first_author"],
                    has_multiple_authors=row["source__has_multiple_authors"],
                    title=source_title,
                )
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source_title,
                        "citation": citation,
                        "chunk_index": row["chunk_index"],
                    }
                )
                vectors.append(vector)

        if not vectors:
            return [], np.empty((0, 384), dtype=np.float32)

        return chunks, np.vstack(vectors)

    def retrieve(self, query, top_n=4, max_total_chars=4000, return_with_sources=False):
        chunks, embeddings = self._load_chunks()
        if not chunks or embeddings.size == 0:
            return []

        q_emb = embed_texts([query], model_name=self.embed_model)[0]
        sims = np.inner(embeddings, q_emb)
        top_k_indices = np.argsort(sims)[::-1][:top_n]

        results = []
        chars = 0
        for idx in top_k_indices:
            chunk_data = chunks[idx]
            chunk_text = chunk_data["text"]
            if chars + len(chunk_text) > max_total_chars:
                break
            if return_with_sources:
                results.append(chunk_data)
            else:
                results.append(chunk_text)
            chars += len(chunk_text)

        return results