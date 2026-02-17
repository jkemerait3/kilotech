import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Choose a small, efficient model (see SBERT docs for alternatives)
EMBED_MODEL = "all-MiniLM-L6-v2"

class SemanticRetriever:
    def __init__(self, folders, max_chunks=300, embed_model=EMBED_MODEL):
        self.folders = folders  # list of folders with JSONL files
        self.model = SentenceTransformer(embed_model)
        self.chunks = []
        self.chunk_sources = []  # List of (file, index)
        self.embeddings = None
        self._index_chunks(max_chunks)

    def _index_chunks(self, max_chunks):
        all_chunks = []
        sources = []
        for folder in self.folders:
            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.jsonl'):
                    path = os.path.join(folder, filename)
                    with open(path, 'r', encoding='utf-8') as f:
                        for ix, line in enumerate(f):
                            obj = json.loads(line)
                            text = obj.get("text") or obj.get("body") or ""
                            text = text.strip()
                            if text:
                                all_chunks.append(text)
                                sources.append((filename, ix))
                                if len(all_chunks) >= max_chunks:
                                    break
                if len(all_chunks) >= max_chunks:
                    break
            if len(all_chunks) >= max_chunks:
                break
        self.chunks = all_chunks
        self.chunk_sources = sources
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)

    def retrieve(self, query, top_n=4, max_total_chars=4000):
        q_emb = self.model.encode([query])[0]
        sims = np.inner(self.embeddings, q_emb)
        top_ids = np.argsort(sims)[::-1][:top_n]
        results = []
        chars = 0
        for idx in top_ids:
            chunk = self.chunks[idx]
            if chars + len(chunk) > max_total_chars:
                break
            results.append(chunk)
            chars += len(chunk)
        return results
