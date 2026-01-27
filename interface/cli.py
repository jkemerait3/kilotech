import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from llm.local_llm import query_llm
from utils import load_inventory, administer_inventory, generate_output_filename

# ==== RETRIEVAL/CONTEXT CONTROL VARIABLES ====
# --------------------------------------------------------------
# >>> TUNE THESE FOR EACH LLM TESTED <<<
MAX_TOTAL_CHARS_CONTEXT = 1800   # per context injection
RETRIEVAL_TOP_N = 6             # How many top relevant chunks to inject each time
RETRIEVER_MODEL = "all-MiniLM-L6-v2"   # can upgrade for larger LLMs
RETRIEVER_MAX_CHUNKS = 600      # higher for lots of data
# --------------------------------------------------------------


class SemanticRetriever:
    def __init__(self, folders, max_chunks=600, embed_model=RETRIEVER_MODEL):
        self.folders = folders
        self.model = SentenceTransformer(embed_model)
        self.chunks = []
        self.chunk_sources = []
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
                            try:
                                obj = json.loads(line)
                                text = obj.get("text", "")
                                if text and isinstance(text, str):
                                    all_chunks.append(text.strip())
                                    sources.append((filename, ix))
                                if len(all_chunks) >= max_chunks:
                                    break
                            except Exception:
                                continue
                if len(all_chunks) >= max_chunks:
                    break
            if len(all_chunks) >= max_chunks:
                break
        self.chunks = all_chunks
        self.chunk_sources = sources
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)

    def retrieve(self, query, top_n=RETRIEVAL_TOP_N, max_total_chars=MAX_TOTAL_CHARS_CONTEXT):
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

# ==== FILE UTILS ====
def list_json_files(folder_path, exclude=None):
    if not os.path.exists(folder_path):
        return []
    docs = sorted(f for f in os.listdir(folder_path) if f.endswith('.json'))
    if exclude:
        docs = [f for f in docs if f not in exclude]
    return docs

def get_user_query():
    query = input("Welcome to KiloTech. Please enter your question: ")
    return query

# ==== MAIN INTERACTIVE FLOW ====
def run_cli():
    # ---- Set up retriever at session start ----
    dsm5_folder = "Data/dsm5_chunks/"
    dataset_folder = "Data/dataset_chunks/"
    retriever = SemanticRetriever(
        [dsm5_folder, dataset_folder],
        max_chunks=RETRIEVER_MAX_CHUNKS
    )

    # ---- original user query ----
    user_query = get_user_query()

    # ---- Retrieve context for summary ----
    summary_context = retriever.retrieve(
        user_query,
        top_n=RETRIEVAL_TOP_N,
        max_total_chars=MAX_TOTAL_CHARS_CONTEXT
    )
    summary_context_text = "\n\n".join(summary_context)

    summary_prompt = (
        "You are an expert in Hawaiian culture and agriculture assisting farmers. "
        "Based on the literature context and the user's query, "
        "write a culturally informed, actionable response to their query. "
        "Avoid naming specific sources.\n\n"
        "--- Literature Context ---\n"
        f"{summary_context_text}\n\n"
        "--- User Query ---\n"
        f"{user_query}\n\n"
    )
    answer = query_llm(summary_prompt)
    print(answer)
    
if __name__ == "__main__":
    run_cli()
