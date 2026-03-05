import os

from llm.local_llm import query_llm
from utils import load_inventory, administer_inventory, generate_output_filename
from retrieval import SemanticRetriever

# ==== RETRIEVAL/CONTEXT CONTROL VARIABLES ====
# --------------------------------------------------------------
# >>> TUNE THESE FOR EACH LLM TESTED <<<
MAX_TOTAL_CHARS_CONTEXT = 1800   # per context injection
RETRIEVAL_TOP_N = 6             # How many top relevant chunks to inject each time
RETRIEVER_MAX_CHUNKS = 600      # higher for lots of data
# --------------------------------------------------------------

# note: SemanticRetriever is defined in retrieval.py with sensible defaults
# the class here used to duplicate that logic; we now import and re-use it

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
    # path is case-sensitive; the data directory in the repo is "Data" not "data"
    Hawaiian_Literature_Folder = 'Data/hawaiian_chunks'
    retriever = SemanticRetriever(
        [Hawaiian_Literature_Folder],
        max_chunks=RETRIEVER_MAX_CHUNKS
    )
    # ensure we actually loaded some chunks
    if not retriever.chunks:
        raise RuntimeError(f"no chunks found in {Hawaiian_Literature_Folder}, check path and JSONL files")

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
