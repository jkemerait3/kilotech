import os
import csv
import json
import ast
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

def get_patient_info():
    print("Hello, and my name is Olive. I am here to help you find the care you need.")
    first_name = input("Please enter your first name: ")
    last_name = input("Please enter your last name: ")
    dob = input("Please enter your date of birth (MM/DD/YYYY): ")
    return first_name, last_name, dob

def get_self_report():
    print("\nPlease describe what brings you in today (your symptoms and concerns):")
    return input("> ")

def generate_csv_output(patient_info, self_report, assessment_summary, results, filename="output/results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        metadata_row = [
            f"{patient_info[0]} {patient_info[1]}",
            patient_info[2],
            self_report,
            assessment_summary
        ]
        writer.writerow(metadata_row)
        max_qs = max(len(r["question_scores"]) for r in results)
        header_row = ["assessment name", "total score"] + [f"q{i+1} score" for i in range(max_qs)]
        writer.writerow(header_row)
        for r in results:
            row = [r["name"], r["total_score"]] + r["question_scores"]
            row += [""] * (len(header_row) - len(row))
            writer.writerow(row)
    print(f"\n✅ Results saved to {filename}")

# ==== MAIN INTERACTIVE FLOW ====
def run_cli():
    # ---- Set up retriever at session start ----
    dsm5_folder = "Data/dsm5_chunks/"
    dataset_folder = "Data/dataset_chunks/"
    retriever = SemanticRetriever(
        [dsm5_folder, dataset_folder],
        max_chunks=RETRIEVER_MAX_CHUNKS
    )
    inventories_folder = "inventories"
    available_files = list_json_files(inventories_folder, exclude={"PHQ-4.json"})

    # ---- intake ----
    patient_info = get_patient_info()
    self_report = get_self_report()

    # ---- PHQ-4 always first ----
    print("\nThank you for sharing this with me. I will now administer PHQ-4, a brief assessment, to better understand your experience.")
    phq4_inventory = load_inventory("inventories/PHQ-4.json")
    phq4_result = administer_inventory(phq4_inventory)

    # ---- RETRIEVE RELEVANT CONTEXT ----
    context_query = (
        "Patient self-report: " + self_report + "\n"
        + "PHQ-4 total score: " + str(phq4_result.get('total_score', 'N/A')) + "\n"
        + "PHQ-4 question scores: " + str(phq4_result.get('question_scores', []))
    )
    retrieved_context = retriever.retrieve(
        context_query,
        top_n=RETRIEVAL_TOP_N,
        max_total_chars=MAX_TOTAL_CHARS_CONTEXT
    )
    context_text = "\n\n".join(retrieved_context)

    # ---- LLM selects inventories ----
    prompt = (
        "You are an expert mental health chatbot triage assistant. "
        "Given the DSM-5 and dataset context below, the patient self-report, and PHQ-4 scores, "
        "select from the following list of inventory filenames ALL those relevant to administer next "
        "in addition to PHQ-4. Respond ONLY with a valid Python list of filenames; no commentary.\n\n"
        "--- Clinical Context ---\n"
        f"{context_text}\n\n"
        "--- Inventories (JSON filenames) ---\n"
        f"{available_files}\n\n"
        "--- Patient self-report ---\n"
        f"{self_report}\n\n"
        f"PHQ-4 total: {phq4_result.get('total_score', 'N/A')}, Question scores: {phq4_result.get('question_scores', [])}\n"
        "Reply ONLY with a single valid Python list containing only valid filenames with ABSOLUTELY NO additional commentary. For example, ['file1.json', 'file2.json']"
    )

    inventories_response = query_llm(prompt).strip()
    # print(f"\nRaw LLM inventory selection response: {inventories_response}")

    try:
        chosen_inventories = ast.literal_eval(inventories_response)
        if not isinstance(chosen_inventories, list):
            raise ValueError("Parsed result is not a list")
    except Exception:
        print("⚠️ Failed to parse LLM response as Python list. No additional inventories will be administered.")
        chosen_inventories = []

    # Only administer files that exist in your folder
    chosen_inventories = [f for f in chosen_inventories if f in available_files]

    if chosen_inventories:
        print("\nThank you for your responses. The following additional questions will help me understand your situation further.")
        for inv in chosen_inventories:
            print(f"- {inv}")
    else:
        print("\nNo additional inventories will be administered based on the current information.")

    results = [phq4_result]
    for filename in chosen_inventories:
        try:
            # print(f"\nNow administering: {filename}")
            inventory = load_inventory(os.path.join(inventories_folder, filename))
            res = administer_inventory(inventory)
            results.append(res)
        except Exception as e:
            print(f"❌ Failed to administer {filename}: {e}")

    # ---- Retrieve context for summary as well ----
    summary_context = retriever.retrieve(
        self_report,
        top_n=RETRIEVAL_TOP_N,
        max_total_chars=MAX_TOTAL_CHARS_CONTEXT
    )
    summary_context_text = "\n\n".join(summary_context)

    summary_prompt = (
        "You are an expert mental health chatbot assisting clinicians. "
        "Based on the clinical context, the patient's self report, "
        "and the administered inventory scores below, write a concise diagnostic impression (2-4 sentences). "
        "Avoid naming specific inventories or stating raw scores.\n\n"
        "--- Clinical Context ---\n"
        f"{summary_context_text}\n\n"
        "--- Self-report ---\n"
        f"{self_report}\n\n"
        "--- Inventory scores summary ---\n"
        f"{[{r['name']: r['total_score']} for r in results]}"
    )
    diagnostic_impression = query_llm(summary_prompt)
    print("Thank you for speaking with me and completing the assessments. Your provider will share the results with you directly.")
    # print(diagnostic_impression)

    filename = generate_output_filename(patient_info[0], patient_info[1], patient_info[2])
    generate_csv_output(patient_info, self_report, diagnostic_impression, results, filename)

if __name__ == "__main__":
    run_cli()
