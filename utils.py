import json
import datetime
from pathlib import Path
from llm.local_llm import MODEL_NAME

def load_inventory(filepath):
    """Load inventory JSON from file path."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def administer_inventory(inventory):
    """Run an inventory via CLI and record scores."""
    print(f"\n{inventory['title']}")
    print(inventory.get("instructions", ""))

    question_scores = []
    for q in inventory["questions"]:
        print(f"\n{q['id']}. {q['text']}")
        for i, choice in enumerate(q["options"]):  # ✅ FIXED: 'options' instead of 'choices'
            print(f"  {i}: {choice['label']}")
        while True:
            try:
                score = int(input("Your answer (number): "))
                if 0 <= score < len(q["options"]):
                    question_scores.append(q["options"][score]["value"])  # ✅ Use mapped value
                    break
                else:
                    print("Invalid input. Try again.")
            except ValueError:
                print("Invalid input. Enter a number.")

    total_score = sum(question_scores)
    return {
        "name": inventory["title"],
        "total_score": total_score,
        "question_scores": question_scores
    }

def generate_output_filename(first_name, last_name, date_str):
    """
    Convert patient name and today's date to formatted output filename like:
    John Smith + 07/04/2025 => JS_2025_07_04.csv
    """
    initials = first_name.strip()[0].upper() + last_name.strip()[0].upper()
    try:
        dob = datetime.datetime.strptime(date_str, "%m/%d/%Y")
        date_fmt = dob.strftime("%Y_%m_%d")
    except Exception:
        date_fmt = "UNKNOWN_DATE"

    return f"output/{initials}_{date_fmt}" +"_" + MODEL_NAME + ".csv"

# Ensure output directory exists
Path("output").mkdir(parents=True, exist_ok=True)

# ==== EVALUATION METRICS ====
def rouge_score(reference, candidate):
    """
    Simple ROUGE-1 (unigram) F1 score: overlap of words between reference and candidate.
    Returns float 0.0–1.0.
    """
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())
    
    if not ref_words or not cand_words:
        return 0.0
    
    overlap = len(ref_words & cand_words)
    precision = overlap / len(cand_words) if cand_words else 0.0
    recall = overlap / len(ref_words) if ref_words else 0.0
    
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def retrieval_precision(retrieved_chunk_texts, relevant_chunk_texts):
    """
    Precision: of the chunks we retrieved, how many are in the relevant set?
    Returns float 0.0–1.0.
    """
    if not retrieved_chunk_texts:
        return 0.0
    
    # Normalize for comparison (lowercase, strip whitespace)
    retrieved_norm = {txt.lower().strip() for txt in retrieved_chunk_texts}
    relevant_norm = {txt.lower().strip() for txt in relevant_chunk_texts}
    
    overlap = len(retrieved_norm & relevant_norm)
    return overlap / len(retrieved_norm)

def retrieval_recall(retrieved_chunk_texts, relevant_chunk_texts):
    """
    Recall: of the relevant chunks, how many did we retrieve?
    Returns float 0.0–1.0.
    """
    if not relevant_chunk_texts:
        return 1.0  # No relevant chunks = perfect recall
    
    retrieved_norm = {txt.lower().strip() for txt in retrieved_chunk_texts}
    relevant_norm = {txt.lower().strip() for txt in relevant_chunk_texts}
    
    overlap = len(retrieved_norm & relevant_norm)
    return overlap / len(relevant_norm)
