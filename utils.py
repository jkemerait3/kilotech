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
