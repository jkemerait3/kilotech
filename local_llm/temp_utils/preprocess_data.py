# Olive/preprocess_data.py

import os
import pandas as pd
import json

# Where to save processed files (change if needed)
PROCESSED_DIR = "output/processed_chunks"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# === OCD DATASET PREPROCESSING ===
def preprocess_ocd(csv_path):
    df = pd.read_csv(csv_path)
    chunk_texts = []
    for idx, row in df.iterrows():
        # Build unified text chunk per patient
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunk_texts.append(text)
    # Save as a JSONL file (one chunk per line)
    out_path = os.path.join(PROCESSED_DIR, "ocd_chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunk_texts:
            f.write(json.dumps({"text": chunk}) + "\n")
    print(f"OCD preprocessed, {len(chunk_texts)} rows saved to: {out_path}")

# === PTSD DATASET PREPROCESSING ===
def preprocess_ptsd(csv_path):
    df = pd.read_csv(csv_path)
    chunk_texts = []
    for idx, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunk_texts.append(text)    
    out_path = os.path.join(PROCESSED_DIR, "ptsd_chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunk_texts:
            f.write(json.dumps({"text": chunk}) + "\n")
    print(f"PTSD preprocessed, {len(chunk_texts)} rows saved to: {out_path}")


# === DAIC-WOZ TRANSCRIPTS PREPROCESSING ===
def preprocess_daic_woz(transcript_dir):
    chunk_texts = []
    for filename in os.listdir(transcript_dir):
        if filename.endswith(".csv") or filename.endswith(".txt"):
            path = os.path.join(transcript_dir, filename)
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(path)
                    # If columns named 'speaker' and 'utterance' (adjust if needed)
                    colnames = [c.lower() for c in df.columns]
                    for idx, row in df.iterrows():
                        # Use speaker and utterance columns, or adjust to file format
                        if 'speaker' in colnames and 'utterance' in colnames:
                            text = f"{row['speaker']}: {row['utterance']}"
                        else:
                            # Fallback: join all columns as text if unsure
                            text = " | ".join([str(cell) for cell in row.values])
                        chunk_texts.append(text)
                elif filename.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            chunk_texts.append(line.strip())
            except Exception as e:
                print(f"❌ Error parsing {filename}: {e}")
    out_path = os.path.join(PROCESSED_DIR, "daic_woz_chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunk_texts:
            f.write(json.dumps({"text": chunk}) + "\n")
    print(f"DAIC-WOZ preprocessed, {len(chunk_texts)} chunks saved to: {out_path}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Set your filenames/paths here
    OCD_CSV = "Data/ocd_patient_dataset.csv"
    PTSD_CSV = "Data/PTSD-Repository-Study-Characteristics.csv"
    DAIC_WOZ_DIR = "Data/DAIC-WOZ Transcripts"

    if os.path.exists(OCD_CSV):
        preprocess_ocd(OCD_CSV)
    else:
        print(f"OCD dataset not found at {OCD_CSV}")
    if os.path.exists(PTSD_CSV):
        preprocess_ptsd(PTSD_CSV)
    else:
        print(f"PTSD repository not found at {PTSD_CSV}")
    if os.path.exists(DAIC_WOZ_DIR):
        preprocess_daic_woz(DAIC_WOZ_DIR)
    else:
        print(f"DAIC-WOZ transcripts folder not found at {DAIC_WOZ_DIR}")
