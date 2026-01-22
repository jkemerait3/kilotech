import fitz  # PyMuPDF
import os
import json
import textwrap

# Directory to save extracted JSONL chunks
OUTPUT_DIR = "dsm5_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to your DSM-5 PDF file (adjust if needed)
PDF_PATH = "DSM5.pdf"

# Mapping of disorder name to PDF page ranges (inclusive, 1-based page numbers)
# PyMuPDF pages are 0-indexed internally, so we subtract 1 when extracting
DISORDERS_PAGE_RANGES = {
    "depression": (205, 223),       # 205-223 inclusive (mapped from your pdf range 205-123 assumed typo: corrected to 205-223)
    "anxiety_panic": (267, 271),
    "ptsd": (316, 325),
    "substance_use": (526, 535),
    "adhd": (104, 110),
    "ocd": (280, 287),
    "asd": (95, 104),
    "bipolar": (168, 184),
    "psychotic_disorders": (132, 135),
    "personality_pathology": (690, 694),
}

def extract_text_from_pages(doc, start_page, end_page):
    """Extract and concatenate text from start_page to end_page inclusive (0-based)."""
    texts = []
    for i in range(start_page, end_page + 1):
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def chunk_text(text, max_words=400):
    """Split text into chunks of approximately max_words words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i+max_words]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text.strip())
    return chunks

def save_chunks_to_jsonl(chunks, disorder_name):
    """Save chunks as a jsonl file with disorder and chunk_id metadata."""
    out_path = os.path.join(OUTPUT_DIR, f"{disorder_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
            record = {
                "disorder": disorder_name,
                "chunk_id": idx,
                "text": chunk
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks for '{disorder_name}' to {out_path}")

def main():
    if not os.path.exists(PDF_PATH):
        print(f"DSM-5 PDF file not found at {PDF_PATH}")
        return
    
    doc = fitz.open(PDF_PATH)
    
    for disorder, (start_page, end_page) in DISORDERS_PAGE_RANGES.items():
        # Convert to zero-based page numbering
        start_idx = start_page - 1
        end_idx = end_page - 1
        
        print(f"Extracting {disorder}: pages {start_page}-{end_page} (zero-based {start_idx}-{end_idx})")
        
        full_text = extract_text_from_pages(doc, start_idx, end_idx)
        
        if not full_text.strip():
            print(f"Warning: No text extracted for {disorder} pages {start_page}-{end_page}")
            continue
        
        chunks = chunk_text(full_text, max_words=400)
        save_chunks_to_jsonl(chunks, disorder)

    doc.close()

if __name__ == "__main__":
    main()
