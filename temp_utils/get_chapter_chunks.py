import fitz  # PyMuPDF
import os
import json
import sys
import re
from pathlib import Path

def sanitize_filename(text):
    """Convert text to a valid filename (lowercase, underscores, no special chars)."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text

def extract_text_from_pages(doc, start_page, end_page):
    """Extract and concatenate text from start_page to end_page inclusive (0-based)."""
    texts = []
    for i in range(start_page, end_page + 1):
        if i < doc.page_count:
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

def save_chunks_to_jsonl(chunks, chapter_name, output_dir):
    """Save chunks as a jsonl file with chapter and chunk_id metadata."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{chapter_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
            record = {
                "chapter": chapter_name,
                "chunk_id": idx,
                "text": chunk
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks for '{chapter_name}' to {out_path}")

def extract_chapters_from_toc(pdf_path):
    """Extract chapter ranges from PDF's table of contents."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    total_pages = doc.page_count
    doc.close()
    
    chapters = {}
    for i, (level, title, page) in enumerate(toc):
        # Only process level 1 entries (top-level chapters)
        if level == 1:
            chapter_name = sanitize_filename(title)
            
            # Find the end page: next chapter's start page - 1, or last page
            if i + 1 < len(toc):
                next_page = toc[i + 1][2]
                end_page = next_page - 1
            else:
                end_page = total_pages
            
            # Convert to 1-based page numbers for consistency
            chapters[chapter_name] = (page, end_page)
    
    return chapters

def extract_entire_pdf_as_chapter(pdf_path):
    """Fallback: treat entire PDF as a single chapter."""
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem
    total_pages = doc.page_count
    doc.close()
    
    return {sanitize_filename(pdf_name): (1, total_pages)}

def process_pdf(pdf_path, output_dir=None, max_words=400):
    """
    Extract chapters from a PDF and save as chunked JSONL files.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save chunks (defaults to Data/{pdf_name}_chunks)
        max_words: Word count per chunk (default 400)
    """
    # Validate PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found at {pdf_path}")
        return
    
    # Set output directory
    if output_dir is None:
        pdf_name = Path(pdf_path).stem
        output_dir = f"Data/{pdf_name}_chunks"
    
    print(f"📄 Processing: {pdf_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Extract chapters from TOC
    print("🔍 Extracting chapters from table of contents...")
    chapters = extract_chapters_from_toc(pdf_path)
    
    if not chapters:
        print("⚠️  No chapters found in TOC. Parsing entire PDF as a single chapter...")
        chapters = extract_entire_pdf_as_chapter(pdf_path)
    
    print(f"✅ Found {len(chapters)} chapter(s)\n")
    
    # Open PDF and extract text
    doc = fitz.open(pdf_path)
    
    for chapter_name, (start_page, end_page) in chapters.items():
        # Convert to zero-based page numbering for extraction
        start_idx = start_page - 1
        end_idx = end_page - 1
        
        print(f"Extracting '{chapter_name}': pages {start_page}-{end_page}")
        
        full_text = extract_text_from_pages(doc, start_idx, end_idx)
        
        if not full_text.strip():
            print(f"⚠️  Warning: No text extracted for {chapter_name}")
            continue
        
        chunks = chunk_text(full_text, max_words=max_words)
        save_chunks_to_jsonl(chunks, chapter_name, output_dir)
    
    doc.close()
    print(f"\n✅ Extraction complete!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 get_chapter_chunks.py <pdf_path> [output_dir] [max_words]")
        print("\nExample:")
        print("  python3 get_chapter_chunks.py Data/raw/HawaiianPlantLife.pdf")
        print("  python3 get_chapter_chunks.py Data/raw/DSM5.pdf Data/dsm5_chunks 400")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    max_words = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    
    process_pdf(pdf_path, output_dir, max_words)

if __name__ == "__main__":
    main()
