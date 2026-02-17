import fitz  # PyMuPDF

PDF_PATH = "/Users/bennettoconnell/Public/Research/kilotech/Data/raw/HawaiianPlantLife.pdf"

doc = fitz.open(PDF_PATH)
toc = doc.get_toc()
print("Table of Contents:")
for level, title, page in toc:
    print(f"Level {level}: {title} - Page {page}")
doc.close()