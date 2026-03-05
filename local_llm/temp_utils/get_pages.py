import fitz

PDF_PATH = "/Users/bennettoconnell/Public/Research/kilotech/Data/raw/HawaiianPlantLife.pdf"

doc = fitz.open(PDF_PATH)
total_pages = doc.page_count
print(f"Total pages: {total_pages}")
doc.close()