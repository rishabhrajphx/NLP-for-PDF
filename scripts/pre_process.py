import fitz  # PyMuPDF
import os

def extract_top_half_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        height = page.rect.height
        text += page.get_text("text", clip=(0, 0, page.rect.width, height/2))
    return text

if __name__ == "__main__":
    pdf_dir = "data/raw/"
    output_file = "data/processed/train_data.json"
    # Process each PDF and prepare the training data
    # This is a placeholder; you need to implement the annotation part