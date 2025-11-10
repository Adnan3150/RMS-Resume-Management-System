import os
import re
import docx2txt
import unicodedata
import pdfplumber
import fitz  # PyMuPDF for layout-based extraction
from typing import List
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# --- DOCX HANDLER ---
def docx_to_text(filepath: str) -> str:
    return docx2txt.process(filepath)

# --- CLEANING ---
def clean_bullets(text: str) -> str:
    bullet_variants = [
        "\uf0fc", "\uf0b7", "\uf076", "\uf0a7", "", "▪", "", "◦", "■", "•"
    ]
    for b in bullet_variants:
        text = text.replace(b, "•")
    text = re.sub(r"•\s*", "• ", text)
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    return text.strip()

# --- PDF TEXT EXTRACTION (pdfplumber) ---
def extract_text_and_tables(pdf_path: str):
    data = {"text": [], "tables": []}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                data["text"].append({"page": page_num, "content": text})
            tables = page.extract_tables()
            for table in tables:
                data["tables"].append({"page": page_num, "table": table})
    return data

# --- OCR FALLBACK FOR SCANNED PDFs ---
def extract_text_with_ocr(pdf_path: str) -> str:
    print("🔍 Using OCR for scanned PDF...")
    pages = convert_from_path(pdf_path)
    text = ""
    for page_num, page in enumerate(pages, start=1):
        text_part = pytesseract.image_to_string(page)
        text += f"\n--- Page {page_num} ---\n{text_part}"
    return text

# --- ADVANCED LAYOUT EXTRACTION USING PyMuPDF ---
def extract_layout_text(pdf_path: str) -> str:
    print("📄 Using PyMuPDF layout-based extraction...")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        text_page = page.get_text("blocks")  # layout-preserving blocks
        text_page_sorted = sorted(text_page, key=lambda x: (x[1], x[0]))  # sort by position
        page_text = "\n".join([t[4] for t in text_page_sorted])
        text += f"\n--- Page {page_num} ---\n{page_text}"
    return text

# --- MAIN PIPELINE ---
def start_preprocess(file_path_list: List[str]):
    for file_path in file_path_list:
        ext = os.path.splitext(file_path)[1].lower()
        text_jd = ""

        if ext == ".pdf":
            # Try pdfplumber first
            jd_raw_data = extract_text_and_tables(file_path)
            text_jd = "".join([t['content'] for t in jd_raw_data["text"] if t['content']])

            # If no text, try layout-based extraction
            if not text_jd.strip():
                text_jd = extract_layout_text(file_path)

            # If still no text, fallback to OCR
            if not text_jd.strip():
                text_jd = extract_text_with_ocr(file_path)

            text_jd = unicodedata.normalize("NFKC", text_jd)

        elif ext in [".doc", ".docx"]:
            text_jd = docx_to_text(file_path)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text_jd = f.read()

        else:
            print(f"❌ Unsupported file format: {ext}")
            continue

        cleaned_text = clean_bullets(text_jd)
        return cleaned_text
