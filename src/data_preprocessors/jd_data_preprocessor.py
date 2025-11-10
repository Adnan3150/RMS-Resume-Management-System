import os
import re
import docx2txt
import unicodedata
import pdfplumber
import unicodedata
from typing import List

def docx_to_text(filepath: str) -> str:
    return docx2txt.process(filepath)

def clean_bullets(text: str) -> str:
    # Common PDF bullet unicode values
    bullet_variants = [
        "\uf0fc",  # square-like bullet
        "\uf0b7",  # round bullet
        "\uf076",  # arrow bullet
        "\uf0a7",  # diamond bullet
    ]
    
    for b in bullet_variants:
        text = text.replace(b, "•")  # replace with standard bullet
    
    # Optional: normalize multiple spaces after bullet
    text = re.sub(r"•\s*", "• ", text)
    return text

def extract_text_and_tables(pdf_path: str):
    data = {"text": [], "tables": []}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text
            text = page.extract_text()
            if text:
                data["text"].append({"page": page_num, "content": text})
            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                data["tables"].append({"page": page_num, "table": table})
    return data

def start_preprocess(file_path_list : List):
    for file_path in file_path_list:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            jd_raw_data=extract_text_and_tables(file_path)
            text_jd=""
            for t in jd_raw_data["text"]:
                text_jd=text_jd+t['content']
            text_jd = unicodedata.normalize("NFKC", text_jd)

        elif ext in [".doc", ".docx"]:
            text_jd= docx_to_text(file_path)
        elif ext ==".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text_jd= f.read()
        else:
            return False
            
        cleaned_text = clean_bullets(text_jd)
        return cleaned_text





# # Example usage
# pdf_data = extract_text_and_tables("Data Scientist_Job Description_Sept_2025 (2).pdf")

# # Print text
# for t in pdf_data["text"]:
#     print(f"\n--- Page {t['page']} ---\n{t['content'].replace("uf0b7","-")}")

# # Print tables
# for tab in pdf_data["tables"]:
#     print(f"\n--- Table on Page {tab['page']} ---")
#     for row in tab["table"]:
#         print(row)