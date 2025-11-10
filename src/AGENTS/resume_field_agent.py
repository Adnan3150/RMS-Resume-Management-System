import os
from math import ceil
import docx2txt
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from src.AGENTS import resume_fields_extractor
from langchain_groq import ChatGroq
import pytesseract
from docx2pdf import convert
from PIL import Image, ImageOps
import subprocess, os

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

def convert_docx_to_pdf(input_path, output_path=None):
    """
    Converts DOCX to PDF using LibreOffice CLI.
    """
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"

    cmd = [
        "soffice", "--headless",
        "--convert-to", "pdf",
        input_path,
        "--outdir", os.path.dirname(output_path)
    ]
    subprocess.run(cmd, check=True)
    return output_path

def docx_to_text(filepath: str) -> str:
    return docx2txt.process(filepath)

def extract_layout_text(pdf_path: str, poppler_path: str | None = None) -> str:
    """
    Layout-aware text extraction using LayoutLMv3 with:
      - safe image validation
      - token-chunking (<=512 tokens)
      - keeps pixel_values intact for each chunk
      - OCR fallback per page on failure
    """
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    all_text = ""

    for i, page_image in enumerate(pages, start=1):
        print(f"🔍 Processing page {i}...")

        # Validate and normalize PIL image
        if page_image is None:
            print(f"⚠️ Page {i} is None — skipping.")
            continue

        try:
            # Handle orientation tags and force RGB
            page_image = ImageOps.exif_transpose(page_image)
            image = page_image.convert("RGB")
        except Exception as e:
            print(f"⚠️ Could not convert page {i} to RGB: {e}. Using OCR fallback.")
            try:
                ocr_text = pytesseract.image_to_string(page_image)
                all_text += f"\n--- Page {i} (OCR fallback) ---\n{ocr_text}"
            except Exception as oerr:
                print(f"   OCR also failed: {oerr}")
            continue

        # Quick sanity checks
        arr = None
        try:
            import numpy as np
            arr = np.array(image)
            if arr.size == 0 or arr.shape[2] != 3:
                print(f"⚠️ Page {i} image array invalid shape {arr.shape}. Skipping.")
                continue
        except Exception:
            # if numpy not available or failed, keep going — processor will raise if broken
            pass

        # Encode with the LayoutLMv3 processor
        try:
            encoded = processor(images=image, return_tensors="pt", truncation=False)
        except Exception as e:
            print(f"⚠️ Processor failed on page {i}: {e}. Using OCR fallback.")
            try:
                ocr_text = pytesseract.image_to_string(image)
                all_text += f"\n--- Page {i} (OCR fallback) ---\n{ocr_text}"
            except Exception as oerr:
                print(f"   OCR also failed: {oerr}")
            continue

        # Ensure pixel_values are present and have 3 channels
        pixel_values = encoded.get("pixel_values", None)
        if pixel_values is None or pixel_values.shape[1] != 3:
            print(f"⚠️ Invalid pixel_values shape: {None if pixel_values is None else pixel_values.shape}. Skipping page.")
            continue

        # token tensors to chunk: input_ids, attention_mask, bbox, token_type_ids (if present)
        seq_len = encoded["input_ids"].shape[1]
        if seq_len == 0:
            print(f"⚠️ Page {i} produced zero-length input_ids. Skipping.")
            continue

        num_chunks = ceil(seq_len / 512)
        page_text = ""

        for chunk_idx in range(num_chunks):
            start = chunk_idx * 512
            end = min((chunk_idx + 1) * 512, seq_len)

            # Build chunk_input carefully:
            chunk_input = {}
            for k, v in encoded.items():
                # token-like tensors (shape [1, seq_len, ...]) should be sliced
                if k in ("input_ids", "attention_mask", "bbox", "token_type_ids"):
                    # handle 2D or 3D tensors safely
                    if v.ndim == 2:
                        chunk_input[k] = v[:, start:end]
                    elif v.ndim == 3:
                        chunk_input[k] = v[:, start:end, ...]
                    else:
                        # unexpected dim: keep as-is (but log)
                        chunk_input[k] = v
                else:
                    # keep pixel_values (and any other image-like tensors) intact
                    chunk_input[k] = v

            # Safe model inference
            try:
                with torch.no_grad():
                    outputs = model(**chunk_input)
            except Exception as e:
                print(f"⚠️ Model inference failed on page {i} chunk {chunk_idx}: {e}")
                print("   Falling back to OCR for this chunk.")
                try:
                    ocr_text = pytesseract.image_to_string(image)
                    page_text += " " + ocr_text
                except Exception as oerr:
                    print(f"   OCR also failed: {oerr}")
                continue

            # Convert ids -> tokens and build readable text
            pred_ids = torch.argmax(outputs.logits, dim=-1)[0].tolist()
            tokens = processor.tokenizer.convert_ids_to_tokens(chunk_input["input_ids"][0])
            tokens_cleaned = [t.replace("##", "") for t in tokens if t not in ("[PAD]", "[CLS]", "[SEP]")]
            page_text += " ".join(tokens_cleaned) + " "

        all_text += f"\n--- Page {i} ---\n{page_text.strip()}"

    return all_text.strip()




def parse_resume_with_llm(pdf_path: str, llm= None):
    """
    Combines LayoutLMv3 extraction + LLM field parsing.
    Returns structured resume fields.
    """
    ext = os.path.splitext(pdf_path)[1].lower()
    if ext in [".doc", ".docx"]:
        clean_text= docx_to_text(pdf_path)
        # pdf_path_new=pdf_path.replace(ext,".pdf")
        # print("pdf_path",pdf_path_new)
        # # convert(pdf_path, pdf_path_new)
        # output_path=convert_docx_to_pdf(pdf_path,pdf_path_new)
    else:
        print("📄 Extracting layout-based text...")
        layout_text = extract_layout_text(pdf_path)

        print("🧹 Cleaning extracted text...")
        clean_text = " ".join(layout_text.split())  # normalize whitespace

    print("🤖 Extracting structured fields with LLM...")
    structured_data = resume_fields_extractor.extract_fields(clean_text, llm)

    return structured_data
