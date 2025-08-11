import gradio as gr
import torch
import pytesseract
from PIL import Image
import numpy as np
from docx import Document
import tempfile
import os
import fitz  # PyMuPDF for PDF reading
import google.generativeai as genai
import psycopg2
import io
from urllib.parse import urlparse
from datetime import datetime

# === CONFIG ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemma-3n-e4b-it")

# === Database ===
def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def create_table_if_not_exists():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id SERIAL PRIMARY KEY,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            raw_text TEXT,
            structured_text TEXT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_to_neon_db(filename, raw_text, structured_text):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prescriptions (filename, raw_text, structured_text)
            VALUES (%s, %s, %s)
        """, (filename, raw_text, structured_text))
        conn.commit()
        cursor.close()
        conn.close()
        return "✅ Saved to Neon DB."
    except Exception as e:
        return f"❌ Failed to save to DB: {e}"

# === OCR ===
def get_ocr_data(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    word_data = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and data['text'][i].strip():
            word = data['text'][i]
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_data.append({'text': word, 'box': [x, y, x + w, y + h], 'top': y, 'left': x})
    return word_data

def group_words_into_lines(word_data, line_threshold=15):
    word_data = sorted(word_data, key=lambda x: x['top'])
    lines, current_line, last_top = [], [], None
    for word in word_data:
        if last_top is None or abs(word['top'] - last_top) <= line_threshold:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda x: x['left']))
            current_line = [word]
        last_top = word['top']
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['left']))
    return lines

# === Main Extraction Logic ===
def process_prescription(file):
    if not file:
        return None, None, "❌ Please upload a file first."

    create_table_if_not_exists()

    file_ext = file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(file.read())
        temp_path = tmp_file.name

    # Handle PDF
    if file_ext == "pdf":
        pdf = fitz.open(temp_path)
        pix = pdf[0].get_pixmap(dpi=300)
        image_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        image = Image.open(temp_path).convert("RGB")

    # OCR
    word_data = get_ocr_data(image)
    lines = group_words_into_lines(word_data)

    raw_text = "\n".join([" ".join([word['text'] for word in line]) for line in lines])

    # Save raw docx
    raw_doc = Document()
    raw_doc.add_heading('Raw Prescription Text', level=1)
    for line in lines:
        raw_doc.add_paragraph(" ".join([word['text'] for word in line]))
    raw_docx_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    raw_doc.save(raw_docx_path)

    # Gemini Structuring
    prompt = f"""
Here is a raw OCR extracted text from a handwritten medical prescription.

Please convert it into a **well-structured document** that includes:
- Hospital Name
- Patient Name
- Age / Gender
- Address
- Date of Visit
- Doctor Name & Degree
- Diagnosis (if found)
- Medications with dosage & timing
- Instructions
- Follow-up Advice (if any)

Here is the raw text:
\"\"\" 
{raw_text} 
\"\"\" 

Return only the well-structured formatted result.
"""
    response = model.generate_content(prompt)
    structured_text = response.text.strip()

    # Save structured docx
    structured_doc = Document()
    structured_doc.add_heading('Structured Prescription', level=1)
    for paragraph in structured_text.split("\n\n"):
        structured_doc.add_paragraph(paragraph.strip())
    structured_docx_path = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    structured_doc.save(structured_docx_path)

    # Save to DB
    db_status = save_to_neon_db(file.name, raw_text, structured_text)

    return raw_docx_path, structured_docx_path, db_status

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Handwritten Prescription Structuring (Gradio)")

    with gr.Row():
        file_input = gr.File(label="Upload Prescription Image or PDF", file_types=[".jpg", ".jpeg", ".png", ".pdf"])
    
    start_btn = gr.Button("🚀 Start Extraction")

    with gr.Row():
        raw_download = gr.File(label="📄 Download Raw Prescription")
        structured_download = gr.File(label="📑 Download Structured Prescription")
    
    db_status_output = gr.Textbox(label="Database Status", interactive=False)

    start_btn.click(
        process_prescription,
        inputs=file_input,
        outputs=[raw_download, structured_download, db_status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
