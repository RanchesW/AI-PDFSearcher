import os
from flask import Flask, request, jsonify, send_from_directory, render_template
import tempfile
import torch
import faiss
import numpy as np
from transformers import pipeline
from PIL import Image
import pytesseract
import logging
import time
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load LLaMA 13B model for text generation and correction
llama_model = pipeline('text-generation', model="path_to_llama_13B_model")  # Update with the actual model path
nlp = pipeline('feature-extraction', model="sberbank-ai/ruBert-base", tokenizer="sberbank-ai/ruBert-base")

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='rus+eng')
        print(f"Extracted text from image: {text[:200]}...")
        return text
    except Exception as e:
        print(f"Failed to extract text from image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def static_file(path):
    return send_from_directory('static', path)

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    start_time = time.time()
    print("Received files:", request.files)
    print("Received form data:", request.form)

    pdf_file = request.files.get('file')
    query = request.form.get('query')

    if not pdf_file or not query:
        return jsonify({'error': 'No file or search query provided'}), 400

    # Save PDF to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    pdf_file.save(temp_pdf_path)

    combined_text = ""

    # Extract images from PDF
    pdf_document = fitz.open(temp_pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(temp_dir, f"image{page_num+1}_{img_index+1}.{image_ext}")
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            text = extract_text_from_image(image_path)
            if text:
                combined_text += " " + text

    # Clean up temporary files
    os.remove(temp_pdf_path)
    os.rmdir(temp_dir)

    if not combined_text.strip():
        return jsonify({"summary": "No relevant information found in the document."})

    # Use LLaMA model to clean and correct the text
    cleaned_text = llama_model(combined_text, max_length=500)[0]['generated_text']

    # Prepare FAISS index
    sentences = [sent.strip() for sent in cleaned_text.split('.') if sent.strip()]
    sentence_vectors = []
    for sentence in sentences:
        vec = torch.tensor(nlp(sentence)).mean(dim=1).numpy().flatten()
        if vec.ndim == 1:
            sentence_vectors.append(vec)
        else:
            print(f"Skipping sentence due to incorrect vector shape: {sentence}")

    if len(sentence_vectors) == 0:
        return jsonify({"summary": "Failed to vectorize sentences."})

    sentence_vectors = np.array(sentence_vectors)
    print(f"Shape of sentence_vectors: {sentence_vectors.shape}")

    dimension = sentence_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(sentence_vectors)

    # Query vector
    query_vector = torch.tensor(nlp(query)).mean(dim=1).numpy().flatten().reshape(1, -1)
    print(f"Query vector shape: {query_vector.shape}")

    # Search in FAISS index
    k = 10  # number of top relevant results to retrieve
    distances, indices = index.search(query_vector, k)
    relevant_snippets = [sentences[i] for i in indices[0]]

    # Ensure unique and non-similar snippets
    unique_snippets = list(dict.fromkeys(relevant_snippets))
    filtered_snippets = []
    for snippet in unique_snippets:
        if not any(difflib.SequenceMatcher(None, snippet, filtered_snippet).ratio() > 0.7 for filtered_snippet in filtered_snippets):
            filtered_snippets.append(snippet)

    # Create summary and generate answer
    combined_snippets = " ".join(filtered_snippets)
    summary = llama_model(combined_snippets, max_length=500)[0]['generated_text']
    response = llama_model(f"Вопрос: {query}\nТекст: {summary}\nОтвет:", max_length=500)
    answer = response[0]['generated_text']

    # Logging resource usage
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    gpus = GPUtil.getGPUs()
    gpu_info = gpus[0] if gpus else None

    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
    if gpu_info:
        logging.info(f"GPU Memory Usage: {gpu_info.memoryUsed} MB / {gpu_info.memoryTotal} MB")
    
    end_time = time.time()
    logging.info(f"Processing time: {end_time - start_time} seconds")

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
