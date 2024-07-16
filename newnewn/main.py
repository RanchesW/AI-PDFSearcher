from flask import Flask, request, jsonify, send_from_directory, render_template
from pdfminer.high_level import extract_text
from transformers import pipeline
import pytesseract
from pdf2image import convert_from_path
import tempfile
import os
import torch
from torch.nn.functional import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load pre-trained models for QA, summarization, and text similarity
qa_model = pipeline('question-answering', model="DeepPavlov/bert-base-cased-conversational")
summarizer = pipeline("summarization", model="cointegrated/rut5-base-multitask")
nlp = pipeline('feature-extraction', model="sberbank-ai/ruBert-base", tokenizer="sberbank-ai/ruBert-base")

stop_words = set(nltk.corpus.stopwords.words('russian'))

def extract_text_from_pdf_page(pdf_path, page_number):
    try:
        text = extract_text(pdf_path, page_numbers=[page_number])
        print(f"Extracted text from page {page_number}: {text[:200]}...")
        return text
    except Exception as e:
        print(f"Failed to extract text from page {page_number}: {e}")
        return None

def extract_text_from_images_page(pdf_path, page_number):
    try:
        pages = convert_from_path(pdf_path, 300, first_page=page_number + 1, last_page=page_number + 1)
        if pages:
            text = pytesseract.image_to_string(pages[0], lang='rus+eng')
            print(f"Extracted text from image of page {page_number}: {text[:200]}...")
            return text
        return None
    except Exception as e:
        print(f"Failed to convert page {page_number} to image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def static_file(path):
    return send_from_directory('static', path)

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
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
    page_number = 0

    while True:
        # Try to extract text from the current page
        text = extract_text_from_pdf_page(temp_pdf_path, page_number)
        if not text:
            text = extract_text_from_images_page(temp_pdf_path, page_number)
        if not text:
            break  # Exit loop if no text is found

        print(f"Full text from page {page_number}: {text}")

        # Use cosine similarity to determine if the page contains relevant information
        if is_relevant_page(text, query):
            combined_text += " " + text

        page_number += 1

    # Clean up temporary files
    os.remove(temp_pdf_path)
    os.rmdir(temp_dir)

    if not combined_text:
        return jsonify({"summary": "No relevant information found in the document."})

    # Perform QA and summarization on the combined text
    page_results = search_relevant_snippets(combined_text, query)
    print(f"Relevant snippets from combined text: {page_results}")

    if not page_results:
        return jsonify({"summary": "No relevant information found in the document."})

    # Sort results by score
    page_results.sort(key=lambda x: x['score'], reverse=True)

    # Create a summary of the found answers
    combined_snippets = " ".join([result['snippet'] for result in page_results])
    summary = create_summary(combined_snippets, query)
    print("Summary:", summary)

    return jsonify({"summary": summary})

def is_relevant_page(text, query):
    query_tokens = word_tokenize(query.lower())
    query_vector = torch.tensor(nlp(query)).mean(dim=1)
    text_vector = torch.tensor(nlp(text)).mean(dim=1)
    similarity = cosine_similarity(text_vector, query_vector).item()

    if similarity > 0.5:  # Set a relevance threshold
        text_tokens = word_tokenize(text.lower())
        common_tokens = set(query_tokens).intersection(set(text_tokens))
        return len(common_tokens) > 1  # Check for multiple common words
    return False

def answer_question(context, question):
    result = qa_model(question=question, context=context)
    return result if result['score'] > 0.1 else None

def search_relevant_snippets(text, query):
    snippets = []
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    query_vector = torch.tensor(nlp(query)).mean(dim=1)
    for sentence in sentences:
        text_vector = torch.tensor(nlp(sentence)).mean(dim=1)
        similarity = cosine_similarity(text_vector, query_vector).item()
        if similarity > 0.3:  # Relevance threshold for snippets
            snippets.append({'snippet': sentence, 'score': similarity})
    return snippets

def create_summary(text, query):
    max_chunk = 1024  # Maximum length of text chunk for summarization
    text_chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    
    # Determine summarization length based on query
    if 'немного' in query or 'вкратце' in query:
        max_length = 50
        min_length = 20
    else:
        max_length = 150
        min_length = 40
    
    summarized_texts = []
    for chunk in text_chunks:
        print(f"Summarizing chunk: {chunk[:200]}...")  # Log the first part of the text chunk
        summarized_text = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summarized_texts.append(summarized_text[0]['summary_text'])
    
    summary = " ".join(summarized_texts)
    if len(word_tokenize(summary)) > 50:
        summary = " ".join(word_tokenize(summary)[:50])
    return summary

if __name__ == '__main__':
    app.run(debug=True, port=5000)
