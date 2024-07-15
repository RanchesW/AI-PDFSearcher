from flask import Flask, request, jsonify, send_from_directory, render_template
from pdfminer.high_level import extract_text
import re
from transformers import pipeline
import pytesseract
from pdf2image import convert_from_path
import tempfile
import os
import torch

app = Flask(__name__)

# Загрузка предобученных моделей для оценки релевантности текста, вопрос-ответ и суммаризации
qa_model = pipeline('question-answering', model="DeepPavlov/bert-base-cased-conversational")
summarizer = pipeline("summarization", model="cointegrated/rut5-base-multitask")
nlp = pipeline('feature-extraction', model="sberbank-ai/ruBert-base", tokenizer="sberbank-ai/ruBert-base")

def extract_text_from_pdf_page(pdf_path, page_number):
    """Извлекает текст с указанной страницы PDF."""
    try:
        text = extract_text(pdf_path, page_numbers=[page_number])
        print(f"Extracted text from page {page_number}: {text[:200]}...")
        return text
    except Exception as e:
        print(f"Failed to extract text from page {page_number}: {e}")
        return None

def extract_text_from_images_page(pdf_path, page_number):
    """Извлекает текст с указанной страницы PDF, преобразуя её в изображение."""
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

    # Сохраняем PDF во временный файл
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    pdf_file.save(temp_pdf_path)

    results = []
    page_number = 0
    max_results = 3  # Максимальное количество релевантных ответов, после которого остановимся

    while True:
        # Попробуем извлечь текст с текущей страницы
        text = extract_text_from_pdf_page(temp_pdf_path, page_number)
        if not text:
            text = extract_text_from_images_page(temp_pdf_path, page_number)
        if not text:
            break  # Если текст не найден, выходим из цикла

        print(f"Full text from page {page_number}: {text}")
        page_results = search_relevant_snippets(text, query)
        print(f"Relevant snippets from page {page_number}: {page_results}")
        
        for snippet in page_results:
            answer = answer_question(snippet, query)
            print(f"Answer from page {page_number} snippet: {answer}")
            if answer:
                results.append(answer)
            if len(results) >= max_results:
                break  # Останавливаемся, если найдено достаточно релевантных ответов

        if len(results) >= max_results:
            break  # Останавливаемся, если найдено достаточно релевантных ответов

        page_number += 1

    # Удаляем временные файлы
    os.remove(temp_pdf_path)
    os.rmdir(temp_dir)

    # Сортируем результаты по вероятности
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Создаем резюме найденных ответов
    combined_text = " ".join([result['answer'] for result in results])
    summary = create_summary(combined_text, query)
    print("Summary:", summary)

    return jsonify({"summary": summary})

def answer_question(context, question):
    result = qa_model(question=question, context=context)
    return result if result['score'] > 0.1 else None

def search_relevant_snippets(text, query):
    """Ищет релевантные фрагменты текста для заданного запроса."""
    snippets = []
    sentences = text.split('.')
    for sentence in sentences:
        text_vector = torch.tensor(nlp(sentence)).mean(dim=1)
        query_vector = torch.tensor(nlp(query)).mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(text_vector, query_vector).item()
        if similarity > 0.3:  # Порог для релевантности
            snippets.append(sentence)
    return snippets

def create_summary(text, query):
    """Создает резюме текста, учитывая запрос пользователя."""
    max_chunk = 1024  # Максимальная длина куска текста для суммаризации
    text_chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    
    # Определяем длину суммаризации на основе запроса
    if 'немного' in query or 'вкратце' in query:
        max_length = 50
        min_length = 20
    else:
        max_length = 150
        min_length = 40
    
    summarized_texts = []
    for chunk in text_chunks:
        print(f"Summarizing chunk: {chunk[:200]}...")  # Логирование первой части куска текста
        summarized_text = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summarized_texts.append(summarized_text[0]['summary_text'])
    
    return " ".join(summarized_texts)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
