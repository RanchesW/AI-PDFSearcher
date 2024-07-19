# AI-PDFSearcher

Веб-приложение на **Flask** использует различные модели машинного обучения для анализа текста в **PDF-документах**. Оно поддерживает функционал извлечения текста как напрямую из **PDF**, так и из изображений внутри них, ответы на вопросы, основанные на тексте, и создание резюме текста.

## Основные функции и рабочий процесс:

### Извлечение текста из PDF и изображений:

Для извлечения текста я использую довольн известную библиотеку которая позволяет распознавать и “**читать**” текст, встроенный в изображения. **Python-tesseract** - это инструмент оптического распознавания символов (OCR) для python.

Тут я написал функцию которая должна отделять текст от фотографии с помощью **Tesseract**, также добавив расширения в плане языков, теперь библиотека может читать русский также как и английский язык:

```
    import pytesseract
    
    def extract_text_from_image(image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='rus+eng')
            print(f"Extracted text from image: {text[:200]}...")
            return text
```
- Открывает изображение по указанному пути с помощью `Image.open(image_path)`.
- Извлекает текст из изображения с использованием `pytesseract.image_to_string(image, lang='rus+eng')`, задав языки для распознавания текста как русский и английский.
- Печатает первые 200 символов извлеченного текста для проверки.
- Возвращает извлеченный текст.

```
    except Exception as e:
        print(f"Failed to extract text from image: {e}")
        return None
```

Если возникает ошибка при открытии изображения или извлечении текста, печатает сообщение об ошибке и возвращает `None`.

В дальнейшем мною был написан кусок кода который позволяет работать функции сверху, а именно была использована библиотека Fitz/PyMuPDF для просмотра, рендеринга и инструментов для работы с такими форматами как PDF, XPS, OpenXPS, CBZ, EPUB и FB2.

```
    pdf_document = fitz.open(temp_pdf_path)
```

Открывает PDF-документ, находящийся по пути temp_pdf_path, с использованием библиотеки PyMuPDF (импортируемой как fitz).

```
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
```
Проходит по каждой странице PDF-документа, загружая страницу по номеру.

```
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_path = os.path.join(temp_dir, f"image{page_num+1}_{img_index+1}.{image_ext}")
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
```

**Для каждого изображения:**
- Получает ссылку на изображение xref.
- Извлекает изображение с помощью pdf_document.extract_image(xref).
- Получает байты изображения и его расширение.
- Формирует путь для сохранения изображения.
- Сохраняет изображение в файл по указанному пути.

```
    text = extract_text_from_image(image_path)
    if text:
        combined_text += " " + text
```

Использует функцию extract_text_from_image для извлечения текста из сохраненных изображений. Если текст найден, добавляет его к переменной combined_text.

> [!NOTE]
> 
> **Этот код использует следующие компоненты:**
>
> + Библиотека PyMuPDF (fitz) для работы с PDF-документами и извлечения изображений.
>
> + Библиотека pytesseract для оптического распознавания текста (OCR).
>
> + Библиотека PIL (Pillow) для работы с изображениями.

### Хранения PDF-файлов во временном хранилище:

    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, pdf_file.filename)
    pdf_file.save(temp_pdf_path)

Функция `tempfile.mkdtemp()` создает временный каталог и возвращает путь к нему.

С помощью функции `os.path.join` формируется полный путь к временному **PDF-файлу**, используя имя исходного файла `pdf_file.filename` и путь к временному каталогу `temp_dir`.

Используя метод `save` объекта `pdf_file`, PDF-файл сохраняется по сформированному временному пути `temp_pdf_path`.

### Очистка временных файлов

Удаляет временный PDF-файл, расположенный по пути temp_pdf_path:

```
    os.remove(temp_pdf_path)
```

Удаляет временный каталог, расположенный по пути temp_dir:

```
    os.rmdir(temp_dir)
```

Проверяет, является ли переменная combined_text пустой или состоит только из пробелов. Если это так, возвращает JSON-ответ с сообщением о том, что в документе не найдено релевантной информации:

```
    if not combined_text.strip():
        return jsonify({"summary": "No relevant information found in the document."})
```

### Токенизация текст

Токенизации текста перед генерацией нового текста:

```
    input_ids = tokenizer.encode(combined_text, return_tensors="pt")
    max_new_tokens = max(1, 500 - input_ids.shape[1])  # Ensure max_new_tokens is always at least 1
```

+ Текст combined_text кодируется в тензор с использованием токенизатора tokenizer.

+ Определяется максимальное количество новых токенов для генерации, которое всегда будет не менее 1.

### LLaMA для очистки и исправления текста 

Этот код генерирует очищенный текст с помощью модели LLaMA, если указано положительное количество новых токенов для генерации. Если нет новых токенов для генерации, то используется оригинальный текст.:

```
    # Generate cleaned text
    if max_new_tokens > 0:
        cleaned_text = llama_model(combined_text, truncation=True, max_length=input_ids.shape[1] + max_new_tokens)[0]['generated_text']
    else:
        cleaned_text = combined_text  # Fallback to the original text if no new tokens can be generated
```

+ Если max_new_tokens больше 0, используется модель llama_model для генерации очищенного текста.
+ Если max_new_tokens равно 0, используется исходный текст combined_text.

### Подготовка индекса FAISS:

1.  Преобразование предложений в векторы:
```
    sentences = [sent.strip() for sent in cleaned_text.split('.') if sent.strip()]
    sentence_vectors = np.array([torch.tensor(nlp(sentence)).mean(dim=1).numpy().flatten() for sentence in sentences if sentence])
    if len(sentence_vectors) == 0:
        return jsonify({"summary": "Failed to vectorize sentences."})
```
Разбивает текст `cleaned_text` на предложения по точке и удаляет пустые строки и пробелы.

2. Преобразование запроса в вектор:
```
    query_vector = torch.tensor(nlp(query)).mean(dim=1).numpy().flatten().reshape(1, -1)
    print(f"Query vector shape: {query_vector.shape}")
```
**Для каждого предложения:**

- Преобразует предложение в векторное представление с использованием модели nlp.
- Берет среднее значение вдоль первого измерения вектора и преобразует его в numpy массив.
- Проверяет, что вектор имеет правильную форму (одномерный массив). Если это так, добавляет его в sentence_vectors, иначе пропускает предложение.
  
3. Проверяет, удалось ли векторизовать предложения:
  
```
    if len(sentence_vectors) == 0:
        return jsonify({"summary": "Failed to vectorize sentences."})
```

4. Преобразует список векторов в массив numpy:

```
    sentence_vectors = np.array(sentence_vectors)
    print(f"Shape of sentence_vectors: {sentence_vectors.shape}")
```

5. Создает и добавляет векторы в **FAISS** индекс:

```
    index = faiss.IndexFlatL2(sentence_vectors.shape[1])
    index.add(sentence_vectors)
```
- Определяет размерность векторов.
- Создает индекс FAISS для поиска по близости (`IndexFlatL2`).
- Добавляет векторы предложений в индекс.

> [!NOTE]
>
> **Этот код использует следующие компоненты:**
>
> + Модуль torch для работы с тензорами.
> 
> + Модель nlp, которая обрабатывает предложения и создает векторные представления.
> 
> + Библиотека numpy для работы с массивами.
> 
> + Библиотека faiss для создания и управления индексом ближайших соседей.

### Создание пользовательского запроса(NLP):

Этот код выполняет преобразование текстового запроса в векторное представление с использованием модели nlp и библиотеки torch, а затем выводит форму полученного вектора. Вот краткое объяснение каждого шага:

```
    query_vector = torch.tensor(nlp(query)).mean(dim=1).numpy().flatten().reshape(1, -1)
    print(f"Query vector shape: {query_vector.shape}")
```

`nlp(query)` - обрабатывает текстовый запрос и преобразует его в набор векторных представлений.

`torch.tensor(...)` - преобразует результат в тензор PyTorch.

`mean(dim=1)` - вычисляет среднее значение вдоль первого измерения (если результат многомерный, это обычно для объединения векторов слов в одно представление).

`numpy()` - преобразует тензор PyTorch в массив NumPy.

`flatten()` - преобразует многомерный массив в одномерный.

`reshape(1, -1)` - изменяет форму массива на 2D, где первая размерность равна 1 (это необходимо для поиска в FAISS, который ожидает 2D массив для поиска ближайших соседей).


### Выполнения поиск в FAISS индексе

1. Устанавливает значение равным 10, что означает, что будут извлекаться 10 наиболее релевантных результатов:

Использует метод search индекса FAISS для поиска ближайших 10 векторов к query_vector. Возвращает расстояния до этих векторов и их индексы:

```
    _, indices = index.search(query_vector, 10)
```

3. Использует индексы, возвращенные FAISS, для извлечения соответствующих предложений из списка sentences:

```
    relevant_snippets = [sentences[i] for i in indices[0]]
```

### Уникальность и непохожесть фрагментов
```
    import difflib

    # Ensure unique and non-similar snippets
    unique_snippets = list(dict.fromkeys(relevant_snippets))
    filtered_snippets = []
    for snippet in unique_snippets:
        if not any(difflib.SequenceMatcher(None, snippet, filtered_snippet).ratio() > 0.7 for filtered_snippet in filtered_snippets):
            filtered_snippets.append(snippet)
```

Использует словарь для удаления дубликатов из списка `relevant_snippets` и сохраняет только уникальные фрагменты текста.

**Для каждого уникального фрагмента:**

- Проверяет, не является ли он слишком похожим (более чем на 70%) на уже добавленные фрагменты в filtered_snippets.
- Если фрагмент не слишком похож, добавляет его в filtered_snippets.

> [!NOTE]
>
> **Этот код использует следующие компонент:**
>
> + Модуль difflib для сравнения схожести строк с помощью SequenceMatcher.


### Создание резюме и генерация ответ

1. Соединяет все фрагменты из `filtered_snippets` в одну строку, разделяя их пробелами:
```
     summary = ' '.join(filtered_snippets)
```

2. Отправляет модель `llama_model` запрос, включающий вопрос и созданное краткое содержание, чтобы получить ответ. Ограничивает длину ответа 500 символами:
```
    final_response = llama_model(f"Вопрос: {query}\nТекст: {summary}\nОтвет:", max_length=500)[0]['generated_text']
```

### Ведение журнала использования ресурсов

1. Использует psutil для получения информации о текущем процессе по его идентификатору (PID):

```
    process = psutil.Process(os.getpid())
```

2. Измеряет процент использования ЦПУ процессом в течение интервала времени (1 секунда):

```
    cpu_percent = process.cpu_percent(interval=1)
```

3. Извлекает информацию об использовании памяти процессом:

```
    memory_info = process.memory_info()
```

4. Использует GPUtil для получения информации о доступных GPU. Если доступен хотя бы один GPU, берет информацию о первом:

```
    gpus = GPUtil.getGPUs()
    gpu_info = gpus[0] if gpus else None
```

5. Логирует использование ЦПУ:

```
    logging.info(f"CPU Usage: {cpu_percent}%")
```

6. Преобразует использование памяти в мегабайты (MB) и логирует его:

```
    logging.info(f"Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
```

7. Логирует использование GPU (если доступно):

```
    if gpu_info:
        logging.info(f"GPU Memory Usage: {gpu_info.memoryUsed} MB / {gpu_info.memoryTotal} MB")
```

8. Измеряет и логирует общее время выполнения процесса:

```
    end_time = time.time()
    logging.info(f"Processing time: {end_time - start_time} seconds")
```

> [!NOTE]
>
> **Этот код использует следующие компоненты:**
>
> + Модуль psutil для получения информации о ЦПУ и памяти.
> 
> + Модуль GPUtil для получения информации о GPU.
> 
> + Модуль logging для логирования информации.
> 
> + Модуль time для измерения времени выполнения.
> 
> + Модуль flask (предполагается) для возврата JSON ответа с использованием jsonify.

                                                                                                                                                                                                                                                                           
### Веб-интерфейс:

Позволяет пользователям загружать **PDF-файлы** и отправлять запросы о их содержимом.
Бэкенд обрабатывает эти входные данные, обрабатывает PDF и возвращает ответ в формате **JSON** на основе запроса.






