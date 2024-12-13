# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # 1. Список книг з назвами та відгуками
# books = [
#     {"title": "Том Сойєр", "review": "Ця книга розповідає про пригоди хлопця, який живе в місті на річці. Він завжди потрапляє у цікаві ситуації."},
#     {"title": "Війна і мир", "review": "Класичний роман, що описує життя аристократії в Росії під час наполеонівських війн."},
#     {"title": "1984", "review": "Дистопія, яка описує тоталітарне суспільство під владою Великого Брата, що контролює всі аспекти життя."},
#     {"title": "Гаррі Поттер", "review": "Книга про хлопця, який дізнається, що він чарівник, і навчається в магічній школі."},
#     {"title": "Майстер і Маргарита", "review": "Цей роман об'єднує фантастичні елементи з реальними подіями, ставлячи під сумнів віру і мораль."}
# ]
#
# # 2. Опис книги, який надає користувач
# user_query = "світ, де головний герой бореться з темними силами в школах."
#
# # 3. Об'єднуємо назву та відгук книги в одну строку
# book_texts = [f"{book['title']} {book['review']}" for book in books]
#
# # Додаємо запит користувача до списку
# texts = book_texts + [user_query]
#
# # 4. Використовуємо TF-IDF для векторизації текстів
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(texts)
#
# # 5. Обчислюємо косинусну подібність між запитом і всіма книгами
# cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#
# # 6. Виводимо книги з найбільшою схожістю
# similarities = cosine_similarities[0]
# sorted_books = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
#
# print("Схожі книги на ваш запит:")
# for idx, similarity in sorted_books[:3]:  # виводимо топ-3 схожих книги
#     print(f"Книга: {books[idx]['title']}, Схожість: {similarity:.4f}")

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Завантаження моделі для отримання векторних представлень тексту
# model = SentenceTransformer(
#     'paraphrase-MiniLM-L6-v2')  # Ви також можете вибрати іншу модель, наприклад, 'bert-base-nli-mean-tokens'
#
# # Тексти для порівняння
# texts = [
#     "I love programming in Python.",
#     "Python is my favorite language for coding.",
#     "I enjoy coding with Python.",
#     "I am learning machine learning with Python."
# ]
#
#
# # Функція для обчислення схожості між текстами
# def calculate_similarity(text1, text2):
#     # Генерація векторів текстів
#     embeddings1 = model.encode([text1])
#     embeddings2 = model.encode([text2])
#
#     # Обчислення косинусної схожості між векторами
#     similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
#     return similarity
#
# user_description = "I love books."
# book_descriptions = [
#     "A young wizard embarks on an adventure to save the world from an ancient curse.",
#     "A detective investigates a series of mysterious deaths in a small town.",
#     "A spaceship travels to distant planets to explore new civilizations.",
#     "This is a story about a boy who likes sucking big fat black cocks",
#     "a b c d"
# ]
#
# # # Порівняння першого тексту з іншими
# # target_text = "I love Python."
# # for i, text in enumerate(texts):
# #     similarity = calculate_similarity(target_text, text)
# #     print(f"Similarity with text {i + 1}: {similarity:.4f}")
#
# for i, text in enumerate(book_descriptions):
#     similarity = calculate_similarity(user_description, text)
#     print(f"Similarity with text {i + 1}: {similarity:.4f}")

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Завантажуємо DistilBERT
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
#
# # Тексти для порівняння
# texts = [
#     # Приклад 1
#     ("The rapid development of artificial intelligence will lead to significant changes in various industries.",
#      "The swift growth of AI is expected to cause major transformations across multiple sectors."),
#
#     # Приклад 2
#     ("The company's new product line is a game-changer in the tech industry.",
#      "The new range of products from the company is revolutionary for the tech sector."),
#
#     # Приклад 3
#     ("Despite the early setbacks, the team managed to develop a groundbreaking software.",
#      "Although there were initial difficulties, the team succeeded in creating an innovative program."),
#
#     # Приклад 4
#     ("In the wake of recent technological advances, traditional jobs are being replaced by automation.",
#      "With the rise of new technologies, many traditional employment opportunities are being taken over by machines.")
# ]
#
#
# # Функція для обчислення схожості між текстами
# def calculate_similarity(text1, text2):
#     # Генерація векторів для кожного тексту
#     embeddings1 = model.encode([text1])
#     embeddings2 = model.encode([text2])
#
#     # Обчислення косинусної схожості між векторами
#     similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
#     return similarity
#
#
# # Порівняння кожної пари текстів
# for i, (text1, text2) in enumerate(texts, start=1):
#     similarity = calculate_similarity(text1, text2)
#     print(f"Similarity between Example {i}: {similarity:.4f}")

# from sentence_transformers import SentenceTransformer, util
#
# # Завантажуємо попередньо натреновану модель SBERT
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# # model = SentenceTransformer('paraphrase-mpnet-base-v2')
# # model = SentenceTransformer('paraphrase-distilroberta-base-v1')
#
# # Тексти для порівняння
# text1 = "Dramatica for Screenwriters by Armando Saldana Mora is a must for any writer..."
# text2 = "I want to read dramatica book for screenwriter"
#
# # Отримуємо вектори для кожного тексту
# embedding1 = model.encode(text1, convert_to_tensor=True)
# embedding2 = model.encode(text2, convert_to_tensor=True)
#
# # Рахуємо косинусну схожість між векторами
# similarity = util.pytorch_cos_sim(embedding1, embedding2)
#
# print(f"Cosine similarity: {similarity.item()}")

import spacy
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Завантажуємо модель SpaCy для лематизації
nlp = spacy.load("en_core_web_sm")

# Завантажуємо список стоп-слів з NLTK
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Ініціалізація стемінг-процесора
stemmer = PorterStemmer()


# Функція для очищення тексту
def preprocess(text):
    # 1. Перетворення в нижній регістр
    text = text.lower()

    # 2. Видалення пунктуації та спецсимволів
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Видалення цифр
    text = re.sub(r'\d+', '', text)

    # 4. Токенізація
    tokens = word_tokenize(text)

    # 5. Видалення стоп-слів
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Лематизація
    doc = nlp(" ".join(tokens))  # Обробка тексту через SpaCy для лематизації
    lemmatized_tokens = [token.lemma_ for token in doc]

    # 7. Стемінг (додатково для порівняння з лематизацією)
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    # Об'єднуємо очищені слова назад у рядок
    cleaned_text = " ".join(stemmed_tokens)

    return cleaned_text


# Тексти для обробки
text1 = "A Walk in the Woods: a Play in Two Acts is an intellectually meaty and fast play."
text2 = "I want to read fun books."

# Очищаємо тексти
text1_cleaned = preprocess(text1)
text2_cleaned = preprocess(text2)

print(f"Cleaned Text 1: {text1_cleaned}")
print(f"Cleaned Text 2: {text2_cleaned}")
