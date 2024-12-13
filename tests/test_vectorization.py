# # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # #
# # # # # Створення TF-IDF векторизатора з біграмами (ngram_range=(1, 2))
# # # # vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
# # # #
# # # # # Припустимо, що у вас є список описів книг
# # # # descriptions = [
# # # #     "A science fiction novel about space exploration and artificial intelligence.",
# # # #     "A deep dive into the world of machine learning and neural networks.",
# # # #     "A thrilling mystery novel with twists and turns."
# # # # ]
# # # #
# # # # # Підготовка векторизованих даних
# # # # X = vectorizer.fit_transform(descriptions)
# # # #
# # # # # Виведення слів та їх TF-IDF значень
# # # # words = vectorizer.get_feature_names_out()
# # # # print(words)
# # #
# # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # #
# # # # # Використовуємо біграми (ngram_range=(1, 2) означає від 1 до 2 слів)
# # # # vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# # # #
# # # # # Тренуємо векторизатор на вашому наборі текстів
# # # # book_texts = ["I want a drama book"]#, "I love fiction", "This is a drama book", "I want to read about drama"]
# # # # book_vectors = vectorizer.fit_transform(book_texts)
# # # #
# # # # # Перевірка створених n-грам
# # # # print(vectorizer.get_feature_names_out())
# # #
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # #
# # # # Векторизатор з біграмами
# # # vectorizer = TfidfVectorizer(ngram_range=(2, 2))
# # #
# # # # Тренуємо на описах книг
# # # book_descriptions = ["I want a drama book", "I love science fiction drama", "This is a romantic comedy", "I enjoy mystery novels"]
# # # book_vectors = vectorizer.fit_transform(book_descriptions)
# # #
# # # # Запит користувача
# # # user_query = ["I want drama book"]
# # #
# # # # Трансформуємо запит в той самий векторний простір
# # # query_vector = vectorizer.transform(user_query)
# # #
# # # # Тепер можна порівнювати вектори запиту і описів книг
# # # from sklearn.metrics.pairwise import cosine_similarity
# # #
# # # similarity_scores = cosine_similarity(query_vector, book_vectors)
# # # print(similarity_scores)
# #
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity
# #
# # # Завантажуємо попередньо навчену модель для отримання векторних представлень речень
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# #
# # # Опис книг і запит користувача
# book_descriptions = [
#     "I want a drama book",
#     "I love science fiction drama",
#     "This is a romantic comedy",
#     "I enjoy mystery novels"
# ]
#
# # import nltk
# # from nltk.tokenize import word_tokenize
# # from nltk.stem import WordNetLemmatizer
# #
# # nltk.download('punkt')
# # nltk.download('wordnet')
# #
# # lemmatizer = WordNetLemmatizer()
# #
# # def preprocess_text(text):
# #     if isinstance(text, str):
# #         tokens = word_tokenize(text.lower())
# #         lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
# #         return ' '.join(lemmatized)
# #     return ''
# #
# # processed_descriptions = [preprocess_text(d) for d in book_descriptions ]
# #
# # print(processed_descriptions)
# #
# user_query = ["I want drama book"]
# # #
# # # # Отримуємо векторні представлення книг та запиту користувача
# # # book_vectors = model.encode(book_descriptions)
# # # query_vector = model.encode(user_query)
# # #
# # # # Порівнюємо подібність за допомогою косинусної подібності
# # # similarity_scores = cosine_similarity(query_vector, book_vectors)
# # #
# # # # Виводимо подібність
# # # print(similarity_scores)
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Параметри для векторизації
# vectorizer = TfidfVectorizer(
#     ngram_range=(1, 2),  # Можна спробувати 1, 3 або 1, 4 для більш точного пошуку
#     stop_words='english',  # Видалення англійських стоп-слів
#     max_df=0.85,  # Ігнорувати слова, що зустрічаються у більше ніж 85% текстів
#     min_df=2      # Ігнорувати слова, які зустрічаються в менше ніж 2 документах
# )
#
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#
# # Попередня обробка: зменшення до нижнього регістру, видалення стоп-слів
# cleaned_descriptions = [' '.join([word.lower() for word in description.split() if word.lower() not in ENGLISH_STOP_WORDS]) for description in book_descriptions]
# cleaned_query = [' '.join([word.lower() for word in user_query[0].split() if word.lower() not in ENGLISH_STOP_WORDS])]
#
# def clear_text(text):
#     return [' '.join([word.lower() for word in description.split() if word.lower() not in ENGLISH_STOP_WORDS]) for description in text]
#
# print(cleaned_descriptions)
# print(cleaned_query)
#
# text_test = "This is only for Julie Strain fans. It's a collection of her photos -- about 80 pages worth with a nice section of paintings by Olivia.If you're looking for heavy literary content, this isn't the place to find it -- there's only about 2 pages with text and everything else is photos.Bottom line: if you only want one book, the Six Foot One ... is probably a better choice, however, if you like Julie like I like Julie, you won't go wrong on this one either."
#
# print(text_test)
# print(clear_text(text_test))
# # Векторизація
# book_vectors = vectorizer.fit_transform(cleaned_descriptions)
# query_vector = vectorizer.transform(cleaned_query)
#
# # Косинусна подібність
# similarity_scores = cosine_similarity(query_vector, book_vectors)
# print(similarity_scores)
# from transformers import BertTokenizer, BertModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Завантажуємо модель BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
#
# def get_bert_embeddings(texts):
#     """
#     Отримуємо векторні представлення для текстів через модель BERT.
#     """
#     embeddings = []
#     for text in texts:
#         # Токенізуємо текст
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         # Отримуємо ембеддинги
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # Бере середнє значення ембеддингу по всіх токенах
#             embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
#
#     return embeddings
#
#
# def search_books(book_descriptions, user_query):
#     """
#     Пошук книг, схожих на запит, з використанням BERT для векторизації.
#     """
#     # Отримуємо векторні представлення для описів книг та запиту
#     book_embeddings = get_bert_embeddings(book_descriptions)
#     query_embedding = get_bert_embeddings(user_query)
#
#     # Обчислюємо косинусну подібність між запитом та книгами
#     similarity_scores = cosine_similarity(query_embedding, book_embeddings)
#
#     # Сортуємо книги за зменшенням схожості
#     sorted_books = sorted(
#         [(book_descriptions[i], similarity_scores[0][i]) for i in range(len(book_descriptions))],
#         key=lambda x: x[1], reverse=True
#     )
#
#     return sorted_books

#
# # Приклад використання
# book_descriptions = [
#     "I want a drama book",
#     "I love science fiction drama",
#     "This is a romantic comedy",
#     "I enjoy mystery novels"
# ]
#
# user_query = ["I want drama book"]
#
# import spacy
#
# nlp = spacy.load("en_core_web_sm")
#
# def lemmatize_text(text):
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
#
# print(lemmatize_text(user_query))
# # Виклик функції для пошуку
# relevant_books = search_books(book_descriptions, user_query)
#
# print("Relevant books found:")
# for book, score in relevant_books:
#     print(f"Book: '{book}', Similarity: {score:.4f}")

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Підготовка текстів
# user_description = ["I love science fiction books, especially those with time travel and complex plots."]
# book_descriptions = [
#     "A thrilling adventure with time travel and mysteries.",
#     "A heartwarming romantic story of two people finding each other.",
#     "A science fiction novel exploring futuristic technology and artificial intelligence."
# ]
#
# # Векторизація текстів
# vectorizer = TfidfVectorizer(stop_words='english')
# all_texts = [user_description] + book_descriptions  # об'єднуємо опис користувача з описами книг
# tfidf_matrix = vectorizer.fit_transform(all_texts)
#
# # Обчислення косинусної подібності між описом користувача і книгами
# cos_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
#
# # Показуємо схожість
# for idx, sim in enumerate(cos_similarities[0]):
#     print(f"Book {idx+1}: {sim:.4f}")
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Крок 1: Попередня обробка тексту
def preprocess_text(text):
    """
    Функція для попередньої обробки тексту:
    - перетворення в нижній регістр
    - видалення небажаних символів
    - лемматизація
    - видалення стоп-слів
    """
    # Перетворення тексту в нижній регістр
    text = text.lower()

    # Видалення всіх небажаних символів (окрім літер і пробілів)
    text = re.sub(r'[^a-z\s]', '', text)

    # Лемматизація і видалення стоп-слів
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(filtered_words)

# Крок 2: Ініціалізація моделей
# Модель трансформера для векторизації текстів
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Тестові дані
user_description = ["I love fantasy books with magic and adventures."]
book_descriptions = [
    "A young wizard embarks on an adventure to save the world from an ancient curse.",
    "A detective investigates a series of mysterious deaths in a small town.",
    "A spaceship travels to distant planets to explore new civilizations.",
    "This is a story about a boy who likes sucking big fat black cocks",
    "a b c d"
]

# Крок 3: Попередня обробка текстів (для ТФІДФ і трансформерів)
user_description_processed = preprocess_text(user_description[0])
book_descriptions_processed = [preprocess_text(desc) for desc in book_descriptions]

print(user_description_processed)
print(book_descriptions_processed)

# Крок 4: Векторизація через трансформер
user_vector = model.encode([user_description_processed])  # Вектор для користувача
book_vectors = model.encode(book_descriptions_processed)  # Вектори для книг

# Крок 5: Векторизація через TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

# Комбінуємо всі тексти в один список для TF-IDF
combined_descriptions = book_descriptions + [user_description[0]]
combined_vectors = vectorizer.fit_transform(combined_descriptions)

# Витягуємо вектор для користувача і для книг з результату TF-IDF
user_vector_tfidf = combined_vectors[-1]  # Вектор для користувача
book_vectors_tfidf = combined_vectors[:-1]  # Вектори для книг

# Крок 6: Обчислення косинусної подібності
cos_similarities = cosine_similarity(user_vector, book_vectors)  # Для трансформера
print(cos_similarities)
cos_similarities_tfidf = cosine_similarity(user_vector_tfidf, book_vectors_tfidf)  # Для TF-IDF

# Крок 7: Виведення результатів
print("Результати косинусної подібності (SentenceTransformer):")
for i, score in enumerate(cos_similarities[0]):
    print(f"Book {i+1}: {score:.4f}")

print("\nРезультати косинусної подібності (TF-IDF):")
for i, score in enumerate(cos_similarities_tfidf[0]):
    print(f"Book {i+1}: {score:.4f}")

