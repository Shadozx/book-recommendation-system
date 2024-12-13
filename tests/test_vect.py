# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Список текстів
# texts = [
#     "Сонце сідає над горизонтом, освітлюючи хмари в розових відтінках.",
#     "Вода в річці прозора і чиста, з кришталевими відблисками сонця.",
#     "Місто розкинулося серед гір, з вузькими вулицями і старими будівлями.",
#     "Літо на морі — теплий вітер і м'який пісок на пляжі."
# ]
#
# # Опис того, яким має бути текст
# description = "Текст про природу, сонце і красиві пейзажі."
#
# # Ініціалізація векторизатора
# vectorizer = TfidfVectorizer()
#
# # Об'єднуємо опис і тексти в один список
# all_texts = [description] + texts
#
# # Векторизуємо всі тексти
# tfidf_matrix = vectorizer.fit_transform(all_texts)
#
# # Обчислюємо косинусну схожість між першим (опис) та іншими текстами
# cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
#
# # Знаходимо індекс тексту з найбільшою схожістю
# most_similar_index = cosine_similarities.argmax()
# print(cosine_similarities)
# # Виводимо найбільш схожий текст
# print(f"Найбільш схожий текст: {texts[most_similar_index]}")

# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Завантажуємо модель spaCy для лематизації
# nlp = spacy.load("uk_core_news_sm")
#
# # Функція для лематизації тексту
# def lemmatize_text(text):
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc if not token.is_stop])
#
# # Список текстів
# texts = [
#     "Сонце сідає над горизонтом, освітлюючи хмари в розових відтінках.",
#     "Вода в річці прозора і чиста, з кришталевими відблисками сонця.",
#     "Місто розкинулося серед гір, з вузькими вулицями і старими будівлями.",
#     "Літо на морі — теплий вітер і м'який пісок на пляжі."
# ]
#
# # Опис того, яким має бути текст
# description = "Текст про природу, сонце і красиві пейзажі."
#
# # Лематизуємо всі тексти
# texts_lemmatized = [lemmatize_text(text) for text in texts]
# description_lemmatized = lemmatize_text(description)
# print(texts_lemmatized)
# print(description_lemmatized)
# # Ініціалізація векторизатора
# vectorizer = TfidfVectorizer()
#
# # Об'єднуємо опис і тексти в один список
# all_texts = [description_lemmatized] + texts_lemmatized
#
# # Векторизуємо всі тексти
# tfidf_matrix = vectorizer.fit_transform(all_texts)
#
# # Обчислюємо косинусну схожість між першим (опис) та іншими текстами
# cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
#
# # Знаходимо індекс тексту з найбільшою схожістю
# most_similar_index = cosine_similarities.argmax()
#
# # Виводимо найбільш схожий текст
# print(f"Найбільш схожий текст: {texts[most_similar_index]}")
# print(f"Схожість: {cosine_similarities[0][most_similar_index]:.4f}")



from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ініціалізація моделі Sentence-BERT
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Список текстів
texts = [
    "Сонце сідає над горизонтом, освітлюючи хмари в розових відтінках.",
    "Вода в річці прозора і чиста, з кришталевими відблисками сонця.",
    "Місто розкинулося серед гір, з вузькими вулицями і старими будівлями.",
    "Літо на морі — теплий вітер і м'який пісок на пляжі. драма"
]

# Опис того, яким має бути текст
description = "Текст про природу, воду і красиві пейзажі і також драму"

# Об'єднуємо опис і тексти
all_texts = [description] + texts

# Векторизуємо тексти за допомогою Sentence-BERT
embeddings = model.encode(all_texts)

# Обчислюємо косинусну схожість між першим (опис) та іншими текстами
cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:])
print(cosine_similarities)
# Знаходимо індекс тексту з найбільшою схожістю
most_similar_index = cosine_similarities.argmax()

# Виводимо найбільш схожий текст
print(f"Найбільш схожий текст: {texts[most_similar_index]}")
print(f"Схожість: {cosine_similarities[0][most_similar_index]:.4f}")