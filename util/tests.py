from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline

# import spacy
# import nltk
# import re
# import string
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
#
# # Завантажуємо модель SpaCy для лематизації
# nlp = spacy.load("en_core_web_sm")
#
# # Завантажуємо список стоп-слів з NLTK
# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = set(stopwords.words('english'))
#
# # Ініціалізація стемінг-процесора
# stemmer = PorterStemmer()
#
#
# # Функція для очищення тексту
# def preprocess(text):
#     # 1. Перетворення в нижній регістр
#     text = text.lower()
#
#     # 2. Видалення пунктуації та спецсимволів
#     text = re.sub(r'[^\w\s]', '', text)
#
#     # 3. Видалення цифр
#     text = re.sub(r'\d+', '', text)
#
#     # 4. Токенізація
#     tokens = word_tokenize(text)
#
#     # 5. Видалення стоп-слів
#     tokens = [word for word in tokens if word not in stop_words]
#
#     # 6. Лематизація
#     doc = nlp(" ".join(tokens))  # Обробка тексту через SpaCy для лематизації
#     lemmatized_tokens = [token.lemma_ for token in doc]
#
#     # 7. Стемінг (додатково для порівняння з лематизацією)
#     stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
#
#     # Об'єднуємо очищені слова назад у рядок
#     cleaned_text = " ".join(stemmed_tokens)
#
#     return cleaned_text



# Завантажуємо дані з CSV
# books_data = pd.read_csv('books_data.csv')  # Датасет з описами книг
# books_rating = pd.read_csv('books_rating.csv')  # Датасет з відгуками
