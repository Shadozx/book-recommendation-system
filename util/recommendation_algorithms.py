import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time
from sentence_transformers import SentenceTransformer
from util.sentiment_analysis import *

# Завантажуємо модель для векторизації
modelText = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


class SparseRecommender:
    def __init__(self, sparse_matrix, user_mapping, book_mapping):
        self.matrix = sparse_matrix
        self.user_mapping = user_mapping  # Словник {user_id: index}
        self.reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        self.book_mapping = book_mapping  # Словник {book_title: index}
        self.reverse_book_mapping = {v: k for k, v in book_mapping.items()}

    def user_based_recommendations(self, user_id, top_n=5):
        # Перевірка, чи існує користувач
        if user_id not in self.user_mapping:
            return []

        # print(self.user_mapping)
        # print(self.user_mapping[user_id])

        # Отримання numeric індексу
        user_numeric_index = self.user_mapping[user_id]

        # Косинусна подібність між користувачами
        user_similarities = cosine_similarity(
            self.matrix[user_numeric_index:user_numeric_index + 1],
            self.matrix
        )



        # Знаходження найближчих сусідів
        similar_user_indices = user_similarities[0].argsort()[::-1][1:6]

        # Агрегація рекомендацій
        recommendations_numeric = np.sum(
            self.matrix[similar_user_indices],
            axis=0
        ).A1

        # Перетворення numeric індексів книг назад у назви
        recommended_book_indices = recommendations_numeric.argsort()[::-1][:top_n]
        recommended_books = [
            self.reverse_book_mapping[idx]
            for idx in recommended_book_indices
        ]

        return recommended_books

    def item_based_recommendations(self, book_title, top_n=5):
        # Перевірка, чи існує книга
        if book_title not in self.book_mapping:
            return []

        # Отримання numeric індексу книги
        book_numeric_index = self.book_mapping[book_title]

        # Косинусна подібність між книгами
        item_similarities = cosine_similarity(
            self.matrix[:, book_numeric_index:book_numeric_index + 1].T,
            self.matrix.T
        )

        # Знаходження найближчих за подібністю книг
        similar_book_indices = item_similarities[0].argsort()[::-1][1:top_n + 1]

        # Перетворення numeric індексів книг назад у назви
        similar_books = [
            self.reverse_book_mapping[idx]
            for idx in similar_book_indices
        ]

        return similar_books


# Приклад створення
def create_recommender(books_rating):
    # Створення маппінгів
    user_mapping = {user: idx for idx, user in enumerate(books_rating['User_id'].unique())}
    book_mapping = {book: idx for idx, book in enumerate(books_rating['Title'].unique())}

    # Створення sparse matrix
    rows = [user_mapping[row['User_id']] for _, row in books_rating.iterrows()]
    cols = [book_mapping[row['Title']] for _, row in books_rating.iterrows()]
    values = books_rating['review/score'].values

    sparse_matrix = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_mapping), len(book_mapping))
    )

    return SparseRecommender(sparse_matrix, user_mapping, book_mapping)

# modelText = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
# modelText = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# modelText = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# modelText = SentenceTransformer('all-MiniLM-L6-v2')
vectorizer = TfidfVectorizer(stop_words='english')

def calculate_similarity(texts, text):
    all_texts = [text] + texts
    print('starting embeddings')
    embeddings = modelText.encode(all_texts)
    print('start calculating')
    cosine_sim = cosine_similarity([embeddings[0]], embeddings[1:])
    print('end calculating')
    return cosine_sim[0]
def calculate_vector_similarity(book_vectors, user_vector):
    similarities = cosine_similarity(user_vector, book_vectors)

    return similarities.argsort()

# Функція для обчислення схожості між текстами
# def calculate_similarity(texts, text):
#     # Генерація векторів для кожного тексту
#     # embeddings1 = modelText.encode(text1, convert_to_tensor=True)
#     # embeddings2 = modelText.encode(text2, convert_to_tensor=True)
#     #
#     # # Обчислення косинусної схожості між векторами
#     # # similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
#     # return util.pytorch_cos_sim(embeddings1, embeddings2).item()
#     all_texts = [text] + texts
#     embeddings = modelText.encode(all_texts)
#     cosine_sim = cosine_similarity([embeddings[0]], embeddings[1:])
#
#     return cosine_sim[0]


def find_best_matching_books(user_description, book_texts, books_data, top_n=5):
    # user_description = preprocess(user_description)

    book_vectors = vectorizer.fit_transform(book_texts['final_combined'].tolist())

    user_vector = vectorizer.transform([user_description])



    # start_time = time.time()
    # book_texts['similarity'] = book_texts['final_combined'].apply(calculate_similarity, text2=user_description).tolist()
    # book_texts['similarity'] = calculate_similarity(book_texts['final_combined'].tolist(), user_description)
    # similarities = calculate_vector_similarity(book_vectors, user_vector)
    similarities = cosine_similarity(user_vector, book_vectors)
    similar_indices = similarities.argsort()[0][top_n:][::-1]
    # print(similarities)
    # print(similar_indices)
    # book_texts['similarity'] = similarities
    # # df_grouped = book_texts_test.groupby('Title', as_index=False).agg({'similarity': 'sum'})
    # # 1. Групуємо за назвою і вибираємо максимальне значення similarity
    # print('here')
    # df_max_similarity = book_texts.groupby('Title', as_index=False)['similarity'].max()
    # print('here')
    # print(df_max_similarity)
    #
    # # 2. Фільтруємо рядки, де similarity > 0.5
    # df_filtered = df_max_similarity.sort_values(by='similarity', ascending=False)
    # print('here filtering')
    # print(df_filtered)
    #
    # end_time = time.time()
    #
    # print(f'Time of filtering - {end_time - start_time} seconds')
    # top_books = {}
    #
    # for idx, row in df_filtered.iterrows():
    #     # Отримуємо назву книги та її similarity
    #     book_title = row['Title']
    #     score = row['similarity']
    #
    #     # Знаходимо ID книги в основному наборі даних book_texts
    #     book_id = int(books_data[books_data['Title'] == book_title].iloc[0]['book_id'])
    #
    #     # Перевіряємо на валідність ID
    #     if book_id <= 0:
    #         print(f"Неправильний ID: {book_id}, пропускаємо.")
    #         continue
    #
    #     # Додаємо дані до словника top_books
    #     if book_id not in top_books:
    #         top_books[book_id] = {
    #             "id": int(book_id),
    #             "title": book_title,
    #             "score": float(score)
    #         }
    #
    #     # Перевірка на кількість обраних книг
    #     if len(top_books) >= top_n:
    #         break
    # print(top_books)
    # return [{"id": book['id'], "title": book["title"], "score": book["score"]} for book in top_books.values()]

    print("Found books len:", len(similar_indices))

    top_books = {}

    for idx in similar_indices:
        # Якщо індекс знаходиться в межах books_data
        # print(idx)
        if idx < len(book_texts):
            # book_id = int(merged_data.iloc[idx]['id'])
            book_title = book_texts.iloc[idx]['Title']

            # book_info = books_data[books_data['id'] == id].iloc[0]

            book_id = int(books_data[books_data['Title'] == book_title].iloc[0]['book_id'])
            score = similarities[0, idx]

            if score <= 0.2:
                continue

            # print(score)

        # Перевірка валідності id
        if book_id <= 0:
            print(f"Неправильний ID: {book_id}, пропускаємо.")
            continue

        # Додаємо дані до словника
        if book_id in top_books:
            top_books[book_id]['score'] += score

        else:
            top_books[book_id] = {
                "id": book_id,
                "title": book_title,
                "score": score
            }

    print(top_books)
    # Сортуємо книги за сумарним score в порядку спадання
    sorted_books = sorted(top_books.values(), key=lambda x: x['score'], reverse=True)
    return [{"id": book['id'], "title": book["title"], "score": book["score"]} for book in sorted_books]


#     return top_books_by_emotion
def find_books_by_emotion(emotion, books_data, emotion_books_rating, top_n=5):
    sentiment_scores = []

    review_text_count_processed = 0

    for idx, row in emotion_books_rating.iterrows():
        review_emotion = row['emotion']
        review_emotion_score = row['emotion_score']

        # print(emotion_text)
        # [{'label': 'desire', 'score': 0.3667565882205963}]


        # if emotion ==
        # Пошук книг за позитивними, негативними чи нейтральними емоціями
        if emotion == review_emotion:
            book_title = row['Title']
            book_id = books_data[books_data['Title'] == book_title].iloc[0]['book_id']
            sentiment_scores.append({"id": int(book_id), "title": book_title, "score": float(review_emotion_score)})

        review_text_count_processed += 1
        if review_text_count_processed % 10 == 0:
            print("Processed review text:", review_text_count_processed)

    print(sentiment_scores)
    # # Сортуємо за оцінкою емоції і повертаємо топ N книг
    # sentiment_scores = sorted(sentiment_scores, key=lambda x: x['score'], reverse=True)
    #
    # # Виводимо результат
    # top_books_by_emotion = sentiment_scores[:top_n]
    #
    # result = [{"id":book["id"], "title": book["title"], "score": book["score"]} for book in top_books_by_emotion]
    #
    # return result

    # Перевірка, чи є в списку sentiment_scores якісь результати
    if not sentiment_scores:
        # Якщо список порожній, повертаємо порожній список або відповідь
        print(f"No books found for emotion: {emotion}")
        return []

    # Перетворюємо список в DataFrame для подальшої обробки
    sentiment_df = pd.DataFrame(sentiment_scores)

    # Групуємо за 'title' та вибираємо максимальний 'score' для кожної книги
    max_scores_df = sentiment_df.loc[sentiment_df.groupby('title')['score'].idxmax()]
    print(max_scores_df)
    # Сортуємо за оцінкою (score) і беремо top N
    top_books_by_emotion = max_scores_df.nlargest(top_n, 'score')

    print(top_books_by_emotion)

    # Перетворюємо в список для повернення результату
    return [{"id": book["id"], "title": book["title"], "score": book["score"]} for _, book in
                  top_books_by_emotion.iterrows()]


def find_books_by_emotion_and_user_dec(emotion, user_description, book_texts, books_data, emotion_books_rating, top_n=5):

    print('Book texts len:', len(book_texts))

    matching_books = find_best_matching_books(user_description, book_texts, books_data, top_n)

    print(matching_books)

    books_titles = [book["title"] for book in matching_books]

    filtered_books_rating = emotion_books_rating[emotion_books_rating['Title'].isin(books_titles)]

    print('Filtered books rating:', len(filtered_books_rating))

    emotion_books = find_books_by_emotion(emotion, books_data, filtered_books_rating, top_n)

    return emotion_books

    # emotion_book_texts = book_texts[book_texts['Title'].isin(books_titles)]

# Ось опис книги, який ми хочемо використати для пошуку
# user_description = """
# i want fun book
# """

# user_description = "I want to read Alaska"

# # Знаходимо топ 5 найбільш схожих книг на основі опису та відгуків
# top_books = find_best_matching_books(user_description, books_data, books_rating, book_vectors, top_n=5)

# # Виводимо результат
# for i, book in enumerate(top_books, 1):
#     print(f"Rank {i}: {book['Title']} (Similarity: {book['Similarity Score']:.2f})")
