import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict


# measure time of executing another function
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Функція {func.__name__} виконала роботу за {end_time - start_time:.4f} секунд")
        return result

    return wrapper

datasets_path = '../datasets'
book_reviews_path = datasets_path + '/book_reviews'
test_folder = book_reviews_path + '/test'

books_count = 500

# books_data = pd.read_csv(test_folder + f'/books_data_{books_count}_test.csv')
print('good')
books_rating = pd.read_csv(test_folder + f'/books_rating_books_{books_count}_test.csv')
# books_rating = pd.read_csv(book_reviews_path + '/books_rating.csv')
print('very good')


# Завантажуємо дані (користувачі, оцінки, книги)
# books_data = pd.read_csv('../datasets/book_reviews/books_data_preview.txt')
# books_data = pd.read_csv('../datasets/book_reviews/books_data_preview.txt')
# books_rating = pd.read_csv('../datasets/book_reviews/books_rating_preview.txt')
print('books rating was downloaded')

# Перевіряємо структуру даних
# print(books_data.head())
# print(books_rating.head())
user_grouped_counts = books_rating.groupby('User_id').size()

# print(books_rating['User_id'].value_counts())
# print(books_rating.groupby('User_id').size())

# from scipy.sparse import csr_matrix
#
# # Приклад створення рідкісної матриці з DataFrame
# user_item_matrix = books_rating.pivot_table(index='User_id', columns='Title', values='review/score', fill_value=0)
# sparse_matrix = csr_matrix(user_item_matrix.values)


import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


# def create_sparse_pivot(books_rating):
#     # Кодуємо користувачів та книги
#     user_encoder = {user: idx for idx, user in enumerate(books_rating['User_id'].unique())}
#     book_encoder = {book: idx for idx, book in enumerate(books_rating['Title'].unique())}
#
#     # Створюємо розріджену матрицю
#     rows = [user_encoder[row['User_id']] for _, row in books_rating.iterrows()]
#     cols = [book_encoder[row['Title']] for _, row in books_rating.iterrows()]
#     values = books_rating['review/score'].values
#
#     # print(rows)
#     # print(cols)
#
#     sparse_matrix = sp.csr_matrix(
#         (values, (rows, cols)),
#         shape=(len(user_encoder), len(book_encoder))
#     )
#
#     return sparse_matrix, user_encoder, book_encoder
#
# start_end = time.time()
# sparse_matrix, user_encoder, book_encoder = create_sparse_pivot(books_rating)
#
# print('Time ', time.time() - start_end)
#
# class SparseRecommender:
#     def __init__(self, sparse_matrix):
#         self.matrix = sparse_matrix
#
#     def user_based_recommendations(self, user_id, top_n=5):
#         # Косинусна подібність між користувачами
#         user_similarities = cosine_similarity(self.matrix[user_id:user_id + 1], self.matrix)
#
#         # Знаходження найближчих сусідів
#         similar_users = user_similarities[0].argsort()[::-1][1:6]
#
#         # Агрегація рекомендацій
#         recommendations = np.sum(
#             self.matrix[similar_users],
#             axis=0
#         ).A1  # Перетворення до одновимірного масиву
#
#         # Сортування рекомендацій
#         top_items = recommendations.argsort()[::-1][:top_n]
#
#         return top_items
#
#     def item_based_recommendations(self, item_id, top_n=5):
#         # Косинусна подібність між книгами
#         item_similarities = cosine_similarity(
#             self.matrix[:, item_id:item_id + 1].T,
#             self.matrix.T
#         )
#
#         # Знаходження найближчих за подібністю книг
#         similar_items = item_similarities[0].argsort()[::-1][1:top_n + 1]
#
#         return similar_items
#
# recommender = SparseRecommender(sparse_matrix)

import scipy.sparse as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
recommender = create_recommender(books_rating)
print(recommender.user_based_recommendations('A1D2C0WDCSHUWZ'))
print(recommender.item_based_recommendations('The Scarlet Letter A Romance'))
# user_ratings = defaultdict(dict)
#
# userCounts = 0
# start_time = time.time()
# for _, row in books_rating.iterrows():
#     user_ratings[row['User_id']][row['Title']] = row['review/score']
#
#     userCounts += 1
#
#     if userCounts % 250000 == 0:
#         print('User counts: ', userCounts, '. Time elapsed: ', time.time()-start_time)
#         # start_time = time.time()
#
#
# print('user rating was downloaded count:', len(user_ratings.keys()))
#
# # Тепер можна обчислювати схожість між користувачами:
# def calculate_cosine_similarity(user1, user2):
#     print(user_ratings[user1].keys())
#     print(user_ratings[user2].keys())
#     common_books = set(user_ratings[user1].keys()).intersection(user_ratings[user2].keys())
#
#     print(common_books)
#     if len(common_books) == 0:
#         return 0  # Якщо немає спільних книг, схожість 0
#     ratings1 = np.array([user_ratings[user1][book] for book in common_books])
#     ratings2 = np.array([user_ratings[user2][book] for book in common_books])
#     return 1 - cosine(ratings1, ratings2)  # Косинусна схожість
#
# # Приклад обчислення схожості між користувачами 1 та 2
# user1 = 'A12A08OL0TZY0W'
# user2 = 'A14OJS0VWMOSWO'
# similarity = calculate_cosine_similarity(user1, user2)
# print(f"Схожість між користувачами {user1} і {user2}: {similarity}")
#

'''
A12A08OL0TZY0W,556
A13OFOB1394G31,514
A14OJS0VWMOSWO,5795
A1D2C0WDCSHUWZ,3146
A1EKTLUL24HDG8,900
A1G37DFO8MQW0M,933
A1JTG5X4VHJV27,536
A1K1JW1C5CUSUZ,1457
A1L43KWWR05PCS,961
A1LMBM1N4EXS5W,588
A1M8PP7MLHNBQB,800
A1MC6BFHWY6WC3,780
A1N1YEMTI9DJ86,1031
A1NATT3PN24QWY,594
A1NC9AGZOBI0M1,533
A1NT7ED5TATUAM,565
A1RAUVCWYHTQI4,648
A1RECBDKHVOJMW,509
A1S3C5OFU508P3,1309
A1T17LMQABMBN5,692
A1X8VZWTOG8IS6,1804
A20EEWWSFMZ1PN,1387
A22DUZU3XVA8HA,626
'''





# при 500 книг
'''
A1D2C0WDCSHUWZ,33
A2NJO6YE954DBH,18
A1OXI0N58TMYY9,18
AFVQZQ8PW0L,14
A20EEWWSFMZ1PN,12
A13IZYHNIQGK4Q,12
ABN5K7K1TM1QA,12
A27Z2M7PNPA6LM,12
AWLFVCT9128JV,12
A2M3NCTFUGI4SR,12
AS1YKFH6RSKFB,12
AHD101501WCN1,12
A14OJS0VWMOSWO,10
A5A25SHEUX6ZK,10
'''