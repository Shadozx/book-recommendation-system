# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # from flask import Flask, request, jsonify
# # #
# # # app = Flask(__name__)
# # #
# # # # Завантажуємо дані з ваших CSV файлів
# # # books_data = pd.read_csv('./datasets/book_reviews/books_data_preview.txt')
# # # books_rating = pd.read_csv('./datasets/book_reviews/books_rating_preview.txt')
# # #
# # # # Обробка пропусків
# # # books_rating['review/text'] = books_rating['review/text'].fillna('')
# # # books_rating['review/summary'] = books_rating['review/summary'].fillna('')
# # #
# # # # Створення матриці оцінок користувачів на книги
# # # user_book_matrix = books_rating.pivot_table(index='User_id', columns='Title', values='review/score')
# # #
# # # # Заповнюємо пропуски нулями для обчислення косинусної схожості
# # # user_book_matrix_filled = user_book_matrix.fillna(0)
# # #
# # # # Обчислюємо косинусну схожість між користувачами
# # # user_similarity = cosine_similarity(user_book_matrix_filled)
# # #
# # # # Створюємо DataFrame для схожості між користувачами
# # # user_similarity_df = pd.DataFrame(user_similarity, index=user_book_matrix.index, columns=user_book_matrix.index)
# # #
# # #
# # # # Функція для знаходження схожих користувачів
# # # def find_similar_users(user_id, top_n=5):
# # #     if user_id not in user_similarity_df.index:
# # #         return []
# # #
# # #     similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:top_n + 1]
# # #     return similar_users.index.tolist()
# # #
# # #
# # # # Функція для генерації рекомендацій
# # # def recommend_books(user_id, top_n=5):
# # #     # Знаходимо схожих користувачів
# # #     similar_users = find_similar_users(user_id, top_n)
# # #
# # #     # Вибираємо книги, які оцінили схожі користувачі, але ще не оцінив поточний користувач
# # #     user_books = set(user_book_matrix.loc[user_id].dropna().index)
# # #     similar_users_books = user_book_matrix.loc[similar_users].apply(lambda x: set(x.dropna().index), axis=1)
# # #
# # #     recommended_books = set()
# # #     for books in similar_users_books:
# # #         recommended_books.update(books)
# # #
# # #     # Вибираємо тільки ті книги, які ще не оцінив поточний користувач
# # #     recommended_books -= user_books
# # #
# # #     # Повертаємо топ-n книг для рекомендацій
# # #     return list(recommended_books)[:top_n]
# # #
# # #
# # # @app.route('/recommendations', methods=['POST'])
# # # def get_recommendations():
# # #     data = request.get_json()
# # #     user_id = data.get('user_id')
# # #     top_n = data.get('top_n', 5)
# # #
# # #     if not user_id:
# # #         return jsonify({'error': 'user_id is required'}), 400
# # #
# # #     recommendations = recommend_books(user_id, top_n)
# # #
# # #     return jsonify({
# # #         'user_id': user_id,
# # #         'recommendations': recommendations
# # #     })
# # #
# # #
# # # if __name__ == '__main__':
# # #     app.run(debug=True, port=9000)
# #
# # import pandas as pd
# # from flask import Flask, request, jsonify
# #
# # app = Flask(__name__)
# #
# # # Завантажуємо дані з ваших CSV файлів
# # books_data = pd.read_csv('./datasets/book_reviews/books_data_preview.txt')
# # books_rating = pd.read_csv('./datasets/book_reviews/books_rating_preview.txt')
# #
# # # Обробка пропусків
# # books_rating['review/text'] = books_rating['review/text'].fillna('')
# # books_rating['review/summary'] = books_rating['review/summary'].fillna('')
# #
# # # Створення матриці оцінок користувачів на книги
# # user_book_matrix = books_rating.pivot_table(index='User_id', columns='Title', values='review/score')
# #
# # print(user_book_matrix)
# #
# # # Функція для генерації спільних рекомендацій між двома конкретними користувачами
# # def recommend_books_for_two_users(user_id1, user_id2, top_n=5):
# #     # Перевіряємо, чи обидва користувачі існують в матриці
# #     if user_id1 not in user_book_matrix.index or user_id2 not in user_book_matrix.index:
# #         return []
# #
# #     # Оцінки обох користувачів
# #     user1_ratings = user_book_matrix.loc[user_id1]
# #     user2_ratings = user_book_matrix.loc[user_id2]
# #
# #     # Знайдемо книги, які оцінив користувач 1, але не оцінив користувач 2
# #     books_user1_not_rated_by_user2 = user1_ratings[user2_ratings.isna()].index
# #
# #     # Повертаємо топ-n книг, які оцінив користувач 1, але не оцінив користувач 2
# #     recommended_books = books_user1_not_rated_by_user2[:top_n]
# #
# #     return list(recommended_books)
# #
# # @app.route('/recommendations/compare', methods=['POST'])
# # def get_recommendations_for_two_users():
# #     # Отримуємо JSON з тіла запиту
# #     data = request.get_json()
# #
# #     user_id1 = data.get('user_id1')
# #     user_id2 = data.get('user_id2')
# #     top_n = data.get('top_n', 5)
# #
# #     # Перевірка наявності id користувачів
# #     if not user_id1 or not user_id2:
# #         return jsonify({'error': 'Both user_id1 and user_id2 are required'}), 400
# #
# #     # Генерація рекомендацій для двох користувачів
# #     recommendations = recommend_books_for_two_users(user_id1, user_id2, top_n)
# #
# #     return jsonify({
# #         'user_id1': user_id1,
# #         'user_id2': user_id2,
# #         'recommendations': recommendations
# #     })
# #
# # if __name__ == '__main__':
# #     app.run(debug=True, port=9000)
#
# import pandas as pd
# from flask import Flask, request, jsonify
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# app = Flask(__name__)
#
# # Завантажуємо дані з ваших CSV файлів
# books_data = pd.read_csv('./datasets/book_reviews/books_data_preview.txt')
# books_rating = pd.read_csv('./datasets/book_reviews/books_rating_preview.txt')
#
# # Обробка пропусків
# books_rating['review/text'] = books_rating['review/text'].fillna('')
# books_rating['review/summary'] = books_rating['review/summary'].fillna('')
#
# # Створення матриці оцінок користувачів на книги
# user_book_matrix = books_rating.pivot_table(index='User_id', columns='Title', values='review/score')
#
# print(user_book_matrix)
#
# # Функція для генерації спільних рекомендацій на основі оцінок двох користувачів
# # def common_recommendations(user_id1, user_id2, top_n=5):
# #     # Перевіряємо, чи обидва користувачі існують в матриці
# #     if user_id1 not in user_book_matrix.index or user_id2 not in user_book_matrix.index:
# #         return []
# #
# #     # Оцінки обох користувачів
# #     user1_ratings = user_book_matrix.loc[user_id1]
# #     user2_ratings = user_book_matrix.loc[user_id2]
# #
# #     # Знайдемо спільні книги (ті, які оцінили обидва користувачі)
# #     common_books = user1_ratings.dropna().index.intersection(user2_ratings.dropna().index)
# #
# #     if len(common_books) == 0:
# #         return [], 0
# #
# #     # Створюємо масив для зберігання оцінок спільних книг для кожного користувача
# #     user1_ratings_common = user1_ratings[common_books]
# #     user2_ratings_common = user2_ratings[common_books]
# #
# #     # Обчислюємо косинусну схожість між оцінками користувачів
# #     similarity_score = cosine_similarity([user1_ratings_common], [user2_ratings_common])[0][0]
# #
# #     # Повертаємо список книг, які оцінив один користувач, але не оцінив інший
# #     user1_unrated_books = user1_ratings[user2_ratings.isna()].index
# #     recommendations = user1_unrated_books[:top_n]
# #
# #     return recommendations, similarity_score
#
# def common_recommendations(user_id1, user_id2, top_n=5):
#     # Отримуємо оцінки для обох користувачів
#     user1_ratings = books_rating[books_rating['user_id'] == user_id1]
#     user2_ratings = books_rating[books_rating['user_id'] == user_id2]
#
#     # Перевіряємо, чи є оцінки для обох користувачів
#     if user1_ratings['review/score'].isna().all() or user2_ratings['review/score'].isna().all():
#         return [], 0  # Якщо у одного з користувачів немає оцінок, повертаємо порожній список і схожість 0
#
#     # Фільтруємо тільки ті книги, для яких є оцінки в обох користувачів
#     common_books = pd.merge(user1_ratings, user2_ratings, on='book_id')
#
#     # Якщо немає спільних книг, неможливо розрахувати подібність
#     if common_books.empty:
#         return [], 0
#
#     # Логіка для розрахунку подібності (наприклад, кореляція Пірсона або косинусна подібність)
#     similarity_score = calculate_similarity(common_books)
#
#     # Отримуємо топ рекомендацій
#     recommendations = get_top_recommendations(user_id1, user_id2, top_n)
#
#     return recommendations, similarity_score
#
#
# # Приклад для розрахунку подібності, наприклад, за допомогою косинусної подібності:
# def calculate_similarity(common_books):
#     # Замість цього можна застосувати конкретний алгоритм для обчислення подібності
#     user1_ratings = common_books['review/score_x']
#     user2_ratings = common_books['review/score_y']
#
#     # Наприклад, косинусна подібність (можна замінити на кореляцію або інший метод)
#     similarity = cosine_similarity(user1_ratings.values.reshape(1, -1), user2_ratings.values.reshape(1, -1))[0][0]
#
#     return similarity
#
# @app.route('/recommendations/common', methods=['POST'])
# def get_common_recommendations():
#     # Отримуємо JSON з тіла запиту
#     print(request.json)
#     data = request.get_json()
#
#     user_id1 = data.get('user_id1')
#     user_id2 = data.get('user_id2')
#     top_n = data.get('top_n', 5)
#     print(user_id1, user_id2, top_n)
#     # Перевірка наявності id користувачів
#     if not user_id1 or not user_id2:
#         return jsonify({'error': 'Both user_id1 and user_id2 are required'}), 400
#
#     print('before common recommendations')
#     # Генерація спільних рекомендацій для двох користувачів
#     recommendations, similarity_score = common_recommendations(user_id1, user_id2, top_n)
#
#     print('after common recommendations')
#     return jsonify({
#         'user_id1': user_id1,
#         'user_id2': user_id2,
#         'similarity_score': similarity_score,
#         'recommendations': list(recommendations)
#     })
#
# @app.route('/welcome')
# def welcome():
#     return jsonify({'Welcome': 'Welcome'})
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

#
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# # Загрузка даних оцінок (припускаємо, що це DataFrame з колонками: user_id, book_id, review/score)
# # books_rating = pd.read_csv('your_dataset.csv')  # це ваш датасет
#
# # Приклад датасету для демонстрації
# data = {
#     'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
#     'book_id': [101, 102, 103, 101, 102, 104, 102, 103, 105],
#     'review/score': [5, 3, np.nan, 4, 2, 5, 5, 4, np.nan]
# }
# books_rating = pd.DataFrame(data)
#
#
# def common_recommendations(user_id1, user_id2, top_n=5):
#     # Отримуємо оцінки для обох користувачів
#     user1_ratings = books_rating[books_rating['user_id'] == user_id1]
#     user2_ratings = books_rating[books_rating['user_id'] == user_id2]
#
#     # Перевірка, чи є оцінки для обох користувачів
#     if user1_ratings['review/score'].isna().all() or user2_ratings['review/score'].isna().all():
#         return [], 0  # Якщо у одного з користувачів немає оцінок, повертаємо порожній список і схожість 0
#
#     # Фільтруємо тільки ті книги, для яких є оцінки в обох користувачів
#     common_books = pd.merge(user1_ratings, user2_ratings, on='book_id')
#
#     # Якщо немає спільних книг, неможливо розрахувати подібність
#     if common_books.empty:
#         return [], 0
#
#     # Логіка для розрахунку подібності (наприклад, кореляція Пірсона або косинусна подібність)
#     similarity_score = calculate_similarity(common_books)
#
#     # Отримуємо топ рекомендацій для користувача user_id1
#     recommendations = get_top_recommendations(user_id1, user_id2, top_n)
#
#     return recommendations, similarity_score
#
#
# # Приклад для розрахунку подібності, наприклад, за допомогою косинусної подібності:
# def calculate_similarity(common_books):
#     # Замість цього можна застосувати конкретний алгоритм для обчислення подібності
#     user1_ratings = common_books['review/score_x']
#     user2_ratings = common_books['review/score_y']
#
#     # Наприклад, косинусна подібність
#     similarity = cosine_similarity(user1_ratings.values.reshape(1, -1), user2_ratings.values.reshape(1, -1))[0][0]
#
#     return similarity
#
#
# def get_top_recommendations(user_id1, user_id2, top_n):
#     # Функція для отримання рекомендацій (можна використати інші методи генерації рекомендацій)
#     # Спершу отримуємо книги, які оцінив перший користувач, але не оцінив другий
#     user1_ratings = books_rating[books_rating['user_id'] == user_id1]
#     user2_ratings = books_rating[books_rating['user_id'] == user_id2]
#
#     # Залишаємо тільки ті книги, які не оцінював user_id2
#     user1_books = user1_ratings[~user1_ratings['book_id'].isin(user2_ratings['book_id'])]
#
#     # Вибираємо топ_n книг для рекомендацій, це можуть бути книги з найвищими оцінками
#     top_recommendations = user1_books.nlargest(top_n, 'review/score')[['book_id', 'review/score']]
#
#     return top_recommendations
#
#
# # Тестування функції:
# user_id1 = 1
# user_id2 = 2
# top_n = 3
#
# recommendations, similarity_score = common_recommendations(user_id1, user_id2, top_n)
#
# print("Рекомендації:")
# print(recommendations)
# print(f"Схожість між користувачами: {similarity_score}")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify

test_folder = '../datasets/book_reviews/test'

# Завантажуємо дані (користувачі, оцінки, книги)
books_data = pd.read_csv(test_folder + '/book_reviews/books_data_250_test.csv')
print('books data was downloaded')
# books_data = pd.read_csv('./datasets/book_reviews/books_data_preview.txt')
# books_rating = pd.read_csv('./datasets/book_reviews/books_rating_preview.txt')
books_rating = pd.read_csv(test_folder + '/book_reviews/books_rating_books_250_test.csv')
print('books rating was downloaded')

# user id A0015610VMNR0JC9XVL1     6
# AZZZT14MS21I6            5


# Перевіряємо структуру даних
# print(books_data.head())
# print(books_rating.head())

# Обробка пропусків: заповнюємо пропуски в оцінках нулями або середнім значенням
books_rating['review/score'] = books_rating['review/score'].fillna(0)

# Створимо таблицю оцінок користувачів для кожної книги
user_ratings = books_rating.pivot_table(index='User_id', columns='Title', values='review/score').fillna(0)


# Функція для обчислення косинусної подібності між користувачами
def calculate_user_similarity(user_ratings):
    # Обчислюємо косинусну подібність між користувачами
    similarity_matrix = cosine_similarity(user_ratings)
    return similarity_matrix


# Функція для отримання рекомендацій для конкретного користувача на основі спільної подібності з іншим користувачем
def get_common_recommendations(user_id1, user_id2, top_n=5):
    # Отримуємо косинусну подібність між усіма користувачами
    similarity_matrix = calculate_user_similarity(user_ratings)

    # Індекси користувачів
    user_idx1 = user_ratings.index.get_loc(user_id1)
    user_idx2 = user_ratings.index.get_loc(user_id2)

    # Подібність між двома користувачами
    similarity_score = similarity_matrix[user_idx1, user_idx2]
    print(similarity_score)
    # Якщо подібність низька, то рекомендації не даються
    if similarity_score < 0.2:
        return [], similarity_score

    # Отримуємо всі книги, які оцінив користувач user_id2
    user2_ratings = user_ratings.loc[user_id2]
    books_rated_by_user2 = user2_ratings[user2_ratings > 0].index.tolist()

    # Фільтруємо книги, які не оцінив користувач user_id1
    user1_ratings = user_ratings.loc[user_id1]
    books_to_recommend = [book_title for book_title in books_rated_by_user2 if user1_ratings[book_title] == 0]

    # Якщо не знайдено книг для рекомендацій, повертаємо порожній список
    if not books_to_recommend:
        return [], similarity_score

    # Отримуємо інформацію про ці книги
    recommended_books = books_data[books_data['Title'].isin(books_to_recommend)][
        ['Title', 'description', 'authors', 'publisher', 'publishedDate', 'categories']]

    return recommended_books.to_dict(orient='records'), similarity_score


# Flask API

app = Flask(__name__)


# @app.route('/recommendations/common', methods=['POST'])
# def get_common_recommendations_api():
#     # Отримуємо дані з запиту
#     data = request.get_json()
#
#     # Збираємо параметри
#     user_id1 = data.get('user_id1')
#     user_id2 = data.get('user_id2')
#     top_n = data.get('top_n', 5)
#
#     if not user_id1 or not user_id2:
#         return jsonify({'error': 'user_id1 and user_id2 are required'}), 400
#
#     # Отримуємо рекомендації
#     recommended_books, similarity_score = get_common_recommendations(user_id1, user_id2, top_n)
#
#     if not recommended_books:
#         return jsonify({'message': 'No common recommendations found', 'similarity_score': similarity_score})
#
#     # Повертаємо список рекомендацій
#     return jsonify({
#         'recommendations': recommended_books,
#         'similarity_score': similarity_score
#     })
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

user_1 = ''