# датасети
# books_data(Title,description,authors,image,previewLink,publisher,publishedDate,infoLink,categories,ratingsCount)
# books_rating(Id,Title,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text)

# import os
# Вимкнути oneDNN оптимізації
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
from flask_cors import CORS  # Додаємо CORS


app = Flask(__name__)
CORS(app)

import pandas as pd
from util.data_processing import load_data
from util.recommendation_algorithms import *



merged_data, books_data, books_rating = load_data('books_data_100_test.csv', 'books_rating_books_100_test.csv')


recommender = create_recommender(books_rating)

print(len(merged_data))
# Перевіряємо результат
print(merged_data[['Title', 'final_combined']].head())

book_texts = merged_data[['Title', 'final_combined']]


# TOKEN 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
# email: norehis608@ikowat.com
# pass: Norehis608!

@app.route('/books/user_desc', methods=['POST'])
def find_books_by_user_description():
    # Отримуємо дані з запиту
    data = request.get_json()  # Очікуємо, що клієнт надішле JSON
    user_description = data.get('user_description')  # Отримуємо опис користувача
    top_n = data.get('top_n', 5)  # Отримуємо top_n, якщо він є, або за замовчуванням 5

    # Перевіряємо, чи є необхідні дані
    if not user_description:
        return jsonify({'error': 'user_description is required'}), 400

    # Викликаємо метод для пошуку відповідних книг
    matching_books = find_best_matching_books(user_description, book_texts, books_data, top_n)

    # for i, book in enumerate(matching_books, 1):
    #     print(f"Rank {i}: {book['Title']} (Similarity: {book['Similarity Score']:.2f})")


    # Повертаємо результат у вигляді JSON
    print(matching_books)
    return jsonify(matching_books)

# Список дозволених емоцій
valid_emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

@app.route('/books/emotion', methods=['POST'])
def find_books_by_user_emotion():
    # Отримуємо JSON з тіла запиту
    data = request.get_json()

    # Отримуємо emotion та top_n
    emotion = data.get('emotion')
    top_n = data.get('top_n', 5)  # Якщо top_n не вказано, за замовчуванням буде 5

    # Перевірка, чи emotion є в списку дозволених емоцій
    if emotion not in valid_emotions:
        return jsonify({"error": "Invalid emotion. Valid emotions are: " + ", ".join(valid_emotions)}), 400

    # Викликаємо функцію для пошуку книг
    matching_books = find_books_by_emotion(emotion, books_rating, top_n)

    # Повертаємо результат у вигляді JSON
    return jsonify(matching_books)

@app.route('/books/all', methods=['GET'])
def get_all_books():
    # Витягнемо дані з обох таблиць (books_data і books_rating)
    books_list = []

    # Отримуємо всі книги з books_data
    for _, row in books_data.iterrows():

        # book_info = {
        #     'title': book_title == 'nan' ? '' : book_title,
        #     'description': row['description'],
        #     'authors': row['authors'] == ,
        #     'image': row['image'] == 'nan' ? '' : row['image'],  # Якщо є зображення, використовуємо його
        # }

        book_info = {
            'id': row['book_id'],
            'title': row['Title'] if pd.notna(row['Title']) else '',  # Якщо 'Title' є nan, заміняємо на порожній рядок
            'description': row['description'] if pd.notna(row['description']) else '',  # Якщо 'description' є nan, заміняємо на порожній рядок
            'authors': str(row['authors']).strip('[\']') if pd.notna(row['authors']) else '',  # Якщо 'authors' є nan, заміняємо на порожній рядок
            'image': row['image'] if pd.notna(row['image']) else '',  # Якщо 'image' є nan, заміняємо на порожній рядок
            # 'isbn': row['isbn'] if pd.notna(row['isbn']) else '',  # Якщо 'isbn' є nan, заміняємо на порожній рядок
            # 'language': row['language'] if pd.notna(row['language']) else '',  # Якщо 'language' є nan, заміняємо на порожній рядок
            # 'publisher': row['publisher'] if pd.notna(row['publisher']) else '',  # Якщо 'publisher' є nan, заміняємо на порожній рядок
            # 'publication_date': row['publication_date'] if pd.notna(row['publication_date']) else '',  # Якщо 'publication_date' є nan, заміняємо на порожній рядок
        }

        books_list.append(book_info)

        if row['Title'] == 'Nation Dance: Religion, Identity and Cultural Difference in the Caribbean':
            print(book_info)

    # print(books_list)
    return jsonify(books_list)

@app.route('/books/<int:id>', methods=['GET'])
def get_book_by_id(id):
    # Шукаємо книгу з відповідним id в books_data
    book_info = books_data[books_data['book_id'] == id].iloc[0]

    if book_info.empty:
        return jsonify({"error": "Book not found"}), 404

    # Формуємо відповідь
    book = {
        'id': int(book_info['book_id']),
        'title': book_info['Title'] if pd.notna(book_info['Title']) else '',
        'description': book_info['description'] if pd.notna(book_info['description']) else '',
        'authors': str(book_info['authors']).strip('[]') if pd.notna(book_info['authors']) else '',
        'image': book_info['image'] if pd.notna(book_info['image']) else '',
        'published': book_info['publishedDate'] if pd.notna(book_info['publishedDate']) else '',
    }

    return jsonify(book)


@app.route('/books/<int:book_id>/reviews', methods=['GET'])
def get_book_reviews(book_id):
    # Отримуємо книгу за book_id з books_data
    book = books_data[books_data['book_id'] == book_id]

    # Якщо книга не знайдена
    if book.empty:
        return jsonify({"error": "Book not found"}), 404

    # Отримуємо всі відгуки для цієї книги з books_rating за назвою книги
    book_title = book.iloc[0]['Title']
    reviews = books_rating[books_rating['Title'] == book_title]

    # Формуємо список відгуків
    reviews_list = []

    for _, row in reviews.iterrows():
        review_info = {
            'id': int(row['review_id']),
            'user': row['profileName'] if pd.notna(row['profileName']) else 'Anonymous',
            'rating': row['review/score'] if pd.notna(row['review/score']) else '',
            'helpfulness': row['review/helpfulness'] if pd.notna(row['review/helpfulness']) else '',
            'summary': row['review/summary'] if pd.notna(row['review/summary']) else '',
            'text': row['review/text'] if pd.notna(row['review/text']) else '',
            'review_time': row['review/time'] if pd.notna(row['review/time']) else ''
        }
        reviews_list.append(review_info)

    return jsonify(reviews_list)

@app.route('/books/users', methods=['POST'])
def get_user_recommendations():
    data = request.get_json()
    user_id = data.get('user_id')
    top_n = data.get('top_n', 5)

    matching_books = recommender.user_based_recommendations(user_id, top_n)

    books = []

    for book_title in matching_books:

        book_id = int(books_data[books_data['Title'] == book_title].iloc[0]['book_id'])

        books.append({"id": book_id, "title": book_title})

    print(books)

    return jsonify(books)

@app.route('/books/items', methods=['POST'])
def get_book_recommendations():
    data = request.get_json()
    book_title = data.get('book_title')
    top_n = data.get('top_n', 5)

    matching_books = recommender.item_based_recommendations(book_title, top_n)

    books = []

    for book_title in matching_books:

        book_id = int(books_data[books_data['Title'] == book_title].iloc[0]['book_id'])

        books.append({"id": book_id, "title": book_title})

    print(books)

    return jsonify(books)


@app.route('/welcome')
def welcome():
    return jsonify('hello!')

if __name__ == '__main__':
    # app.run(debug=True, port=9000)
    app.run(debug=False, port=9000)
