# app/data_processing.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data(books_data_name, books_rating_name):
    datasets_path = './datasets'
    book_reviews_path = datasets_path + '/book_reviews'
    test_path = book_reviews_path + '/test/'

    books_data = pd.read_csv(test_path + books_data_name)
    print('books_data was downloaded')
    books_rating = pd.read_csv(test_path + books_rating_name)
    print('books_rating was downloaded')

    books_data['description'] = books_data['description'].fillna('')
    books_rating['review/text'] = books_rating['review/text'].fillna('')
    books_rating['review/summary'] = books_rating['review/summary'].fillna('')

    books_data['combined_text'] = books_data['Title'] + ' ' + books_data['description']
    books_rating['combined_reviews'] = books_rating['review/summary'] + ' ' + books_rating['review/text']

    books_data['book_id'] = range(1, len(books_data) + 1)
    books_rating['review_id'] = range(1, len(books_rating) + 1)

    merged_data = pd.merge(books_data[['book_id', 'Title', 'combined_text']],
                           books_rating[['Title', 'combined_reviews']],
                           on='Title',
                           how='outer')

    merged_data['final_combined'] = merged_data['combined_text'] + ' ' + merged_data['combined_reviews']
    merged_data['final_combined'] = merged_data['final_combined'].fillna('')

    return merged_data, books_data, books_rating


# datasets_path = './datasets'
# book_reviews_path = datasets_path + '/book_reviews'
#
# books_data = pd.read_csv(book_reviews_path + '/books_data_preview.txt')
# print('good')
# books_rating = pd.read_csv(book_reviews_path + '/books_rating_preview.txt')
# print('very good')
#
# # Обробка пропусків
# books_data['description'] = books_data['description'].fillna('')
# books_rating['review/text'] = books_rating['review/text'].fillna('')
# books_rating['review/summary'] = books_rating['review/summary'].fillna('')
# # Об'єднуємо назви книг, описи та відгуки в один загальний текст
# books_data['combined_text'] = books_data['Title'] + ' ' + books_data['description']
# books_rating['combined_reviews'] = books_rating['review/summary'] + ' ' + books_rating['review/text']
#
# books_data['book_id'] = range(1, len(books_data) + 1)
# books_rating['review_id'] = range(1, len(books_rating) + 1)
# print('Books data len: ', len(books_data))
# print('Books rating len: ', len(books_rating))
# # Векторизація тексту описів книг + відгуків
# # vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
#
# # Об'єднуємо два датафрейми по полю 'Title' з використанням зовнішнього об'єднання
# merged_data = pd.merge(books_data[['book_id', 'Title', 'combined_text']],#, 'description']],
#                        books_rating[['Title', 'combined_reviews']],#, 'review/summary', 'profileName']],
#                        on='Title',
#                        how='outer')
#
# merged_data['combined_text'] = merged_data['combined_text'].fillna('')
# merged_data['combined_reviews'] = merged_data['combined_reviews'].fillna('')
# # Створюємо новий стовпчик, який містить об'єднаний текст
# merged_data['final_combined'] = merged_data['combined_text'] + ' ' + merged_data['combined_reviews']
#
# # Якщо в деяких рядках є пропуски в комбінованих відгуках або описах, замінюємо їх на порожній рядок
# merged_data['final_combined'] = merged_data['final_combined'].fillna('')
# # merged_data['final_combined'] = merged_data['final_combined'].apply(preprocess)
#
# print(len(merged_data))
# # Перевіряємо результат
# print(merged_data[['Title', 'final_combined']].head())
#
# book_texts = merged_data[['Title', 'final_combined']]
