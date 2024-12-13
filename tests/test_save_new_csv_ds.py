# датасети
# books_data(Title,description,authors,image,previewLink,publisher,publishedDate,infoLink,categories,ratingsCount)
# books_rating(Id,Title,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text)
import time
import pandas as pd

start_time = time.time()

datasets_path = '../datasets'
book_reviews_path = datasets_path + '/book_reviews'

books_data = pd.read_csv(book_reviews_path + '/books_data.csv')
print('good')
print(books_data.shape)
books_rating = pd.read_csv(book_reviews_path + '/books_rating.csv')
print('very good')
print(books_rating.shape)

# # Вибираємо перші 1000 книг з books_data
# books_data_1000_df = books_data_df.head(1000)
#
# # Тепер для кожної книги з books_data_1000_df ми будемо шукати відгуки в books_rating_df за назвою (Title)
# # Зробимо це через злиття (merge) таблиць по полю "Title"
# merged_df = pd.merge(books_data_1000_df, books_rating_df, on="Title", how="left")
#
# # Збережемо результат в нові CSV файли
# books_data_1000_df.to_csv('books_data_1000.csv', index=False)
# merged_df.to_csv('books_ratings_on_books_1000.csv', index=False)
#
# print("Дані збережено у файли books_data_1000.csv та books_ratings_filtered.csv")


books_count = 1000

# Вибираємо перші n книг з books_data
books_data_df = books_data.head(books_count)

# Отримуємо список назв книг для фільтрації відгуків
book_titles = books_data_df['Title'].tolist()

books_data_df = books_data_df[books_data_df['Title'].isin(book_titles)]

# Фільтруємо відгуки по цим назвам
books_ratings_df = books_rating[books_rating['Title'].isin(book_titles)]

test_folder = book_reviews_path + '/test'

print(books_data_df)
print(books_data_df.shape)
print(books_ratings_df.shape)
# Збережемо результат в нові CSV файли
books_data_df.to_csv(test_folder + f'/books_data_{books_count}_test.csv', index=False)
books_ratings_df.to_csv(test_folder + f'/books_rating_books_{books_count}_test.csv', index=False)
print(f'done for {time.time()-start_time} seconds')