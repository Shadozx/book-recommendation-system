# датасети
# books_data(Title,description,authors,image,previewLink,publisher,publishedDate,infoLink,categories,ratingsCount)
# books_rating(Id,Title,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text)

import pandas as pd


datasets_path = '../datasets'
book_reviews_path = datasets_path + '/book_reviews'
test_folder = book_reviews_path + '/test'

books_count = 500

books_data = pd.read_csv(test_folder + f'/books_data_{books_count}_test.csv')
print('good')
books_rating = pd.read_csv(test_folder + f'/books_rating_books_{books_count}_test.csv')
print('very good')

print(books_data)
print(books_rating)