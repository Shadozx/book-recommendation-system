import pandas as pd

from util.data_processing import load_data
from util.recommendation_algorithms import find_books_by_emotion

books_count = 100

merged_data, books_data, books_rating = load_data(f'books_data_{books_count}_test.csv', f'books_rating_books_{books_count}_test.csv')


# books_rating = pd.read_csv(book_reviews_path + '/books_rating.csv')
print('very good')


from util.sentiment_analysis import create_emotion_analysis_df

books_rating = books_rating.head(20)


emotion_books_rating = create_emotion_analysis_df(books_rating)

print(emotion_books_rating)

print(find_books_by_emotion('admiration', books_data, emotion_books_rating))

