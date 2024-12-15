# app/sentiment_analysis.py
import time
import pandas as pd
from transformers import pipeline

detectEmotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

def analyze_sentiment(review_text):
    # return detectEmotion(review_text[:512])

    try:
        result = detectEmotion(review_text[:512])  # Обмежуємо довжину тексту до 512 символів
        return result[0]['label'], result[0]['score']
    except Exception as e:
        print(f"Error processing text: {e}")
        return None, None

def create_emotion_analysis_df(books_rating):
    # Список для збереження результатів
    start_time = time.time()
    print('Total review count:', len(books_rating))

    sentiment_results = []

    # Проходимо по всіх записах у books_rating
    for idx, row in books_rating.iterrows():
        title = row['Title']
        review_text = row['review/text']

        # Аналіз емоції для кожного тексту огляду
        emotion, score = analyze_sentiment(review_text)

        # Додаємо результат до списку
        sentiment_results.append({
            # 'review_id': row['review_id'],
            'Title': title,
            'User_id':row['User_id'],
            'review/text': review_text,
            'emotion': emotion,
            'emotion_score': score
        })

        # Виведення статусу обробки
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} reviews")

    # Створюємо DataFrame з результатів
    sentiment_df = pd.DataFrame(sentiment_results)

    end_time = time.time()

    print('Total review time:', end_time - start_time)

    return sentiment_df
