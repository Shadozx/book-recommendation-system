# app/sentiment_analysis.py
from transformers import pipeline

detectEmotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

def analyze_sentiment(review_text):
    return detectEmotion(review_text)
