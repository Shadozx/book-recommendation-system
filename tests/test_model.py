# # from transformers import pipeline
# #
# # # Створення пайплайну для генерації тексту
# # text_generator = pipeline("text-generation", model="gpt2")
# #
# # # Вхідний текст (контекст)
# # input_text = "Once upon a time in a faraway land"
# #
# # # Генерація тексту
# # result = text_generator(input_text, max_length=50, num_return_sequences=1)
# #
# # # Виведення результату
# # print(result[0]['generated_text'])
# from transformers import GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
# text = '''0826414346,Dr. Seuss: American Icon,,A30TK6U7DNS82R,Kevin Killian,10/10,5.0,1095724800,Really Enjoyed It,
# I don't care much for Dr. Seuss but after reading Philip Nel's book I changed my mind--that's a good testimonial to the power of Rel's writing and thinking.
# Rel plays Dr. Seuss the ultimate compliment of treating him as a serious poet as well as one of the 20th century's most interesting visual artists, and after reading his book I decided that a trip to the Mandeville Collections of the library at University of California in San Diego was in order, so I could visit some of the incredible Seuss/Geisel holdings they have there.
# There's almost too much to take in, for, like William Butler Yeats, Seuss led a career that constantly shifted and metamoprhized itself to meet new historical and political cirsumstances, so he seems to have been both a leftist and a conservative at different junctures of his career, both in politics and in art. As Nel shows us, he was once a cartoonist for the fabled PM magazine and, like Andy Warhol, he served his time slaving in the ad business too. All was in the service of amusing and broadening the minds of US children. Nel doesn't hesitate to administer a sound spanking to the Seuss industry that, since his death, has seen fit to license all kinds of awful products including the recent CAT IN THE HAT film with Mike Myers. Oh, what a cat-astrophe!
# The book is great and I can especially recommend the work of the picture editor who has given us a bounty of good illustrations.'''
#
# tokens = tokenizer.tokenize(text)
# print(f"Токени: {tokens}")
# print(f"Кількість токенів: {len(tokens)}")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Завантажуємо модель для перетворення текстів в вектори
model = SentenceTransformer('all-MiniLM-L6-v2')

# Опис книги
user_description = "Книга повинна бути фантастикою, з елементами пригод та містики. Головний герой повинен мати надзвичайні здібності та переживати внутрішню боротьбу."

# Відгук про книгу
book_review = "Ця книга — чудова історія про хлопця з надзвичайними здібностями, який потрапляє у захопливі пригоди в світі, де магія і наука переплітаються. Головний герой бореться з власними страхами і сумнівами."

# Перетворення текстів в вектори
user_description_vector = model.encode([user_description])
book_review_vector = model.encode([book_review])

# Обчислення косинусної схожості
similarity_score = cosine_similarity(user_description_vector, book_review_vector)[0][0]

# Виведення результату
print(f"Схожість: {similarity_score:.2f}")
