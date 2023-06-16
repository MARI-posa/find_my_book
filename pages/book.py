import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import re
import pickle
import requests
from io import BytesIO

st.title("Книжные рекомендации")

# Загрузка модели и токенизатора
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Загрузка датасета и аннотаций к книгам
books = pd.read_csv('all+++.csv')
books['author'].fillna('other', inplace=True)

annot = books['annotation']

# Получение эмбеддингов аннотаций каждой книги в датасете
length = 256

# Определение запроса пользователя
query = st.text_input("Введите запрос")

num_books_per_page = st.selectbox("Количество книг на странице:", [3, 5, 10], index=0)

col1, col2 = st.columns(2)
generate_button = col1.button('Сгенерировать')

if generate_button:
    with open("book_embeddings256xxx.pkl", "rb") as f:
        book_embeddings = pickle.load(f)

    query_tokens = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=length,  # Ограничение на максимальную длину входной последовательности
        pad_to_max_length=True,  # Дополним последовательность нулями до максимальной длины
        return_tensors='pt'  # Вернём тензоры PyTorch
    )

    with torch.no_grad():
        query_outputs = model(**query_tokens)
        query_hidden_states = query_outputs.hidden_states[-1][:, 0, :]
        query_hidden_states = torch.nn.functional.normalize(query_hidden_states)

    # Вычисление косинусного расстояния между эмбеддингом запроса и каждой аннотацией
    cosine_similarities = torch.nn.functional.cosine_similarity(
        query_hidden_states.squeeze(0),
        torch.stack(book_embeddings)
    )

    cosine_similarities = cosine_similarities.numpy()

    indices = np.argsort(cosine_similarities)[::-1]  # Сортировка по убыванию

    for i in indices[:num_books_per_page]:
        col1, col2 = st.columns([5,7])

        with col2:
        #cols = st.columns(2)  # Создание двух столбцов для размещения информации и изображения
            st.write("## " + books['title'][i])
            st.markdown("**Автор:** " + books['author'][i])
            st.markdown("**Аннотация:** " + books['annotation'][i])
            image_url = books['image_url'][i]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.markdown(f"**[Купить книгу]({books['page_url'][i]})**")
        with col1:
            st.write("<div style='text-align: center; font-size: 5px;'></div>", unsafe_allow_html=True)
            st.image(image)
            st.write(f'совпадение с запросом: {cosine_similarities[i]:.2f}')
            st.markdown(books['genre'][i])
            st.write("---")
