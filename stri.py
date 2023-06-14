import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

st.title("Книжные рекомендации")

# Загрузка модели и токенизатора
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Загрузка датасета и аннотаций к книгам
books = pd.read_csv('book_train.csv')
annot = books['annotation']

# Предобработка аннотаций и получение эмбеддингов
embeddings = []
for annotation in annot:
    annotation_tokens = tokenizer.encode_plus(
        annotation,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**annotation_tokens)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-2]
        embeddings.append(torch.mean(last_hidden_state, dim=1).squeeze())

# Получение эмбеддинга запроса от пользователя
query = st.text_input("Введите запрос")
query_tokens = tokenizer.encode_plus(
    query,
    add_special_tokens=True,
    max_length=128,
    pad_to_max_length=True,
    return_tensors='pt'
)

# Проверка, был ли введен запрос
if query:
    with torch.no_grad():
        query_outputs = model(**query_tokens)
        query_hidden_states = query_outputs.hidden_states
        query_last_hidden_state = query_hidden_states[-2]
        query_embedding = torch.mean(query_last_hidden_state, dim=1).squeeze()

    # Вычисление косинусного расстояния между эмбеддингом запроса и каждой аннотацией
    cosine_similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        torch.stack(embeddings)
    )

    cosine_similarities = cosine_similarities.numpy()

    indices = np.argsort(cosine_similarities)[::-1]

    st.header("Рекомендации")
    for i in indices[:10]:
        st.write(books['title'][i])
