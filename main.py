import streamlit as st

st.set_page_config(layout="wide")

import base64
import streamlit as st
import plotly.express as px

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.ibb.co/0m0ZYt6/books-assortment-with-dark-background.jpg");
background-size: 110%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(1,1,1,1);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

div.css-1n76uvr.esravye0 {{
background-color: rgba(238, 238, 238, 0.5);
border: 10px solid #EEEEEE;
padding: 5% 5% 5% 10%;
border-radius: 5px;
}}



</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

col1, col2, col3 = st.columns([3,5,2])

with col2:
    st.title('📚Smart Book Search by FindMyBook:🔍')

col1, col2, col3 = st.columns([2,5,2])

with col2:
    st.markdown("<div style='text-align: center; font-size: 30px;'>Team members:</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 25px;'>📌 Maria K.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;📌 Ilvir Kh.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 25px;'>📌 Viktoria K.&nbsp;&nbsp;&nbsp;&nbsp;📌 Anna F.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 25px;'>📌 Ivan N.&nbsp;&nbsp;&nbsp;&nbsp;</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 25px;'></div>", unsafe_allow_html=True)
