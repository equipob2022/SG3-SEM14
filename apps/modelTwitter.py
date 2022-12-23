import streamlit as st
import twint
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
import sys
#usar textblob para analizar el sentimiento de cada tweet
from textblob import TextBlob
from config import load_tweet
sys.path.append('../')

def app():
    asyncio.set_event_loop(asyncio.new_event_loop())
    palabra = st.text_input('Ingrese el topic a analizar', 'Nintendo')
    st.subheader('Análisis de sentimiento de '+palabra)
    #establecer el numero de tweets a buscar
    num = st.number_input('Ingrese el numero de tweets a buscar', 1, 5000, 10)
    #crear un dataframe vacio
    df_tweets = pd.DataFrame()
    with st.spinner('Extrayendo tweets 🐥🐥🐥, espere por favor...'):
        #buscar los tweets
        df_tweets = load_tweet(palabra, num)

    #poner un mensaje mientras se cargan los tweets con emojis
    #escribir en streamlit
    #hacemos una funcion para limpiar los tweets
    def clean_text_round1(text):
        '''Poner el texto en minúsculas, elimine el texto entre corchetes, elimine la puntuación y elimine las palabras que contienen números.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    # creamos una nueva columna para los tweets limpios
    round1 = lambda x: clean_text_round1(x)
    df_tweets['clean_tweet'] = df_tweets['tweet'].apply(round1)
    # Ahora aplicaremos la siguiente técnica: Eliminar algunas puntuaciones, textos o palabras que no tengan sentido
    def clean_text_round2(text):
        '''Suprimir algunos signos de puntuación adicionales y texto sin sentido.'''
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        return text

    round2 = lambda x: clean_text_round2(x)
    df_tweets['clean_tweet'] = df_tweets['clean_tweet'].apply(round2)
    #funcion para eliminar todos los caracteres que no sean letras en ingles
    def remove_non_ascii_1(text):
        '''Remove non-ASCII characters from list of tokenized words'''
        return re.sub(r'[^\x00-\x7f]',r'', text)
    #aplicamos la funcion a la columna de tweets
    df_tweets['clean_tweet'] = pd.DataFrame(df_tweets.clean_tweet.apply(remove_non_ascii_1))
    #las filas que no tienen tweets se eliminan
    df_tweets = df_tweets[df_tweets['clean_tweet'] != '']
    st.subheader('Tweets extraidos')
    st.write(df_tweets)
    #guardar los tweets en un csv
    df_tweets.to_csv('tweets.csv', index=False)

    #leer el csv de los tweets
    df_clean = pd.read_csv('tweets.csv')
    # Vamos a crear una matriz documento-término usando CountVectorizer, y a excluir las stop words comunes en inglés
    #stopword en español
    #obtener nuestras stop words en base a los tweets
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(df_clean.clean_tweet)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = df_clean.index
    #imprimir la matriz documento-término
    st.subheader('Matriz documento-término')
    st.write(data_dtm)
    #hacemos una nube de palabras para ver las palabras mas comunes

    # Join the different processed titles together.
    long_string = ','.join(list(df_tweets['clean_tweet'].values))

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud in streamlit
    st.subheader('Nube de palabras')
    st.image(wordcloud.to_array())

    #mostrar un grafico de barras con las palabras mas comunes
    #modo oscuro
    sns.set_style('darkgrid')

    # Crear un dataframe de las palabras más comunes
    data_words = pd.DataFrame(wordcloud.words_.items(), columns=['word', 'count'])
    #seleccionar las 10 palabras mas comunes
    data_words = data_words.iloc[:10, :]
    # Visualizar los palabras más comunes con plotly.express
    fig = px.bar(data_words, x='word', y='count', color='count', height=400)
    st.plotly_chart(fig)
    #mensaje de que se esta calculando el sentimiento
    with st.spinner('Calculando el sentimiento de los tweets ❤️'):
        #crear una funcion para calcular el sentimiento
        def detect_sentiment(text):
            return TextBlob(text).sentiment.polarity

        #crear una funcion para calcular la subjetividad
        def detect_subjectivity(text):
            return TextBlob(text).sentiment.subjectivity

        # aplicamos la funcion a la columna de tweets
        df_tweets['sentiment'] = df_tweets['clean_tweet'].apply(detect_sentiment)
        df_tweets['subjectivity'] = df_tweets['clean_tweet'].apply(detect_subjectivity)
        #mostrar el dataframe con los tweets y sus sentimientos
        st.subheader('Tweets con sentimiento')
        #selecciona las columnas que nos interesan
        st.write(df_tweets[['tweet', 'sentiment', 'subjectivity']])

