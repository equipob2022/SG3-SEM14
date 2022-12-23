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
import gensim
from gensim import matutils, models
import scipy.sparse
from gensim import interfaces, utils, matutils
sys.path.append('../')

def app():
    asyncio.set_event_loop(asyncio.new_event_loop())
    palabra = st.text_input('Ingrese el topic a analizar', 'Pedro Castillo')
    st.subheader('Análisis de sentimiento de '+palabra)
    #establecer el numero de tweets a buscar
    num = st.number_input('Ingrese el numero de tweets a buscar', 1, 5000, 100)
    #crear un dataframe vacio
    df_tweets = pd.DataFrame()
    with st.spinner('Extrayendo tweets 🐥🐥🐥, espere por favor...'):
        #buscar los tweets
        df = load_tweet(palabra, num)
    # seleccionar solo las columnas que necesitamos
    df_tweets = df[['date', 'tweet', 'username', 'nlikes', 'nreplies', 'nretweets']]
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
    st.subheader('Palabras más comunes')
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
    
        #hacer una nube de palabras con los tweets positivos
        #seleccionar los tweets positivos
        df_pos = df_tweets[df_tweets['sentiment'] > 0]
        # Join the different processed titles together.
        long_string = ','.join(list(df_pos['clean_tweet'].values))

        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        
        # Generate a word cloud
        wordcloud.generate(long_string)
        
        # Visualize the word cloud in streamlit
        st.subheader('Nube de palabras de tweets positivos')
        st.image(wordcloud.to_array())

        #hacer una nube de palabras con los tweets negativos
        #seleccionar los tweets negativos
        df_neg = df_tweets[df_tweets['sentiment'] < 0]
        # Join the different processed titles together.
        long_string = ','.join(list(df_neg['clean_tweet'].values))

        # Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

        # Generate a word cloud
        wordcloud.generate(long_string)

        # Visualize the word cloud in streamlit
        st.subheader('Nube de palabras de tweets negativos')
        st.image(wordcloud.to_array())


    with st.spinner('Cargando grafica de sentimiento'):
        #grafico de sentimiento y subjetividad con plotly
        st.subheader('Grafico de sentimiento y subjetividad')
        fig = px.scatter(df_tweets, x="sentiment", y="subjectivity", color="sentiment",
                            hover_data=['tweet'])
        st.write("Eje horizontal: Mientras más cercano a 1, más positivo es el comentario Mientras más cercano a -1, más negativo es el sentimiento.")
        st.write("Eje vertical: Mientras más cercano a 1, más subjetivo es el comentario Mientras más cercano a 0, más objetivo es el comentario.")
        st.plotly_chart(fig)
    with st.spinner('Contando comentarios positivos y negativos'):
        #hacer un grafico circular de los sentimientos positivos y negativos con plotly
        # si el sentimiento es mayor a 0, es positivo, si es menor a 0 es negativo 
        #contar los tweets positivos y negativos
        df_tweets['label'] = df_tweets['sentiment'].apply(lambda x: 'Positivo' if x > 0 else 'Negativo')
        #crear un dataframe con los sentimientos
        df_sent = df_tweets['label'].value_counts().reset_index()
        df_sent.columns = ['sentimiento', 'total']
        #grafico circular
        st.subheader('Contador de comentarios positivos y negativos')
        fig = px.pie(df_sent, values='total', names='sentimiento', title='Sentimientos')
        st.plotly_chart(fig)

    num_temas = st.slider('Numero de temas', 1, 10, 5)
    with st.spinner('Analizando temas de los tweets'):
        # crear un diccionario de palabras para el modelo
        cv = CountVectorizer(stop_words='english')
        data_cv = cv.fit_transform(df_tweets.clean_tweet)
        data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
        data_stop.index = df_tweets.index
        #crear el modelo de LDA
        # Convertir una matriz dispersa de conteos en un corpus gensim
        corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_stop.transpose()))

        # Gensim también requiere un diccionario de todos los términos y su ubicación respectiva en la matriz de documentos de términos
        id2word = dict((v, k) for k, v in cv.vocabulary_.items())

        # Crear modelo lda (equivalente a "fit" en sklearn)
        lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_temas, passes=40)
        #guardar cada topico como la combinacion de las palabras
        # de cada topico
        topics = lda.show_topics(formatted=False)
        # estraee solo la palabra de cada topico
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in topics]
        topics_words
        #armar una string con las palabras de cada topico unidad porcomas
        topics_string = []
        for topic in topics_words:
            topics_string.append(' '.join(topic[1]))
        topics_string
        #renombrar la columna de los topico
        topics_string = pd.DataFrame(topics_string, columns=['topic'])
        topics_string
        #armar un dataframe con los topico y las palabras
        df_topics_names= pd.DataFrame(topics_string)

        # Ver los temas en el modelo LDA
        st.subheader('Temas en los tweets')
        for i in range(0, df_topics_names.shape[0]):
            st.write('Tema', i, ':', df_topics_names.iloc[i, 0])
        # Echemos un vistazo a los temas que contiene cada tweet
        # y guardarlo en un dataframe
        corpus_transformed = lda[corpus]
        topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in corpus_transformed]
        df_topics = pd.DataFrame(topics, columns=['Topico', 'Importancia'])
        #grafucar los topico con plotly
        st.subheader('Grafico de los topico')
        fig = px.histogram(df_topics, x="Topico", y="Importancia", color="Topico", height=400)
        st.plotly_chart(fig)
    #mostrar que usuario tiene el comentario mas positivo
    with st.spinner('Calculando el usuario con el comentario mas positivo'):
        #seleccionar el tweet mas positivo
        tweet_positivo = df_tweets.loc[df_tweets['sentiment'].idxmax()]
        #mostrar el tweet mas positivo
        st.subheader('El tweet mas positivo y su usuario')
        st.write(tweet_positivo['tweet'])
        st.write("Usuario 😂😉: "+tweet_positivo['username'])
    #mostrar que usuario tiene el comentario mas negativo
    with st.spinner('Calculando el usuario con el comentario mas negativo'):
        #seleccionar el tweet mas negativo
        tweet_negativo = df_tweets.loc[df_tweets['sentiment'].idxmin()]
        #mostrar el tweet mas negativo
        st.subheader('El tweet mas negativo y su usuario')
        st.write(tweet_negativo['tweet'])
        st.write("Usuario 😢😔: "+tweet_negativo['username'])
    #mostrar el usuario con mas likes en sus tweets
    with st.spinner('Calculando el usuario con mas likes en sus tweets'):
        #seleccionar el usuario con mas likes
        usuario_likes = df_tweets.loc[df_tweets['nlikes'].idxmax()]
        #mostrar el usuario con mas likes
        st.subheader('El usuario con mas likes en sus tweets')
        st.write("Usuario 👍👍: "+usuario_likes['username'])
    #mostrar el usuario con mas retweets en sus tweets
    with st.spinner('Calculando el usuario con mas retweets en sus tweets'):
        #seleccionar el usuario con mas retweets
        usuario_retweets = df_tweets.loc[df_tweets['nretweets'].idxmax()]
        #mostrar el usuario con mas retweets
        st.subheader('El usuario con mas retweets en sus tweets')
        st.write("Usuario 🔼🔼: "+usuario_retweets['username'])


