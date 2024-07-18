import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import re
import warnings

# Cosas adicionales que necesitamos para los modelos de ML
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
warnings.filterwarnings('ignore', message='X does not have valid feature names')


# ------------------------------- Cargar modelos y datos --------------------------

## Modelo Analisis de sentimiento...
modelo_sentiment = joblib.load('..\\Sentiment\\Modelo_Sentiment.joblib')
vectorizador_sentiment = joblib.load('..\\Sentiment\\Vectorizador_Sentiment.joblib')


## Modelo Recomendaciones...
#Cargamos nuestro modelo:
modelo_recomendation = joblib.load('..\\SistemaRecomendacion\\Model_Recomendation.joblib')

# Cargar el dataset
df_recomendation = pd.read_parquet('..\\Data\\gmap_meta_reviews_sentiment.parquet', engine= 'pyarrow')

# Creamos un nuevo Dataframe para cambiar el nombre de unas columnas
df_clean = pd.DataFrame()
df_clean['user_id'] = df_recomendation['user_id']
df_clean['name_user'] = df_recomendation['name_y']
df_clean['restaurant_id'] = df_recomendation['gmap_id']
df_clean['name_restaurant'] = df_recomendation['name_x']
df_clean['address'] = df_recomendation['address']
df_clean['rating'] = df_recomendation['rating']
df_clean['category'] = df_recomendation['category']
df_clean['classification'] = df_recomendation['avg_rating']
df_clean['review'] = df_recomendation['text']

df_recomendation = df_clean.head(50000)

# Crear un objeto Reader y definir el rango de los ratings
reader_recomendation = Reader(rating_scale=(1, 5))
# Crear el dataset de Surprise
data_recomendation = Dataset.load_from_df(df_recomendation[['user_id', 'name_restaurant', 'rating']], reader_recomendation)
# Dividir el dataset en entrenamiento y prueba
trainset_recomendation, testset_recomendation = train_test_split(data_recomendation, test_size=0.2, random_state=42)

## Modelo Rating Prediction
#Cargamos los modelos al inicializar la API
modelo_rating = joblib.load('..\\Rating\\Modelo_Rating_Logistic.joblib')
vectorizador_rating = joblib.load('..\\Rating\\Vectorizador_Rating.joblib')


## Modelo Fake Reviews
#Cargamos nuestros modelos y vectorizador
modelo_fake_revs = joblib.load('..\\FakeRevs\\Modelo_FakeRevs.joblib')
vectorizador_fake_revs = joblib.load('..\\FakeRevs\\Vectorizador_FakeRevs.joblib')


## Modelo Review Classifier
# Cargamos los modelos
modelo_rev_class = joblib.load('..\\ReviewsClasifier\\Modelo_RevClasifier.joblib')
vectorizador_rev_class = joblib.load('..\\ReviewsClasifier\\Vectorizador_RevClasifier.joblib')


# Funciones
def predict_rating(rev):
    new_review_vector = vectorizador_rating.transform([rev]).toarray()
    prediction = modelo_rating.predict(new_review_vector)
    return prediction[0]


def get_unrated_restaurants(user_id, trainset):
    uid = trainset.to_inner_uid(user_id)
    user_rated_items = set(j for (j, _) in trainset.ur[uid])
    all_items = set(trainset.all_items())
    unrated_items = all_items - user_rated_items
    return unrated_items

def get_recommendations_for_user(user_id, trainset, model, n=10):
    unrated_items = get_unrated_restaurants(user_id, trainset)
    predictions = []
    for item_id in unrated_items:
        raw_item_id = trainset.to_raw_iid(item_id)
        predictions.append((raw_item_id, model.predict(user_id, raw_item_id).est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

def analisis_sentimiento(review):
    review_transformada = vectorizador_sentiment.transform([review])
    rating_predicho = modelo_sentiment.predict(review_transformada)
    return rating_predicho[0]

def clean_text_fake_revs(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def clean_text_rev_class(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def analisis_reseña_falsa(review):
    cleaned_review = clean_text_fake_revs(review)
    prediction = modelo_fake_revs.predict(vectorizador_fake_revs.transform([cleaned_review]).toarray())
    return prediction[0]

def analisis_aspectos_reseña(review):
    cleaned_new_review = clean_text_rev_class(review)
    new_review_vector = vectorizador_rev_class.transform([cleaned_new_review]).toarray()
    prediction = modelo_rev_class.predict(new_review_vector)
    
    return prediction

# Streamlit app
st.title("Sistema de recomendación y análisis de reseñas de restaurantes")

# Rating Prediction
st.header("Predecir la Calificación del Restaurant a partir de la Reseña")
review_input = st.text_area("Introduce la reseña del restaurante:")
if st.button("Predecir calificación"):
    rating = predict_rating(review_input)
    st.write(f"La calificación prevista para la Reseña es {rating}")

# Enlace de la visualización de Looker Studio
looker_studio_url = "https://lookerstudio.google.com/embed/reporting/92454d30-31b7-44b8-97ea-6e915c49483e/page/p_qndu6slwid"

# Utilizar un iframe para incrustar la visualización
st.markdown(f'<iframe width="100%" height="600" src="{looker_studio_url}" frameborder="0" style="border:0" allowfullscreen></iframe>', unsafe_allow_html=True)

# Recommendation System
st.header("Recomendaciones de restaurantes")
usuario1 = df_recomendation['user_id'][2500]
st.write(usuario1)
user_id_input = st.text_input("Ingrese user ID:")
if st.button("Obtener recomendaciones"):
    recommendations = get_recommendations_for_user(user_id_input, trainset_recomendation, modelo_recomendation, n=5)
    st.write(f"Las 5 mejores recomendaciones para el usuario {user_id_input}:")
    for restaurant_id, estimated_rating in recommendations:
        st.write(f"Restaurant: {restaurant_id}, Calificación estimada: {int(estimated_rating)}")

# Sentiment Analysis
st.header("Análisis de sentimiento de la reseña del restaurante")
review_sentiment_input = st.text_area("Ingrese la reseña del restaurante para realizar un análisis de sentimiento:")
if st.button("Analizar Sentimiento"):
    sentiment = analisis_sentimiento(review_sentiment_input)
    sentiment_text = "Neutro" if sentiment == 1 else ("Bueno" if sentiment == 2 else "Malo")
    st.write(f"El sentimiento previsto para la revisión es: {sentiment_text}")

# Sentiment Analysis and Fake Review Detection
st.header("Análisis de sentimiento y detección de reseñas falsas")
review_sentiment_input = st.text_area("Ingrese la reseña del restaurante para realizar un análisis de sentimiento y detección de falsedad:")
if st.button("Analizar Sentimiento y Detectar Falsedad"):
    sentiment = analisis_sentimiento(review_sentiment_input)
    st.write(f"El sentimiento previsto para la revisión es: {sentiment}")

    is_fake = analisis_reseña_falsa(review_sentiment_input)
    if is_fake == 1:
        st.write("Esta reseña es sospechosa de ser falsa.")
    else:
        st.write("Esta reseña parece ser genuina.")

# Aspect Analysis
st.header("Análisis de aspectos de la reseña")
review_aspect_input = st.text_area("Ingrese la reseña del restaurante para analizar los aspectos (servicio, comida, ambiente):")
if st.button("Analizar Aspectos"):
    aspects = analisis_aspectos_reseña(review_aspect_input)
    st.write("Análisis de aspectos de la reseña:")
    st.write(f"Servicio: {'Positivo' if aspects[0][0] else 'Negativo'}")
    st.write(f"Comida: {'Positivo' if aspects[0][1] else 'Negativo'}")
    st.write(f"Ambiente: {'Positivo' if aspects[0][2] else 'Negativo'}")