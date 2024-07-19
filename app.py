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
import os
from PIL import Image
import base64

# -------------------------------- Establecemos paths relativos -----------------------------------------
# Obtener la ruta absoluta del archivo
#Path base
base_dir = os.path.dirname(os.path.abspath(__file__))

#Sentimiento
modelo_Sentiment_path = os.path.join(base_dir,  'Sentiment', 'Modelo_Sentiment.joblib')
vectorizador_Sentiment_path = os.path.join(base_dir,  'Sentiment', 'Vectorizador_Sentiment.joblib')

#Recomendacion
modelo_recomendacion_path = os.path.join(base_dir,  'SistemaRecomendacion', 'Model_Recomendation.joblib')

#Cargar los datos del dataset
data_path = os.path.join(base_dir,  'Data', 'gmap_meta_reviews_sentiment.parquet')

#Prediccion Ratin
modelo_rating_path = os.path.join(base_dir,  'Rating', 'Modelo_Rating_Logistic.joblib')
vectorizador_rating_path = os.path.join(base_dir,  'Rating', 'Vectorizador_Rating.joblib')

#Reviews Falsas
modelo_fakerevs_path = os.path.join(base_dir,  'FakeRevs', 'Modelo_FakeRevs.joblib')
vectorizador_fakerevs_path = os.path.join(base_dir,  'FakeRevs', 'Vectorizador_FakeRevs.joblib')

#Reviews Classifier
modelo_revsclass_path = os.path.join(base_dir,  'ReviewsClasifier', 'Modelo_RevClasifier.joblib')
vectorizador_revsclass_path = os.path.join(base_dir,  'ReviewsClasifier', 'Vectorizador_RevClasifier.joblib')

#Imagenes
path_images = os.path.join(base_dir,  'Images')
image_1_path = os.path.join(path_images, '1.jpeg')
image_2_path = os.path.join(path_images, '2.jpeg')
image_3_path = os.path.join(path_images, '3.jpeg')
image_4_path = os.path.join(path_images, '4.jpeg')
image_5_path = os.path.join(path_images, '5.jpeg')
image_6_path = os.path.join(path_images, '6.jpeg')
image_logo_path = os.path.join(path_images, 'Logo.jpeg')
image_fondo_path = os.path.join(path_images, 'Fondo.jpeg')

#main para estilos
main_path = os.path.join(base_dir,  'Style', 'main.css')

# Cosas adicionales que necesitamos para los modelos de ML
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
warnings.filterwarnings('ignore', message='X does not have valid feature names')


# ------------------------------- Cargar modelos y datos --------------------------

## Modelo Analisis de sentimiento...
#modelo_sentiment = joblib.load('..\\Sentiment\\Modelo_Sentiment.joblib')
#vectorizador_sentiment = joblib.load('..\\Sentiment\\Vectorizador_Sentiment.joblib')
modelo_sentiment = joblib.load(modelo_Sentiment_path)
vectorizador_sentiment = joblib.load(vectorizador_Sentiment_path)


## Modelo Recomendaciones...
#Cargamos nuestro modelo:
modelo_recomendation = joblib.load(modelo_recomendacion_path)

# Cargar el dataset
df_recomendation = pd.read_parquet(data_path, engine= 'pyarrow')

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
modelo_rating = joblib.load(modelo_rating_path)
vectorizador_rating = joblib.load(vectorizador_rating_path)


## Modelo Fake Reviews
#Cargamos nuestros modelos y vectorizador
modelo_fake_revs = joblib.load(modelo_fakerevs_path)
vectorizador_fake_revs = joblib.load(vectorizador_fakerevs_path)


## Modelo Review Classifier
# Cargamos los modelos
modelo_rev_class = joblib.load(modelo_revsclass_path)
vectorizador_rev_class = joblib.load(vectorizador_revsclass_path)


## -------------------------------- Funciones -------------------------------------------
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

# función para el fondo
def get_base64_image(image_file):
    """Función para convertir una imagen en base64."""
    with open(image_file, "rb") as image:
        return base64.b64encode(image.read()).decode()


## -------------------------------------- Cuerpo de la página ------------------------------

#Nombre de nuestra página
st.set_page_config(page_title="Intelidata 🍴", layout="wide")
email_address ="jaredaugustolunaleon@gmail.com"

# Agregar el fondo personalizado
background_image = get_base64_image(image_fondo_path)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{background_image}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Agregar el logotipo en la parte superior derecha
logo_image = get_base64_image(image_logo_path)
st.markdown(
    f"""
    <style>
    .logo {{
        position: fixed;
        top: 10px;
        right: 10px;
        width: 100px; /* Ajusta el tamaño según tu necesidad */
    }}
    </style>
    <img src="data:image/jpeg;base64,{logo_image}" class="logo">
    """,
    unsafe_allow_html=True
)


# archivo css que configura los cuadros y textos del contacto
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css(main_path)


# Aplicar CSS personalizado para el tamaño y color del texto
st.markdown(
    """
    <style>
    .title-text {
        font-size: 24px; /* Tamaño grande para el título */
        color: #000000; /* Color negro */
        font-weight: bold; /* Negrita */
        text-align: justify;
    }
    .header-text {
        font-size: 32px; /* Tamaño del encabezado */
        color: #000000; /* Color negro */
        font-weight: bold; /* Negrita */
    }
    .normal-text {
        font-size: 22px; /* Tamaño de texto normal */
        color: #000000; /* Color negro */
        text-align: justify;
    }
    .model-text {
        font-size: 28px; /* Tamaño de texto normal */
        color: #000000; /* Color negro */
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<h1 class='header-text'>Hola, somos Intelidata 👋</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='title-text'>Creamos soluciones basadas en datos para la toma de decisiones!</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class='normal-text'>
        En nuestra consultora, nos apasiona la tecnología y la innovación. Nos especializamos en el procesamiento y análisis de datos, ofreciendo soluciones personalizadas a nuestros clientes que les permiten tomar decisiones de manera inteligente y mejorar significativamente su negocio. Con nuestra página web, obtendrás una herramienta poderosa diseñada para optimizar operaciones, atraer más clientes y maximizar tus ingresos en el sector de restaurantes. Permítenos ayudarte a transformar tu restaurante con tecnología de vanguardia y estrategias efectivas que impulsarán tu éxito!!!
        </p>
        """,
        unsafe_allow_html=True
    )


#sobre nosotros
# Aplicar CSS personalizado para el tamaño y color del texto
st.markdown(
    """
    <style>
    .big-text {
        font-size: 22px; /* Tamaño de letra grande */
        color: #000000; /* Color negro */
        text-align: justify;
    }
    .highlight-text {
        font-weight: bold; /* Negrita para resaltar */
        color: #000000; /* Color negro */
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.write("---")
    st.markdown("<h2 class='title-text'>Sobre nosotros 🔍</h2>", unsafe_allow_html=True)
    st.markdown(

        """
        <div class="big-text">
        El objetivo de este proyecto es desarrollar un sistema integral de análisis del mercado de restaurantes en el condado de Los Ángeles, C.A. orientado a inversores del sector de restaurantes.<br><br>

        <span class="highlight-text">Nuestra metodología:</span><br>
        - La predicción del Rating a partir de una reseña: usamos el método de Regresión Lógistica para demostrarte la calidad de investigación<br>
        - Sistema de recomendación de restaurants: Si sos usuario podes consultar nuestras recomendaciones de lugares<br>
        - Sistema de análisis de sentimientos: Ingresa un comentario para corroborar nuestras tecnologías de análisis de sentimientos<br>
        - Validación de reseñas inteligente: Ingresa una reseña y nuestros avanzados modelos de Machine Learning analizarán su objetividad, garantizando la autenticidad y confiabilidad de las opiniones<br>
        - Análisis Inteligente de Aspectos: Nuestro avanzado modelo de Machine Learning desglosa cada reseña para destacar los aspectos positivos y negativos de la comida, el servicio y la ambientación<br><br>


        <span class="highlight-text">Nuestro resultado:</span><br>
        - Identificación y Caracterización de Modelos de Negocio: Analizamos y definimos diversos modelos de negocio adaptados al sector restaurantero para optimizar la rentabilidad y sostenibilidad<br>
        - Evaluación de Restaurantes: Implementamos un sistema de evaluación basado en las valoraciones y opiniones de los clientes, proporcionando una visión integral y precisa de la calidad del servicio y la satisfacción del cliente<br>
        - Creamos y aplicamos un sistema de segmentación avanzada que permite categorizar a los clientes según sus preferencias y comportamientos, facilitando estrategias de marketing más efectivas y personalizadas<br><br>
        <span class="highlight-text">Si esto resulta interesante para ti puedes contactarnos a través del formulario que encontrarás al final de la página!</span>

        </div>
        """,
        unsafe_allow_html=True
    )


# Rating Prediction
# modelos
with st.container():
    st.write("---")
    st.markdown("<h1 class='header-text'>Modelos</h1>", unsafe_allow_html=True)
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        image = Image.open(image_6_path)
        st.image(image, use_column_width=True)
    with text_column:
        st.markdown("<h2 class='model-text'>Predicción de rating según la review</h2>",unsafe_allow_html=True)
        st.markdown(
            """
            <div class="big-text">

            Se utiliza un modelo de análisis NLP, para predecir el rating que el usuario le dara al establecimiento según la review escrita del mismo. Así si contamos con datos faltantes podemos utilizar este sistema para rellenar dicha información.

            </div>
            """
            ,unsafe_allow_html=True
        )
        st.markdown("<div class='big-text'>Introduce la reseña:</div>",unsafe_allow_html=True )
        review_input = st.text_input('', key= 'rating')
        if st.button("Predecir calificación"):
            rating = predict_rating(review_input)
            st.markdown(f"<p style='color: black;'>La calificación prevista para la Reseña es {rating}</p>", unsafe_allow_html=True)

# Recommendation System
with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        image = Image.open(image_2_path)
        st.image(image, use_column_width=True)
    with text_column:
        st.markdown("<h2 class='model-text'>Recomendación de restaurantes para usuarios</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="big-text">

            En este sistema, al ingresar el ID de un usuario, nuestro modelo arrojará 5 recomendaciones de restaurantes para el usuario. Restaurantes a los cuales el usuario aún no a dejado una review.
            
            </div>
            """
            , unsafe_allow_html=True
        )
        st.markdown("<div class='big-text'>Introduce el ID del usuario:</div>",unsafe_allow_html=True )
        user_id_input = st.text_input("", key= 'recomendacion')
        if st.button("Obtener recomendaciones"):
            recommendations = get_recommendations_for_user(user_id_input, trainset_recomendation, modelo_recomendation, n=5)
            st.markdown(f"<p style='color: black;'>Las 5 mejores recomendaciones para el usuario {user_id_input}:</p>", unsafe_allow_html=True)
            for restaurant_id, estimated_rating in recommendations:
                st.markdown(f"<p style='color: black;'>Restaurant: {restaurant_id}, Calificación estimada: {int(estimated_rating)}</p>", unsafe_allow_html=True)

# Sentiment Analysis
with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        image = Image.open(image_3_path)
        st.image(image, use_column_width=True)
    with text_column:
        st.markdown("<h2 class='model-text'>Análisis de sentimientos</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="big-text">

            Este modelo predice si el sentimiento de la reviews del usuario es bueno, neutro o malo. Con esto podemos seccionar en un futuro más nuestros datos.

            </div>
            """
            , unsafe_allow_html=True
        )
        st.markdown("<div class='big-text'>Introduce la reseña::</div>",unsafe_allow_html=True )
        review_sentiment_input = st.text_input("", key= 'sentimiento')
        if st.button("Analizar Sentimiento"):
            sentiment = analisis_sentimiento(review_sentiment_input)
            sentiment_text = "Neutro" if sentiment == 1 else ("Bueno" if sentiment == 2 else "Malo")
            st.write(f"<p style='color: black;'>El sentimiento previsto para la revisión es: {sentiment_text}</p>", unsafe_allow_html=True)

# Sentiment Analysis and Fake Review Detection
with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        image = Image.open(image_4_path)
        st.image(image, use_column_width=True)
    with text_column:
        st.markdown("<h2 class='model-text'>Autenticación de reseñas</h2>",unsafe_allow_html=True)
        st.markdown(
            """
            <div class="big-text">

            Este modelo analizará las resñas en busca de reseñas "Falsas" que puedan aportar información poco veridica sobre algun establecimiento. El modelo se basa en la busqueda de palabras superlativas para rechazar reseñas exageradamente buenas y malas.

            </div>
            """
            , unsafe_allow_html=True
        )
        st.markdown("<div class='big-text'>Introduce la reseña::</div>",unsafe_allow_html=True)
        review_sentiment_input = st.text_input("", key= 'fake')
        if st.button("Analizar Sentimiento y Detectar Falsedad"):
            sentiment = analisis_sentimiento(review_sentiment_input)
            st.markdown(f"<p style='color: black;'>El sentimiento previsto para la revisión es: {sentiment}</p>", unsafe_allow_html=True)
            is_fake = analisis_reseña_falsa(review_sentiment_input)
            if is_fake == 1:
                st.markdown("<p style='color: black;'>Esta reseña es sospechosa de ser falsa.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: black;'>Esta reseña parece ser genuina.</p>", unsafe_allow_html=True)

# Aspect Analysis
with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        image = Image.open(image_5_path)
        st.image(image, use_column_width=True)
    with text_column:
        st.markdown("<h2 class='model-text'>Análisis inteligente de aspectos</h2>", unsafe_allow_html=True)
        st.write(
            """
            <div class="big-text">

            Nuestro modelo analizará la reseña en busca de tres topicos diferentes (servicio, comida y ambiente del lugar). Según la reseña clasificara si el ambiente fue bueno o no, si la comida fue buena o no y si el servicio fue bueno o no. Esto nos ayudara a segmentar aún más nuestros datos para un analisis más potente.

            </div>
            """
            , unsafe_allow_html=True 
        )
        st.markdown("<div class='big-text'>Introduce la reseña:</div>",unsafe_allow_html=True)
        review_aspect_input = st.text_input("", key= 'service')
        if st.button("Analizar Aspectos"):
            aspects = analisis_aspectos_reseña(review_aspect_input)
            st.markdown("<p style='color: black;'>Análisis de aspectos de la reseña:</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Servicio: {'Positivo' if aspects[0][0] else 'Negativo'}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Comida: {'Positivo' if aspects[0][1] else 'Negativo'}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Ambiente: {'Positivo' if aspects[0][2] else 'Negativo'}</p>", unsafe_allow_html=True)

with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        image = Image.open(image_1_path)
        st.image(image, use_column_width=True)
    with text_column:
        st.markdown("<h2 class='model-text'>Visualización de datos</h2>", unsafe_allow_html=True)
        st.write(
            """
            <div class="big-text">

            Si sientes que no tienes una visión clara de datos de tu negocio lo que necesitas es una aplicación en la que puedas tener toda la información de interes de tu empresa. Aquí te presentamos el MVP de un dashboard que podría ser tuyo!

            </div>
            """
            , unsafe_allow_html=True
        )
        # Enlace de la visualización de Looker Studio
        # Enlace de la visualización de Looker Studio
        #nuevo dashoard
        #looker_studio_url = "https://lookerstudio.google.com/u/0/reporting/a2b77bd1-2428-4682-be09-56533a248a94/page/p_pewigcs7id"
        looker_studio_url = "https://lookerstudio.google.com/embed/reporting/92454d30-31b7-44b8-97ea-6e915c49483e/page/p_qndu6slwid"
        

        # Utilizar un iframe para incrustar la visualización
        st.markdown(f'<iframe width="100%" height="600" src="{looker_studio_url}" frameborder="0" style="border:0" allowfullscreen></iframe>', unsafe_allow_html=True)

# contacto
with st.container():
    st.write("---")
    st.markdown("<h2 class='model-text'>Ponte en contacto con nosotros! 📧</h2>", unsafe_allow_html=True)
    st.write("##")
    contact_form = f"""
    <form action="https://formsubmit.co/{email_address}" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Tu nombre" required>
        <input type="email" name="email" placeholder="Tu email" required>
        <textarea name="message" placeholder="Tu mensaje aquí" required></textarea>
        <button type="submit">Enviar</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()