import gradio as gr
import pandas as pd
import random
import numpy as np
import nltk
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from dataSet import DataSet
import tensorflow as tf
from tensorflow.keras import backend as K

if __name__ == "__main__":

    # Se descargan los recursos necesarios de NLTK para tokenización de palabras y lista de palabras vacías (stopwords).
    nltk.download('punkt')
    nltk.download('stopwords')

    # Se carga un archivo CSV con datos de emociones y se define una función para preprocesar el texto: tokenización, conversión a minúsculas, eliminación de signos de puntuación y palabras vacías.
    DataSet.loadDataSet()
    emociones_df, recommendations_df, interactions_df = DataSet.loadDataSet()

    # Preprocesamiento del texto del dataset
    stop_words = set(stopwords.words('spanish'))

    def preprocesar_texto(texto):
        tokens = word_tokenize(texto.lower())
        tokens = [palabra for palabra in tokens if palabra.isalnum() and palabra not in stop_words]
        return tokens

    emociones_df['tokens'] = emociones_df['text'].apply(preprocesar_texto)

    # Convertir las palabras en secuencias numéricas y se obtiene un índice de palabras. Convertir tokens a secuencias
    tokenizador = Tokenizer()
    tokenizador.fit_on_texts(emociones_df['tokens'])
    secuencias = tokenizador.texts_to_sequences(emociones_df['tokens'])
    indice_palabras = tokenizador.word_index

    # Se realiza padding para asegurar que todas las secuencias tengan la misma longitud
    longitud_maxima = max(len(seq) for seq in secuencias)
    secuencias_padded = pad_sequences(secuencias, maxlen=longitud_maxima, padding='post')

    # Las etiquetas (emociones) se convierten a formato one-hot encoding para su uso en el modelo
    from sklearn.preprocessing import LabelEncoder
    codificador_etiquetas = LabelEncoder()
    etiquetas_integer = codificador_etiquetas.fit_transform(emociones_df['emotion'])
    etiquetas_one_hot = to_categorical(etiquetas_integer)

    # Se definen los parámetros del modelo, incluyendo el tamaño del vocabulario, la dimensión de los embeddings y las unidades LSTM
    tamano_vocabulario = len(indice_palabras) + 1
    dimension_embedding = 100
    unidades_lstm = 128

    # Definición de F1-score
    def f1_score(y_true, y_pred):
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        return f1

    # Se construye un modelo secuencial con una capa de embedding, 
    # una capa LSTM con dropout, una capa densa adicional con activación
    # relu y una capa densa con activación softmax para la clasificación
    modelo = Sequential()
    modelo.add(Embedding(input_dim=tamano_vocabulario, output_dim=dimension_embedding, input_length=longitud_maxima))
    modelo.add(SpatialDropout1D(0.2))
    modelo.add(LSTM(unidades_lstm, dropout=0.2, recurrent_dropout=0.2))
    modelo.add(Dense(64, activation='relu'))  # Capa adicional
    modelo.add(Dense(4, activation='sigmoid'))  # Cambiada a sigmoid

    # El modelo se compila con pérdida de entropía cruzada categórica y el optimizador Adam. Luego, se entrena el modelo con las secuencias preprocesadas y las etiquetas codificadas.
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])

    # Entrenar el modelo
    modelo.fit(secuencias_padded, etiquetas_one_hot, epochs=50, batch_size=32, validation_split=0.2)

    def recomendar_actividad(emocion):
        # Filtrar emoción
        recommendations_df_filtradas = recommendations_df[recommendations_df['emotion'] == emocion]

        # Si no hay recommendations_df para la emoción dada, devolver False
        if recommendations_df_filtradas.empty:
            return 'False'

        # Convertir las recommendations_df filtradas en una lista
        return random.choice(recommendations_df_filtradas['text'].tolist())

    # Se define una función para predecir la emoción de un texto dado utilizando el modelo entrenado
    def predecir_emocion(texto, umbral=0.92):
        tokens = preprocesar_texto(texto)
        secuencia = tokenizador.texts_to_sequences([tokens])
        secuencia_padded = pad_sequences(secuencia, maxlen=longitud_maxima, padding='post')
        prediccion = modelo.predict(secuencia_padded)
        max_probabilidad = np.max(prediccion)
        if max_probabilidad < umbral:
            return None
        emocion_predicha = codificador_etiquetas.inverse_transform([np.argmax(prediccion)])
        return emocion_predicha[0]

    # Cargar CSV de interacciones y respuestas
    vectorizador = TfidfVectorizer()

    # Esta función obtiene una respuesta basada en la similitud de texto entre la entrada del usuario y las interacciones almacenadas
    def obtener_respuesta_interaccion(entrada_usuario):
        interacciones = interactions_df['entrada'].tolist()
        respuestas = interactions_df['respuesta'].tolist()

        X = vectorizador.fit_transform(interacciones)
        vector_usuario = vectorizador.transform([entrada_usuario])

        similitudes = cosine_similarity(vector_usuario, X)
        max_similitud = max(similitudes[0])

        if max_similitud > 0.8:  # Umbral de similitud
            indice = similitudes[0].tolist().index(max_similitud)
            return respuestas[indice]
        else:
            return None

    # Esta función utiliza primero el sistema experto para encontrar una respuesta. Si no encuentra una respuesta adecuada, predice la emoción y recomienda una actividad basada en la emoción detectada.
    def respuesta_chatbot(mensaje, historial):
        # Primero intentar encontrar una respuesta en el sistema experto
        respuesta = obtener_respuesta_interaccion(mensaje)
        if not respuesta:  # Si no hay una respuesta en el sistema experto
            emocion_predicha = predecir_emocion(mensaje)
            if emocion_predicha:  # Si se detecta una emoción
                recomendacion_actividad = recomendar_actividad(emocion_predicha)
                if recomendacion_actividad != 'False':  # Si hay una recomendación
                    respuesta = f"{recomendacion_actividad}"
                else:
                    respuesta = "No entiendo. Por favor, escribe otra cosa."
            else:
                respuesta = "No entiendo. Por favor, escribe otra cosa."

        historial.append((mensaje, respuesta))
        return historial, historial

    # Crear la interfaz con Gradio
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def responder(mensaje, historial_chat):
            mensaje_bot, historial_chat = respuesta_chatbot(mensaje, historial_chat)
            return "", historial_chat

        msg.submit(responder, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(share=True)

