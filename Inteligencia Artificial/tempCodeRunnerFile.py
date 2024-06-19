    """Se definen los parámetros del modelo, incluyendo el tamaño del vocabulario, la dimensión de los embeddings y las unidades LSTM"""
    tamano_vocabulario = len(indice_palabras) + 1
    dimension_embedding = 100
    unidades_lstm = 128