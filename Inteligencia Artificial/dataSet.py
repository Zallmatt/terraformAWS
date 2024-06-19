import os 
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataSet:
    def loadDataSet():  
        directorio_desagregado = os.path.dirname(os.path.abspath(__file__))
        ruta_emociones = os.path.join(directorio_desagregado, 'files', 'emociones.csv')
        ruta_recomendaciones = os.path.join(directorio_desagregado, 'files', 'recomendaciones.csv')
        ruta_interacciones = os.path.join(directorio_desagregado, 'files', 'interacciones.csv')
        emociones_df = pd.read_csv(ruta_emociones, encoding='latin1')
        recommendations_df = pd.read_csv(ruta_recomendaciones, encoding='latin1')
        interactions_df = pd.read_csv(ruta_interacciones, encoding='latin1')
        return emociones_df, recommendations_df, interactions_df
    