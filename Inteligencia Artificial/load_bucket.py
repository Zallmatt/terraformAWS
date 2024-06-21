import os
import pandas as pd
import boto3
from io import StringIO

class DataSet:
    def loadDataSet():
        # Credenciales de AWS
        aws_access_key_id = 'aca van las claves'
        aws_secret_access_key = 'aca van las claves'
        bucket_name = 'ucp-bot-rn'

        # Crear una sesión de boto3
        s3 = boto3.client('s3',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)

        # Leer archivos desde S3
        def read_s3_csv(file_key):
            obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            return pd.read_csv(StringIO(obj['Body'].read().decode('latin1')))

        # Rutas de los archivos en el bucket S3
        ruta_emociones = 'emociones.csv'
        ruta_recomendaciones = 'recomendaciones.csv'
        ruta_interacciones = 'interacciones.csv'

        emociones_df = read_s3_csv(ruta_emociones)
        recommendations_df = read_s3_csv(ruta_recomendaciones)
        interactions_df = read_s3_csv(ruta_interacciones)
        

        return emociones_df, recommendations_df, interactions_df
