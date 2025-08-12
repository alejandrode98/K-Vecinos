import pandas as pd
from utils import db_connect

engine = db_connect()

url = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
df = pd.read_csv(url, sep=';')

df.columns = df.columns.str.replace(' ', '_')

try:
    df.to_sql('winequality_red', engine, if_exists='replace', index=False)
    print("Tabla 'winequality_red' creada y datos cargados exitosamente.")
except Exception as e:
    print(f"Ocurrió un error al cargar los datos en la base de datos: {e}")