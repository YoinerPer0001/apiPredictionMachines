from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pickle
import pandas as pd
from tensorflow.keras.models import load_model


# db_conn = None
# db_pool = None

# now = datetime.datetime.utcnow()

# load_dotenv()

# DATABASE_URL = os.getenv("DATABASE_URL")

# # Dependencia para obtener sesión DB por solicitud
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# Definir modelo de datos de entrada
class InputData(BaseModel):
    valor: float
    sensor: str
    maquina: str

app = FastAPI()

# Cargar encoders y scaler una vez al iniciar la API
with open('./dataModelNeuronal/le_sensor.pkl', 'rb') as f:
    le_sensor = pickle.load(f)

with open('./dataModelNeuronal/le_maquina.pkl', 'rb') as f:
    le_maquina = pickle.load(f)

with open('./dataModelNeuronal/le_etiqueta.pkl', 'rb') as f:
    le_etiqueta = pickle.load(f)

with open('./dataModelNeuronal/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Cargar modelo
modelo = load_model("./dataModelNeuronal/modelo_mantenimiento.keras")

@app.get("/")
def home():
    return {"mensaje": "API corriendo en Render 🎉"}

@app.post("/predecir")
def predecir(data: InputData):
    # Convertir entrada a DataFrame
    df_nuevo = pd.DataFrame([data.dict()])

    # Codificar variables categóricas
    try:
        df_nuevo['sensor_encoded'] = le_sensor.transform(df_nuevo['sensor'])
        df_nuevo['maquina_encoded'] = le_maquina.transform(df_nuevo['maquina'])
    except Exception as e:
        return {"error": f"Error en la codificación: {str(e)}"}

    # Preparar datos para el modelo
    X_nuevo = df_nuevo[['valor', 'sensor_encoded', 'maquina_encoded']]
    X_nuevo_scaled = scaler.transform(X_nuevo)

    # Realizar predicción
    pred = modelo.predict(X_nuevo_scaled)
    clase_predicha = (pred > 0.5).astype(int)[0][0]
    probabilidad = float(pred[0][0])

    # Mapear etiqueta para mayor claridad
    etiqueta = "Mantenimiento" if clase_predicha == 0 else "Normal"

    return {
    "prediccion": int(clase_predicha),        # <- convertir a int nativo
    "etiqueta": etiqueta,
    "probabilidad": float(probabilidad)       # <- asegurar float nativo (ya lo tienes)
    }

