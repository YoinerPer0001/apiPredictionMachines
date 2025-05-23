from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pickle
import pandas as pd
from db import SessionLocal
from tensorflow.keras.models import load_model
from sqlalchemy.orm import Session
from models import Lectura
from sqlalchemy import text
import json
from apscheduler.schedulers.background import BackgroundScheduler
from uuid import uuid4
from datetime import datetime




db_conn = None
db_pool = None

def iniciar_cron():
    scheduler = BackgroundScheduler()
    
    def tarea():
        db = SessionLocal()
        try:
            procesar_lecturas(db)
        finally:
            db.close()
    
    scheduler.add_job(tarea, 'interval', seconds= 5)
    scheduler.start()

# Llama al iniciar la app
iniciar_cron()

# Dependencia para obtener sesión DB por solicitud
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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


def procesar_lecturas(db: Session = Depends(get_db)):
    print("Se llamo")
    query = text('''
                SELECT 
                lecturas.id AS lectura_id,
                sensores.id AS sensor_id,
                sensores.tipo AS tipo_sensor,
                sensores.unidad AS unidad,
                nodos.id AS nodo_id,
                maquinas.id AS maquina_id,
                maquinas.modelo AS modelo,
                lecturas.etiqueta,
                lecturas.valor,
                lecturas.timestamp
            FROM lecturas
            INNER JOIN sensores ON lecturas.sensor_id = sensores.id
            INNER JOIN nodos ON sensores.nodo_id = nodos.id
            INNER JOIN maquinas ON nodos.maquina_id = maquinas.id
            WHERE etiqueta IS NULL
            ORDER BY lecturas.timestamp ASC
            LIMIT 5;
        ''')
    lecturas = db.execute(query).fetchall()
    resultados = []

    for lectura in lecturas:
        # convertir Row a dict
        lectura_dict = dict(lectura._mapping)  
        print(lectura)
        prediccion = predecir(InputData(valor= lectura.valor, sensor=lectura.tipo_sensor, maquina=lectura.modelo))
        print(prediccion["etiqueta"])
        resultados.append(lectura_dict)
        
        db.execute(text('UPDATE lecturas SET etiqueta = :etiqueta WHERE id = :id'),{"etiqueta": prediccion["etiqueta"], "id": lectura.lectura_id})
        db.commit()
        
        if prediccion["prediccion"] == 0:
            db.execute(text('''
                INSERT INTO alertas ("id", "descripcion", "nivel", "createdAt", "updatedAt", "sensor_id")
                VALUES (:id, :descripcion, :nivel, :createdAt, :updatedAt, :sensor_id)
            '''), {
                "id": str(uuid4()),
                "descripcion": "Valor medido: " + lectura.valor +" "+ lectura.unidad,
                "nivel": "media",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "sensor_id": lectura.sensor_id
            })
            db.commit()
        
        
        
    return resultados
    


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

