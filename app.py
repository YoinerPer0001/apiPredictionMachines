from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import asyncpg
from sqlalchemy.orm import Session
from db import SessionLocal, engine
import models
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid
import datetime
import asyncio

db_conn = None

now = datetime.datetime.utcnow()

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

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

@app.on_event("startup")
async def listen_pg_notify():
    global db_pool
    db_pool = await asyncpg.create_pool(dsn=DATABASE_URL)
    conn = await db_pool.acquire()
    await conn.add_listener("nueva_lectura", handle_notification)
    print("🟢 Escuchando canal 'nueva_lectura'...")

async def handle_notification(conn, pid, channel, payload):
    # Procesa en segundo plano para no bloquear el listener
    asyncio.create_task(procesar_lectura(payload))

async def procesar_lectura(payload):
    lectura_id = uuid.UUID(payload)
    now = datetime.utcnow()

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM lecturas
            INNER JOIN sensores ON lecturas.sensor_id = sensores.id
            INNER JOIN nodos ON sensores.nodo_id = nodos.id
            INNER JOIN maquinas ON nodos.maquina_id = maquinas.id
            WHERE lecturas.id = $1
        """, lectura_id)

        if not row:
            print(f"Lectura con ID {lectura_id} no encontrada.")
            return

        data = InputData(
            valor=row['valor'],
            sensor=row['tipo'],
            maquina=row['modelo']
        )

        prediccion = predecir(data)

        # Actualiza la etiqueta
        await conn.execute(
            "UPDATE lecturas SET etiqueta = $1 WHERE id = $2;",
            prediccion["etiqueta"], lectura_id
        )

        if prediccion["prediccion"] == 1:
            # Normal: eliminar alerta si existe
            exist = await conn.fetchrow("SELECT * FROM alertas WHERE sensor_id = $1", row["sensor_id"])
            if exist:
                await conn.execute("DELETE FROM alertas WHERE sensor_id = $1", row["sensor_id"])
                print(" Alerta eliminada")
            else:
                print("No existe alerta para eliminar")
        else:
            # Mantenimiento: crear alerta si no existe
            exist = await conn.fetchrow("SELECT * FROM alertas WHERE sensor_id = $1", row["sensor_id"])
            if not exist:
                await conn.execute("""
                    INSERT INTO alertas (id, sensor_id, descripcion, nivel, createdAt, updatedAt)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, uuid.uuid4(), row["sensor_id"], "Revisar equipo mantenimiento", "media", now, now)
                print(" Alerta creada")
            else:
                print("Alerta ya existe")

        print(f"Procesada lectura: {lectura_id}")
    
@app.get("/lecturas")
def leer_lecturas(db: Session = Depends(get_db)):
    return db.query(models.Lectura).all()

# @app.post("/predecir")
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

