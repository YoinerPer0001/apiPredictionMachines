import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# Cargar encoders y scaler
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

# Nuevo dato
nuevo_dato = {'valor': 2.7, 'sensor': 'Vibración', 'maquina': 'Greenheck SQ Series'}
df_nuevo = pd.DataFrame([nuevo_dato])

# Codificación
df_nuevo.loc[:, 'sensor_encoded'] = le_sensor.transform(df_nuevo['sensor'])
df_nuevo.loc[:, 'maquina_encoded'] = le_maquina.transform(df_nuevo['maquina'])

# Escalado
X_nuevo = df_nuevo[['valor', 'sensor_encoded', 'maquina_encoded']]
X_nuevo_scaled = scaler.transform(X_nuevo)

# Predicción
pred = modelo.predict(X_nuevo_scaled)

# Interpretar resultado
clase_predicha = (pred > 0.5).astype(int)
if(clase_predicha[0][0] == 0):
    print("Mantenimiento")
else:
    print("Normal")

print("Probabilidad:", pred)
print("Predicción:", clase_predicha[0][0])
