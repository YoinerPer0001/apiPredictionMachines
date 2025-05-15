# models.py
from sqlalchemy import Column, Integer, String, Float, UUID, DATE
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Lectura(Base):
    __tablename__ = "lecturas"
    id = Column(UUID, primary_key=True, index=True)
    valor = Column(Float)
    timestamp = Column(DATE)
    etiqueta = Column(String)
    sensor_id = Column(Integer)
