import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("modelo_vgg.h5")

conn = sqlite3.connect("usos.db")
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resultado TEXT,
                prob REAL,
                filename TEXT
              )""")
conn.commit()

st.title("Classificador de Pneumonia em Raio-X")

uploaded = st.file_uploader("Envie uma imagem", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)

    img_resized = img.resize((224,224))
    arr = np.array(img_resized)/255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr)[0][0])
    pred = "Pneumonia" if prob > 0.5 else "Normal"

    st.subheader(f"Resultado: {pred}")
    st.write(f"Probabilidade: {prob:.4f}")

    cur.execute("INSERT INTO historico (resultado, prob, filename) VALUES (?, ?, ?)",
                (pred, prob, uploaded.name))
    conn.commit()
