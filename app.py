import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import io

# --- Configurações do Modelo e Banco de Dados ---
# Seu link do Google Drive: https://drive.google.com/file/d/1_nT9nibatPXxcqEKFocFClpSeST5mUH9/view?usp=sharing
# É necessário converter o link para um link de download direto.
# A estrutura para um link de download direto do Drive (substituindo o ID) é:
MODEL_ID = "1_nT9nibatPXxcqEKFocFClpSeST5mUH9"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
MODEL_FILENAME = "modelo_vgg.h5"


# 1. Função para Baixar o Modelo
@st.cache_resource
def load_and_cache_model():
    """Baixa o modelo do Drive (se não existir) e o carrega."""
    
    if not os.path.exists(MODEL_FILENAME):
        st.info("Baixando o modelo grande do Google Drive pela primeira vez... Isso pode levar alguns segundos.")
        
        try:
            # Uso de 'stream=True' para lidar com arquivos grandes de forma eficiente
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Lança exceção para status 4xx/5xx

            # Salva o arquivo em disco
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                 st.error("Erro ao baixar. O link pode não ser de acesso público/direto.")
                 return None

            with open(MODEL_FILENAME, "wb") as f:
                # Usa st.progress para mostrar o status do download
                progress_bar = st.progress(0, text=f"Baixando {MODEL_FILENAME}...")
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = min(int((downloaded_size / total_size) * 100), 100)
                    progress_bar.progress(progress, text=f"Baixando {MODEL_FILENAME} ({progress}%)")
                progress_bar.empty()
            
            st.success("Modelo baixado com sucesso!")

        except requests.exceptions.RequestException as e:
            st.error(f"Erro de conexão ao tentar baixar o modelo: {e}")
            return None
        
    try:
        # Carrega o modelo localmente (agora ele existe no disco)
        model = tf.keras.models.load_model(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo Keras: {e}")
        return None

# Carrega o modelo. Esta função será executada apenas uma vez.
model = load_and_cache_model()

# --- Configuração do Banco de Dados ---
# O banco de dados SQLite funciona bem no Streamlit Cloud, mas pode ser redefinido
# se o aplicativo for reiniciado (dependendo da configuração do Streamlit Cloud).
conn = sqlite3.connect("usos.db")
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resultado TEXT,
                prob REAL,
                filename TEXT
              )""")
conn.commit()


# --- Interface do Streamlit ---
st.title("Classificador de Pneumonia em Raio-X")

if model is not None:
    uploaded = st.file_uploader("Envie uma imagem de Raio-X (peito)", type=["jpg","png","jpeg"])

    if uploaded:
        # Carregamento e exibição da imagem
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption='Imagem Enviada', use_column_width=True)
        st.divider()

        # Pré-processamento e previsão (usando o VGG com 224x224)
        with st.spinner('Realizando a predição...'):
            img_resized = img.resize((224,224))
            arr = np.array(img_resized)/255.0
            arr = np.expand_dims(arr, axis=0)

            # O modelo provavelmente retorna a probabilidade da primeira classe (Pneumonia, se for binário)
            prob = float(model.predict(arr)[0][0])
            pred = "Pneumonia" if prob > 0.5 else "Normal"
            
            st.success("Predição concluída!")

        # Exibição dos resultados
        st.subheader(f"Diagnóstico: **{pred}**")
        st.metric(label="Probabilidade de Pneumonia (Classe 1)", value=f"{prob:.4f}")
        
        # Inserção no histórico
        cur.execute("INSERT INTO historico (resultado, prob, filename) VALUES (?, ?, ?)",
                    (pred, prob, uploaded.name))
        conn.commit()
        
        # Opção de ver o histórico (extra)
        if st.checkbox("Mostrar Histórico de Uso"):
            st.dataframe(cur.execute("SELECT * FROM historico ORDER BY id DESC").fetchall(),
                         column_names=["ID", "Resultado", "Prob.", "Nome do Arquivo"])

else:
    st.error("O modelo não pôde ser carregado. Verifique o link de download direto do Google Drive e as permissões de acesso.")
