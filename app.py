import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# --- Configurações do Modelo e Banco de Dados ---
REPO_ID = "viniciuslima47/pneumonia-vgg-model" # Seu repositório no Hugging Face
MODEL_FILENAME = "final_vgg16_model.h5"
# MODEL_PATH não é mais necessário aqui, pois hf_hub_download retorna o caminho

# 1. Função para Baixar e Carregar o Modelo
# O st.cache_resource garante que o download e o carregamento do modelo só ocorram uma vez.
@st.cache_resource
def load_and_cache_model():
    """Baixa o modelo do Hugging Face Hub (se não existir) e o carrega."""
    
    # 1. Obter o Token
    # O HF_TOKEN deve ser configurado nas Streamlit Secrets
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except KeyError:
        st.error("Erro: A chave secreta 'HF_TOKEN' não está configurada no Streamlit Secrets.")
        return None

    st.info(f"Baixando e carregando o modelo grande do Hugging Face Hub ({MODEL_FILENAME})...")
    
    try:
        # 2. Baixar o arquivo e armazená-lo em cache localmente no sistema do Streamlit Cloud
        # O hf_hub_download gerencia o cache. Ele só baixa se não existir.
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            token=hf_token  # Usa o token para acesso
        )
        st.success("Download do modelo concluído. Carregando Keras...")

        # 3. Carregar o modelo Keras a partir do caminho baixado
        # 
        model = tf.keras.models.load_model(downloaded_path)
        return model
        
    except Exception as e:
        # Erro genérico de download/carregamento
        st.error(f"Erro ao baixar ou carregar o modelo. Verifique o REPO_ID, o nome do arquivo e o token.")
        st.code(f"Detalhes do Erro: {e}")
        return None

# Carrega o modelo.
model = load_and_cache_model()

# --- Configuração do Banco de Dados ---
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
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption='Imagem Enviada', use_column_width=True)
        st.divider()

        with st.spinner('Realizando a predição...'):
            img_resized = img.resize((224,224))
            arr = np.array(img_resized)/255.0
            arr = np.expand_dims(arr, axis=0)

            prob = float(model.predict(arr)[0][0])
            pred = "Pneumonia" if prob > 0.5 else "Normal"
            
            st.success("Predição concluída!")

        # Exibição dos resultados
        st.subheader(f"Diagnóstico: **{pred}**")
        st.metric(label="Probabilidade de Pneumonia (Classe 1)", value=f"{prob:.4f}")
        
        # Enviando para o banco de dados
        cur.execute("INSERT INTO historico (resultado, prob, filename) VALUES (?, ?, ?)",
                    (pred, prob, uploaded.name))
        conn.commit()
        
        # Mostrar Histórico
        if st.checkbox("Mostrar Histórico de Uso"):
            st.dataframe(cur.execute("SELECT * FROM historico ORDER BY id DESC").fetchall(),
                         column_names=["ID", "Resultado", "Prob.", "Nome do Arquivo"])

else:
    st.error("O modelo não pôde ser carregado.")
