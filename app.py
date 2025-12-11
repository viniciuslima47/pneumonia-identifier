import streamlit as st
import onnxruntime as ort
import numpy as np
import sqlite3
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# --- Configurações do Repositório ---
# Substitua pelo SEU REPOSITÓRIO (se for diferente)
REPO_ID = "viniciuslima47/pneumonia-vgg-model" 
MODEL_FILENAME = "modelo_pneumonia.onnx" # O arquivo ONNX

# --- Configuração do Banco de Dados SQLite ---
# Este banco de dados 'usos.db' será criado automaticamente no Space
conn = sqlite3.connect("usos.db")
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS historico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resultado TEXT,
                prob REAL,
                filename TEXT
              )""")
conn.commit()

# --- Função de Carregamento do Modelo (ONNX) ---
@st.cache_resource
def load_onnx_model():
    """Baixa e carrega o modelo ONNX usando onnxruntime."""
    try:
        st.info(f"Baixando modelo {MODEL_FILENAME}...")
        
        # 1. Baixa o arquivo ONNX do Hugging Face Hub (usando cache)
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        
        # 2. Inicia a sessão de inferência do ONNX
        session = ort.InferenceSession(model_path)
        st.success("Modelo ONNX carregado com sucesso!")
        return session
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ONNX: {e}")
        st.caption(f"Verifique o REPO_ID ({REPO_ID}), se o arquivo ONNX está no repo, e se o setup.sh foi executado.")
        return None

# Carrega o modelo na inicialização
session = load_onnx_model()

# --- Interface do Streamlit ---
st.title("Classificador de Pneumonia (Versão ONNX)")
st.markdown("Diagnóstico de Raio-X com alto desempenho e estabilidade.")

if session is None:
     st.error("O aplicativo não pode rodar. O modelo falhou ao carregar.")
else:
    uploaded = st.file_uploader("Envie uma imagem de Raio-X (peito)", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption='Imagem Enviada', width=350)
        
        if st.button("Analisar Raio-X"):
            with st.spinner('Processando imagem...'):
                try:
                    # 1. Pré-processamento (Igual ao VGG16)
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized).astype(np.float32)
                    img_array /= 255.0 # Normalização
                    img_array = np.expand_dims(img_array, axis=0) # Adiciona dimensão de batch

                    # 2. Inferência via ONNX Runtime
                    input_name = session.get_inputs()[0].name
                    outputs = session.run(None, {input_name: img_array})
                    
                    # Assume saída de uma única probabilidade (0 a 1)
                    prob = float(outputs[0][0][0])
                    
                    # 3. Definição do Resultado
                    pred = "Pneumonia" if prob > 0.5 else "Normal"
                    
                    st.success("Análise concluída!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Diagnóstico:")
                        if pred == "Pneumonia":
                            st.error(f"**{pred}**")
                        else:
                            st.success(f"**{pred}**")
                    with col2:
                        st.metric(label="Confiança do Modelo", value=f"{prob:.4f}")

                    # 4. Salvar no Banco de Dados SQLite
                    cur.execute("INSERT INTO historico (resultado, prob, filename) VALUES (?, ?, ?)",
                                (pred, prob, uploaded.name))
                    conn.commit()

                except Exception as e:
                    st.error(f"Erro durante o processamento (Inferência): {e}")

# --- Seção de Histórico ---
st.divider()
if st.checkbox("Mostrar Histórico de Uso"):
    st.markdown("### Registros do Banco de Dados")
    data = cur.execute("SELECT * FROM historico ORDER BY id DESC").fetchall()
    st.dataframe(data, use_container_width=True, hide_index=True,
                 column_config={
                     0: "ID", 1: "Resultado", 2: "Probabilidade", 3: "Nome do Arquivo"
                 })
